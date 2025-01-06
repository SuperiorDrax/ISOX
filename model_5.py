'''
Ablation study part 5.
Transformer + Pixel Shuffle + SwiGLU + Cross-scale Embedding + AFNO
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

class ISOX(nn.Module):
    def __init__(self, input_time, output_num):
        super(ISOX, self).__init__()
        # Drop path rate is linearly increased as the depth increases
        # No more drop path
        # drop_path_list = torch.linspace(0, 0.2, 8)
        self.input_time = input_time
        self.output_num = output_num

        # Patch embedding
        self._input_layer = PatchEmbedding((2, 4, 8, 16), self.input_time, 64)

        # Four basic layers
        self.var_num = 36
        self.layer1 = EarthSpecificLayer(4, 64, 8)
        self.layer2 = EarthSpecificLayer(6, 128, 16)
        self.layer3 = EarthSpecificLayer(6, 128, 16)
        self.layer4 = EarthSpecificLayer(4, 64, 8)

        # Upsample and downsample
        self.downsample = DownSample(64)
        self.upsample = UpSample(128, 64)

        # Patch Recovery
        self._output_layer = PatchRecovery(2, self.output_num, 128)
      
    def forward(self, input):
        '''Backbone architecture'''
        # Embed the input fields into patches
        x = self._input_layer(input)

        # Encoder, composed of two layers
        # Layer 1, shape (40, 38, 72, C), C = 128
        x = self.layer1(x, 36, 38, 72)

        # Store the tensor for skip-connection
        skip = x

        # Downsample from (40, 38, 72) to (40, 19, 36)
        x = self.downsample(x, 36, 38, 72)

        # Layer 2, shape (20, 19, 36, 2C), C = 128
        x = self.layer2(x, 36, 19, 36)
        
        # Decoder, composed of two layers
        # Layer 3, shape (40, 19, 36, 2C), C = 128
        x = self.layer3(x, 36, 19, 36)

        # Upsample from (40, 19, 36) to (40, 38, 72)
        x = self.upsample(x)

        # Layer 4, shape (40, 38, 72, C), C = 128
        x = self.layer4(x, 36, 38, 72)

        # Skip connect, in last dimension(C from 128 to 256)
        x = torch.cat([skip, x],dim=2)

        # Recover the output fields from patches
        output = self._output_layer(x, 36, 38, 72)
        return output

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, input_time, total_dim):
        super(PatchEmbedding, self).__init__()
        '''Patch embedding operation'''
        # Here we use convolution to partition data into cubes
        # The coming data is (B,4,40,73,144) where 4 is incoming time, see as channels
        # The final three of 39 is constants: lsm, topo and coslat.
        self.patch_size = patch_size
        self.input_time = input_time
        self.total_dim = total_dim
        self.projs_pressure = nn.ModuleList()
        self.projs_const = nn.ModuleList()

        for i, ps in enumerate(self.patch_size):
            if i == len(patch_size) - 1:
                dim = self.total_dim // 2 ** i
            else:
                dim = self.total_dim // 2 ** (i + 1)
            stride = (1, self.patch_size[0], self.patch_size[0])
            # No padding here! Padding is done in the forward part.
            self.projs_pressure.append(nn.Conv3d(self.input_time, dim, kernel_size=(1, ps, ps), stride=stride))
            self.projs_const.append(nn.Conv3d(1, dim, kernel_size=(1, ps, ps), stride=stride))
        
        # Crossformer++ uses a layer norm here, so I copied
        self.norm = nn.LayerNorm(total_dim)

    def forward(self, input):
        input_pressure=input[:,:,:-3,:,:]
        input_const=input[:,:,-3:,:,:].mean(axis=1, keepdim=True)

        # patch_size = (2, 4, 8, 16), in accordance with different shape of kernel size
        # From (B,4,37,73,144)/(B,1,3,73,144) to (B,128,37,38,72)/(B,128,3,38,72)
        # Finally (B,128,40,38,72)
        pressure_list = []
        for i in range(len(self.projs_pressure)):
            # Manually pad first, use circular in the longitudinal dimension
            # The padding size should +2/+1 in the latitudinal diemnsion if you want to get a 38 in output
            padding = (self.patch_size[i] - self.patch_size[0]) // 2
            conv_this = F.pad(input_pressure, (padding, padding, 0, 0, 0, 0), mode='circular')
            conv_this = F.pad(conv_this, (0, 0, padding+2, padding+1, 0, 0), mode='constant', value=0)

            pressure = self.projs_pressure[i](conv_this)
            pressure_list.append(pressure)
        pressure_list = torch.cat(pressure_list, dim=1)

        const_list = []
        for i in range(len(self.projs_const)):
            # Manually pad first, use circular in the longitudinal dimension
            # The padding size should +2/+1 in the latitudinal diemnsion if you want to get a 38 in output
            padding = (self.patch_size[i] - self.patch_size[0]) // 2
            conv_this = F.pad(input_const, (padding, padding, 0, 0, 0, 0), mode='circular')
            conv_this = F.pad(conv_this, (0, 0, padding+2, padding+1, 0, 0), mode='constant', value=0)

            const = self.projs_const[i](conv_this)
            const_list.append(const)
        const_list = torch.cat(const_list, dim=1)

        x = torch.cat([pressure_list, const_list], dim=2)

        # Reshape x for calculation of linear projections
        # From (B,128,40,38,72) to (B,40*38*72,128)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(x.shape[0], 36*38*72, x.shape[-1])

        x = self.norm(x)
        return x
    
class PatchRecovery(nn.Module):
    def __init__(self, patch_scale, output_num, dim):
        # Dim = 256
        super(PatchRecovery, self).__init__()
        '''Patch recovery operation'''
        # Hear we use two transposed convolutions to recover data
        self.output_num = output_num
        self.patch_scale = patch_scale
        #self.conv = nn.ConvTranspose2d(in_channels=dim*36, out_channels=self.output_num, kernel_size=patch_size[1:], stride=patch_size[1:])
        # We use Pixel Shuffle here in substitute of Conv Transpose
        self.conv = nn.Conv2d(in_channels=dim*36, out_channels=self.output_num*self.patch_scale**2, kernel_size=(1,1), stride=(1,1))
        self.ps = nn.PixelShuffle(self.patch_scale)
        self.crop = nn.Conv2d(in_channels=self.output_num, out_channels=self.output_num, kernel_size=(4,1), stride=(1,1))
      
    def forward(self, x, Z, H, W):
        # The inverse operation of the patch embedding operation
        # patch_size = (2, 4, 4) as in the original paper, but we use (1,2,2)
        # Reshape x back to three dimensions, the input is (B,40*38*72,256) and output should be (B,37,73,144)
        x = torch.permute(x, (0, 2, 1))
        x = x.contiguous().view(x.shape[0], x.shape[1], Z, H, W)
        x = x.contiguous().view(x.shape[0], x.shape[1]*Z, H, W)
        # Here x is (B, 40*256, 38, 72)
        output = self.crop(self.ps(self.conv(x)))
        #output = self.crop(self.conv(x))
        
        return output

class DownSample(nn.Module):
    def __init__(self, dim):
        # Dim = 128
        super(DownSample, self).__init__()
        '''Down-sampling operation'''
        # A linear function and a layer normalization
        self.linear = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(4*dim)
    
    def forward(self, x, Z, H, W):
        # Reshape x to three dimensions for downsampling
        # From (B,40*38*72,128) to (B,40,38,72,128)
        x = x.contiguous().view(x.shape[0], Z, H, W, x.shape[-1])
        
        Z, H, W = x.shape[1:4]
        # From (B,40,38,72,128) to (B,40*19*36,4*128)
        # Reshape x to facilitate downsampling
        x = x.contiguous().view(x.shape[0], Z, H//2, 2, W//2, 2, x.shape[-1])
        # Change the order of x
        x = torch.permute(x, (0,1,2,4,3,5,6))
        x = x.contiguous().view(x.shape[0], Z*(H//2)*(W//2), 4 * x.shape[-1])

        # Call the layer normalization
        x = self.norm(x)

        # Decrease the channels of the data to reduce computation cost
        x = self.linear(x)
        # The output is (B,40*19*36,2*128)
        return x

class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpSample, self).__init__()
        '''Up-sampling operation'''
        # Linear layers without bias to increase channels of the data
        self.linear1 = nn.Linear(input_dim, output_dim*4, bias=False)

        # Linear layers without bias to mix the data up
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        # Call the linear functions to increase channels of the data
        # From (B,40*19*36,2*128) to (B,40*19*36,4*128)
        x = self.linear1(x)

        # From (B,40*19*36,4*128) to (B,40,38,72,128)
        # Reshape x to facilitate upsampling.
        x = x.contiguous().view(x.shape[0], 36, 19, 36, 2, 2, x.shape[-1]//4)
        # Change the order of x
        x = torch.permute(x, (0,1,2,4,3,5,6))
        # Reshape to get Tensor with a resolution of (40,38,72)
        x = x.contiguous().view(x.shape[0], 36, 38, 72, x.shape[-1])

        # Reshape x back to (B,40*38*72,128)
        x = x.contiguous().view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1])

        # Call the layer normalization
        x = self.norm(x)

        # Mixup normalized tensors
        # Final output is (B,40*38*72,128)
        x = self.linear2(x)
        return x
  
class EarthSpecificLayer(nn.Module):
    def __init__(self, depth, dim, heads):
        # Two kinds of ESLs :
        # depth = 2, dim = 128, heads = 8
        # depth = 6, dim = 256, heads = 16
        super(EarthSpecificLayer, self).__init__()
        '''Basic layer of our network, contains 2 or 6 blocks'''
        self.depth = depth
        self.blocklist = nn.ModuleList([EarthSpecificBlock(dim, heads) for x in range(depth)])
        
    def forward(self, x, Z, H, W):
        for block in self.blocklist:
            x = block(x, Z, H, W)
        return x

class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, num_blocks):
        # Two kinds of ESBs :
        # dim = 128, heads = 8
        # dim = 256, heads = 16
        super(EarthSpecificBlock, self).__init__()
        '''
        3D transformer block with Earth-Specific bias and window attention, 
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
        The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
        '''
        # Define the window size of the neural network 
        self.dim = dim
        self.num_blocks = num_blocks

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = SwiGLU(dim)
        self.filter = AdaptiveFourierNeuralOperator(self.dim, self.num_blocks)

    def forward(self, x, Z, H, W):
        # The data entering AFNO must be (B, N, C) where N = H*W.
        # The incoming data is (B,40*73*144,128) or (B,40*37*73,256)
        # Turning it to (B,73*144,40*128) or (B,37*73,40*256) first.
        x = x + self.filter(self.norm1(x), Z, H, W)

        # Rolling back to (B,40*73*144,128) or (B,40*37*73,256)
        x = x.contiguous().view(x.shape[0], Z*H*W, self.dim)

        # Remove drop path here and in checkpoint part. Dont ask me why, I just dont want it.
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.linear(self.norm2(x))

        return x
    
class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()
        self.hidden_size = dim

        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)

        self.softshrink = 0.01

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, Z, H, W):
        B, N, C = x.shape
        bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x.reshape(B, Z, H, W, C).float()
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        x_real_1 = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1])
        x_real_2 = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
        x_imag_2 = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.softshrink)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.hidden_size)
        x = torch.fft.irfftn(x, s=(Z, H, W), dim=(1, 2, 3), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias
  
class SwiGLU(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        out_features = in_features
        hidden_features = int(in_features*2.5)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=True)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, Z, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
    """
    B, Z, H, W, C = x.shape
    x = x.contiguous().view(B, Z // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows
