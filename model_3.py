'''
Ablation study part 3 extra.
Transformer + Pixel Shuffle + SwishMlp.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

class PModel(nn.Module):
    def __init__(self, input_time, output_num):
        super(PModel, self).__init__()
        # Drop path rate is linearly increased as the depth increases
        # No more drop path
        # drop_path_list = torch.linspace(0, 0.2, 8)
        self.input_time = input_time
        self.output_num = output_num

        # Patch embedding
        self._input_layer = PatchEmbedding((1, 2, 2), self.input_time, 64)

        # Four basic layers
        self.layer1 = EarthSpecificLayer(4, 64, 8)
        self.layer2 = EarthSpecificLayer(6, 128, 16)
        self.layer3 = EarthSpecificLayer(6, 128, 16)
        self.layer4 = EarthSpecificLayer(4, 64, 8)

        # Upsample and downsample
        self.downsample = DownSample(64)
        self.upsample = UpSample(128, 64)

        # Patch Recovery
        self._output_layer = PatchRecovery((1, 2, 2), self.output_num, 128)
      
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
    def __init__(self, patch_size, input_time, dim):
        super(PatchEmbedding, self).__init__()
        '''Patch embedding operation'''
        # Here we use convolution to partition data into cubes
        # The coming data is (B,3,40,73,144) where 3 is incoming time, see as channels
        # The final two of 39 is constants: lsm and topo.
        self.input_time = input_time
        self.conv_pressure = nn.Conv3d(in_channels=self.input_time, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.conv_const = nn.Conv3d(in_channels=1, out_channels=dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, input):
        # Zero-pad the input
        input_pressure=input[:,:,:-3,:,:]
        input_const=input[:,:,-3:,:,:].mean(axis=1,keepdim=True)

        '''------------------------------------------PAD-------------------------------------------------------'''
        # From (B,3,37,73,144) to (B,3,39,76,144)
        # From (B,3,73,144) to (B,3,76,144)
        input_pressure = F.pad(input_pressure, (0, 0, 2, 1), mode='constant', value=0)
        input_const = F.pad(input_const, (0, 0, 2, 1), mode='constant', value=0)

        # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches
        # patch_size = (2, 4, 4) in the original paper, but here we use (1,2,2)
        # From (B,3,39,76,144)/(B,3,76,144) to (B,128,39,38,72)/(B,128,1,38,72)
        # Finally (B,128,40,38,72)
        input_pressure = self.conv_pressure(input_pressure)
        input_const = self.conv_const(input_const)
        x=torch.cat([input_pressure,input_const],axis=2)

        # Reshape x for calculation of linear projections
        # From (B,128,40,38,72) to (B,40*38*72,128)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(x.shape[0], 36*38*72, x.shape[-1])
        return x
    
class PatchRecovery(nn.Module):
    def __init__(self, patch_size, output_num, dim):
        # Dim = 256
        super(PatchRecovery, self).__init__()
        '''Patch recovery operation'''
        # Hear we use two transposed convolutions to recover data
        self.output_num = output_num
        #self.conv = nn.ConvTranspose2d(in_channels=dim*36, out_channels=self.output_num, kernel_size=patch_size[1:], stride=patch_size[1:])
        # We use Pixel Shuffle here in substitute of Conv Transpose
        self.conv = nn.Conv2d(in_channels=dim*36, out_channels=self.output_num*4, kernel_size=(1,1), stride=(1,1))
        self.ps = nn.PixelShuffle(2)
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
        for i, block in enumerate(self.blocklist):
            roll = (i % 2 == 1)
            x = block(x, Z, H, W, roll=roll)
        return x

class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, heads):
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
        self.window_size = (2, 4, 4)

        # Initialize serveral operations
        if dim==64:
            self.crop = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1,3,1), stride=(1,1,1))
        elif dim==128:
            self.crop = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1,2,1), stride=(1,1,1))
        # self.drop_path = DropPath(drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = SwiGLU(dim)
        self.attention = EarthAttention3D(dim, heads, self.window_size)

    def forward(self, x, Z, H, W, roll):
        # Save the shortcut for skip-connection
        shortcut = x

        # Reshape input to three dimensions to calculate window attention
        # From (B,40*38*72,128) / (B,40*19*36,256) to (B,40,38,72,128) / (B,40,19,36,256)
        x = x.contiguous().view(x.shape[0], Z, H, W, x.shape[2])

        # Zero-pad input if needed
        # From (B,40,38,72,128) / (B,40,19,36,256) to (B,40,40,72,128) / (B,40,20,36,256)
        '''------------------------------------------PAD-------------------------------------------------------'''
        if self.dim == 64:
            x = F.pad(x, (0, 0, 0, 0, 1, 1), mode='constant', value=0)
        elif self.dim == 128:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), mode='constant', value=0)

        # Store the shape of the input for restoration
        ori_shape = x.shape
        Z, H, W = x.shape[1], x.shape[2], x.shape[3]

        if roll:
            '''------------------------------------------MASK-------------------------------------------------------'''
            shift_size=(self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2)
            # Roll x for half of the window for 3 dimensions
            x = torch.roll(x, shifts=(self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2), dims=(1, 2, 3))
            # Generate mask of attention masks
            # If two pixels are not adjacent, then mask the attention between them
            # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
            img_mask = torch.zeros((1, Z, H, W, 1)).cuda()
            z_slices = (slice(0, -self.window_size[0]),slice(-self.window_size[0], -shift_size[0]),slice(-shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),slice(-self.window_size[1], -shift_size[1]),slice(-shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),slice(-self.window_size[2], -shift_size[2]),slice(-shift_size[2], None))
            cnt = 0
            for z in z_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, z, h, :, :] = cnt
                        cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.contiguous().view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))
        else:
            # e.g., zero matrix when you add mask to attention
            mask = None

        # Reorganize data to calculate window attention
        # From (B,40,40,72,128) to (B,20,2,10,4,18,4,128) to (B,20,10,18,2,4,4,128)
        # From (B,40,20,36,256) to (B,20,2,5,4,9,4,256) to (B,20,5,9,2,4,4,256)
        x_window = x.contiguous().view(x.shape[0], Z//self.window_size[0], self.window_size[0], H // self.window_size[1], self.window_size[1], W // self.window_size[2], self.window_size[2], x.shape[-1])
        x_window = torch.permute(x_window, (0, 1, 3, 5, 2, 4, 6, 7))

        # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube
        # From (B,20,10,18,2,4,4,128) to (B*20*10*18,2*4*4,128)
        # From (B,20,5,9,2,4,4,256) to (B*20*5*9,2*4*4,256)
        x_window = x.contiguous().view(-1, self.window_size[0]* self.window_size[1]* self.window_size[2], x.shape[-1])

        # Apply 3D window attention with Earth-Specific bias
        x_window = self.attention(x_window, mask, roll)

        # Reorganize data to original shapes
        # From (B*20*10*18,2*4*4,128) to (B,40,40,72,128)
        # From (B*20*5*9,2*4*4,256) to (B,40,20,36,256)
        x = x_window.contiguous().view(-1, Z // self.window_size[0], H // self.window_size[1], W // self.window_size[2], self.window_size[0], self.window_size[1], self.window_size[2], x_window.shape[-1])
        x = torch.permute(x, (0, 1, 4, 2, 5, 3, 6, 7))
        x = x.contiguous().view(ori_shape)

        if roll:
            # Roll x back for half of the window
            x = torch.roll(x, shifts=[-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2], dims=(1, 2, 3))

        # Crop the zero-padding
        # From (B,40,40,72,128) / (B,40,20,36,256) to (B,40,38,72,128) / (B,40,19,36,256)
        '''------------------------------------------CROP-------------------------------------------------------'''
        x = torch.permute(x, (0, 4, 1, 2, 3))
        x = self.crop(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))

        # Reshape the tensor back to the input shape
        # Back to (B,40*38*72,128) or (B,40*19*36,256)
        x = x.contiguous().view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4])

        # Main calculation stages
        # No drop path here.
        # x = shortcut + self.norm1(x)
        # x = x + self.norm2(self.linear(x))

        # Replace it with checkpoint.
        x = shortcut + self.norm1(x)
        x = x + self.norm2(self.linear(x))

        # The output is (B,40*38*72,128) or (B,40*19*36,256)
        return x
    
class EarthAttention3D(nn.Module):
    def __init__(self, dim, heads, window_size):
        super(EarthAttention3D, self).__init__()
        '''
        3D window attention with the Earth-Specific bias, 
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
        '''
        # Initialize several operations
        self.linear1 = nn.Linear(in_features=dim, out_features=dim*3, bias=True)
        self.linear2 = nn.Linear(in_features=dim, out_features=dim, bias=True)
        # Softmax will cause nan here, change it to sigmoid
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(dropout_rate)

        # Store several attributes
        self.head_number = heads
        self.dim = dim
        self.scale = (dim//heads)**-0.5
        self.window_size = window_size

        # For each type of window, we will construct a set of parameters according to the paper
        # Making these tensors to be learnable parameters
        self.bias_size = self.window_size[0]*self.window_size[1]*self.window_size[2]
        self.earth_specific_bias = torch.empty(self.bias_size, self.bias_size, heads)

        # Initialize the tensors using Truncated normal distribution
        nn.init.trunc_normal_(self.earth_specific_bias, std=0.02)
        self.earth_specific_bias = nn.Parameter(self.earth_specific_bias.contiguous().view(-1, heads), requires_grad=True)
        
        # Construct position index to reuse self.earth_specific_bias
        self._construct_index()
        
    def _construct_index(self):
        ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
        # Index in the pressure level of query matrix
        coords_zi = torch.arange(self.window_size[0])
        # Index in the pressure level of key matrix
        coords_zj = -torch.arange(self.window_size[0])*self.window_size[0]

        # Index in the latitude of query matrix
        coords_hi = torch.arange(self.window_size[1])
        # Index in the latitude of key matrix
        coords_hj = -torch.arange(self.window_size[1])*self.window_size[1]

        # Index in the longitude of the key-value pair
        coords_w = torch.arange(self.window_size[2])

        # Change the order of the index to calculate the index in total
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
        coords_flatten_1 = torch.flatten(coords_1, start_dim=1) 
        coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        coords = torch.permute(coords, (1, 2, 0))

        # Shift the index for each dimension to start from 0
        coords[:, :, 2] += self.window_size[2] - 1
        coords[:, :, 1] *= 2 * self.window_size[2] - 1
        coords[:, :, 0] *= (2 * self.window_size[2] - 1)*self.window_size[1]*self.window_size[1]

        # Sum up the indexes in three dimensions
        self.position_index = torch.sum(coords, dim=-1)

        # Flatten the position index to facilitate further indexing
        self.position_index = torch.flatten(self.position_index)
        
    def forward(self, x, mask, roll):
        # Linear layer to create query, key and value
        original_shape = x.shape
        x = self.linear1(x)

        # reshape the data to calculate multi-head attention
        # From (B*20*10*18,2*4*4,128*3) to 3*(B*20*10*18,8,2*4*4,16)
        # From (B*20*5*9,2*4*4,256*3) to 3*(B*20*5*9,16,2*4*4,16)
        qkv = x.contiguous().view(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number)
        query, key, value = torch.permute(qkv, (2, 0, 3, 1, 4))

        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        # From (B*20*10*18,8,2*4*4,16) to (B*20*10*18,8,2*4*4,2*4*4)
        # From (B*20*5*9,16,2*4*4,16) to (B*20*5*9,16,2*4*4,2*4*4)
        attention=(query @ key.transpose(-2, -1))
        
        # self.earth_specific_bias is a set of neural network parameters to optimize. 
        EarthSpecificBias = self.earth_specific_bias[self.position_index]

        # Reshape the learnable bias to the same shape as the attention matrix
        EarthSpecificBias = EarthSpecificBias.contiguous().view(self.window_size[0]*self.window_size[1]*self.window_size[2], self.window_size[0]*self.window_size[1]*self.window_size[2], self.head_number)
        EarthSpecificBias = torch.permute(EarthSpecificBias, (2, 0, 1))
        EarthSpecificBias = EarthSpecificBias.contiguous().view(torch.Size([1])+EarthSpecificBias.shape)

        # Add the Earth-Specific bias to the attention matrix
        attention = attention + EarthSpecificBias

        # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
        # Mask is (B*20*10*18,2*4*4,2*4*4) or (B*20*5*9,2*4*4,2*4*4)
        # Attention is (B*20*10*18,8,2*4*4,2*4*4) or (B*20*5*9,16,2*4*4,2*4*4)
        '''--------------------------------------------------MASK--------------------------------------------------'''
        if roll:
            nW = mask.shape[0]
            attention = attention.contiguous().view(original_shape[0]// nW, nW, self.head_number, original_shape[1], original_shape[1]) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.contiguous().view(-1, self.head_number, original_shape[1], original_shape[1])
        else:
            pass

        attention = self.softmax(attention)
        # attention = self.sigmoid(attention)
        # attention = self.dropout(attention)

        # Calculated the tensor after spatial mixing.
        # From (B*20*10*18,8,2*4*4,2*4*4) to (B*20*10*18,8,2*4*4,16)
        # From (B*20*5*9,16,2*4*4,2*4*4) to (B*20*5*9,16,2*4*4,16)
        x = torch.matmul(attention, value)

        # Reshape tensor to the original shape
        # From (B*20*10*18,8,2*4*4,16) to (B*20*10*18,2*4*4,8,16) to (B*20*10*18,2*4*4,128)
        # From (B*20*5*9,16,2*4*4,16) to (B*20*5*9,2*4*4,16,16) to (B*20*5*9,2*4*4,256)
        x = torch.permute(x, (0, 2, 1, 3))
        x = x.contiguous().view(original_shape)

        # Linear layer to post-process operated tensor
        x = self.linear2(x)
        # x = self.dropout(x)
        return x
  
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
