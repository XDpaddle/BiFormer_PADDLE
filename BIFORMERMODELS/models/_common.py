# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from einops import rearrange

class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.transpose([0, 3, 1, 2]) #NCHW
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1]) #NHWC

        return x

class Attention(nn.Layer):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.shape
        x = x.reshape([x.shape[0], H*W, x.shape[3]])
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose([0,1,3,2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0,2,1,3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = x.reshape([B, H, W, C])
        return x

class AttentionLePE(nn.Layer):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2D(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: paddle.zeros_like(x)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.shape
        x = x.reshape([x.shape[0], H*W, x.shape[3]])
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        lepe = self.lepe(x.reshape([B,H,W,C]).transpose([0,3,1,2]))
        lepe = lepe.transpose([0,2,3,1]).reshape([B, N, C])

        attn = (q @ k.transpose([0,1,3,2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0,2,1,3]).reshape([B, N, C])
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = x.reshape([B,H,W,C])
        return x



# class nchwAttentionLePE(nn.Module):
#     """
#     Attention with LePE, takes nchw input
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = qk_scale or self.head_dim ** -0.5

#         self.qkv = nn.Conv2D(dim, dim*3, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
#                     lambda x: torch.zeros_like(x)

#     def forward(self, x:torch.Tensor):
#         """
#         args:
#             x: NCHW tensor
#         return:
#             NCHW tensor
#         """
#         B, C, H, W = x.size()
#         q, k, v = self.qkv.forward(x).chunk(3, dim=1) # B, C, H, W

#         attn = q.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2) @ \
#                k.view(B, self.num_heads, self.head_dim, H*W)
#         attn = torch.softmax(attn*self.scale, dim=-1)
#         attn = self.attn_drop(attn)

#         # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
#         output:torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H*W).transpose(-1, -2)
#         output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
#         output = output + self.lepe(v)

#         output = self.proj_drop(self.proj(output))

#         return output
