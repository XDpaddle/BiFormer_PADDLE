"""
Core of BiFormer, Bi-Level Routing Attention.

To be refactored.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from einops import rearrange
# from torch import Tensor


class TopkRouting(nn.Layer):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(axis=-1)
    
    # def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
    def forward(self, query, key):
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        # 通常情况下，使用 detach 方法可以将张量从计算图中分离出来，使其成为不可训练的常量，用于在某些情况下防止梯度更新或减少内存消耗。
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose([0,2,1]) # (n, p^2, p^2)
        topk_attn_logit, topk_index = paddle.topk(attn_logit, k=self.topk, axis=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index
        

class KVGather(nn.Layer):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    # def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
    def forward(self, r_idx, r_weight, kv):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.shape
        topk = r_idx.shape[-1]
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel? 
        topk_kv = paddle.take_along_axis(kv.reshape([n, 1, p2, w2, c_kv]).expand([-1, p2, -1, -1, -1]), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                indices=r_idx.reshape([n, p2, topk, 1, 1]).expand([-1, -1, -1, w2, c_kv]), # (n, p^2, k, w^2, c_kv)
                                # index=r_idx.reshape([49,1]),
                                # index=r_idx,
                                axis=2
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.reshape(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class QKVLinear(nn.Layer):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias_attr=bias)
    
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], axis=-1)  # in：64 out:192 192的前64通道为q 后面128为kv
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

class BiLevelRoutingAttention(nn.Layer):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5


        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2D(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: paddle.zeros_like(x)
        
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2D(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2D(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2D(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2D(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v 
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(axis=-1)

        self.auto_pad=auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
         # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.shape

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            # x = F.pad(x, (0, 0, # dim=-1  # torch
            #               pad_l, pad_r, # dim=-2
            #               pad_t, pad_b)) # dim=-3
            x = F.pad(x, (0, 0, # dim=0  # torch
                          pad_t, pad_b, # dim=1
                          pad_l, pad_r,
                          0, 0)) # dim=2           

            _, H, W, _ = x.shape # padded size
        else:
            N, H, W, C = x.shape
            assert H%self.n_win == 0 and W%self.n_win == 0 #
        ###################################################


        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = x.reshape([N,self.n_win,int(H/self.n_win),self.n_win,int(W/self.n_win),C])
        x = x.transpose([0,1,3,2,4,5])
        x = x.reshape([N,self.n_win*self.n_win,int(H/self.n_win),int(W/self.n_win),C])

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x) 

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = q.reshape([q.shape[0],q.shape[1],q.shape[2]*q.shape[3],q.shape[4]])
        kv_pix = self.kv_down(kv.reshape([kv.shape[0]*kv.shape[1],kv.shape[2],kv.shape[3],kv.shape[4]]).transpose([0,3,1,2]))
        kv_pix = kv_pix.reshape([N,self.n_win,self.n_win,kv_pix.shape[1],kv_pix.shape[2],kv_pix.shape[3]])
        kv_pix = kv_pix.transpose([0,1,2,4,5,3])
        kv_pix = kv_pix.reshape([kv_pix.shape[0],kv_pix.shape[1]*kv_pix.shape[2],kv_pix.shape[3]*kv_pix.shape[4],kv_pix.shape[5]])

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp



        lepe_input = kv[..., self.qk_dim:].reshape([kv.shape[0],self.n_win,self.n_win,kv.shape[2],kv.shape[3],kv.shape[4]-self.qk_dim])
        lepe_input = lepe_input.transpose([0,5,1,3,2,4])
        lepe_input = lepe_input.reshape([lepe_input.shape[0],lepe_input.shape[1],lepe_input.shape[2]*lepe_input.shape[3],lepe_input.shape[4]*lepe_input.shape[5]])
        lepe = self.lepe(lepe_input)
        lepe = lepe.transpose([0,2,3,1])





        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], axis=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        
        ######### do attention as normal ####################
        # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        k_pix_sel = k_pix_sel.reshape([k_pix_sel.shape[0],k_pix_sel.shape[1],k_pix_sel.shape[2],k_pix_sel.shape[3],self.num_heads,int(k_pix_sel.shape[4]/self.num_heads)])
        k_pix_sel = k_pix_sel.transpose([0,1,4,5,2,3])
        k_pix_sel = k_pix_sel.reshape([(k_pix_sel.shape[0]*k_pix_sel.shape[1]),k_pix_sel.shape[2],k_pix_sel.shape[3],(k_pix_sel.shape[4]*k_pix_sel.shape[5])])
        # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        v_pix_sel = v_pix_sel.reshape([v_pix_sel.shape[0],v_pix_sel.shape[1],v_pix_sel.shape[2],v_pix_sel.shape[3],self.num_heads,int(v_pix_sel.shape[4]/self.num_heads)])
        v_pix_sel = v_pix_sel.transpose([0,1,4,5,2,3])
        v_pix_sel = v_pix_sel.reshape([v_pix_sel.shape[0]*v_pix_sel.shape[1],v_pix_sel.shape[2],v_pix_sel.shape[3],v_pix_sel.shape[4]*v_pix_sel.shape[5]])        
        v_pix_sel = v_pix_sel.transpose([0,1,3,2])
        # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
        q_pix = q_pix.reshape([q_pix.shape[0],q_pix.shape[1],q_pix.shape[2],self.num_heads,int(q_pix.shape[3]/self.num_heads)])
        q_pix = q_pix.transpose([0,1,3,2,4])
        q_pix = q_pix.reshape([q_pix.shape[0]*q_pix.shape[1],q_pix.shape[2],q_pix.shape[3],q_pix.shape[4]])

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = out.reshape([N,self.n_win,self.n_win,self.num_heads,int(H/self.n_win),int(W/self.n_win),out.shape[3]])
        out = out.transpose([0,1,4,2,5,3,6])
        out = out.reshape([N,out.shape[1]*out.shape[2],out.shape[3]*out.shape[4],out.shape[5]*out.shape[6]])

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            # out = out[:, :H_in, :W_in, :].contiguous()
            out = out[:, :H_in, :W_in, :]

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out
