import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SSTransformer(nn.Module):
    def __init__(self, patch_sizes, in_channel, out_channel, 
                 depth, head, dim_head, mlp_head, dropout):
        super().__init__()

        self.Zs = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(10)])


        ## 1. Spatial Transformer
        self.spatial_token     = nn.Parameter(torch.randn(1, 1, patch_sizes**2))
        self.spatial_embedding = nn.Sequential(
                nn.Linear(patch_sizes**2, patch_sizes**2),
                nn.LayerNorm(patch_sizes**2),
                nn.ReLU(inplace=True),)
        
        self.spatial_path = nn.ModuleList([])
        for _ in range(depth):
            self.spatial_path.append(nn.ModuleList([
                Residual(PreNorm(dim=patch_sizes**2, fn=Attention(dim=patch_sizes**2,   heads = head, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim=patch_sizes**2, fn=FeedForward(dim=patch_sizes**2, hidden_dim=mlp_head, dropout = dropout)))]))

        ## 2. Spectral Transformer
        self.spectral_token     = nn.Parameter(torch.randn(1, 1, in_channel))
        self.spectral_embedding = nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.ReLU(inplace=True),)
        
        self.spectral_path = nn.ModuleList([])
        for _ in range(depth):
            self.spectral_path.append(nn.ModuleList([
                Residual(PreNorm(dim=in_channel, fn=Attention(dim=in_channel,   heads=head, dim_head=dim_head, dropout = dropout))),
                Residual(PreNorm(dim=in_channel, fn=FeedForward(dim=in_channel, hidden_dim=mlp_head, dropout = dropout)))]))
        
        self.TD = nn.Sequential(
            nn.Linear(patch_sizes**2+1, patch_sizes**2),
            nn.LayerNorm(patch_sizes**2),
            nn.ReLU(inplace=True),)
        
        ## 3.    
        self.fusion_path = nn.ModuleList([])
        for _ in range(depth):
            self.fusion_path.append(nn.ModuleList([
                Residual(PreNorm(dim=in_channel*2, fn=Attention(dim=in_channel*2,   heads=head, dim_head=dim_head, dropout = dropout))),
                Residual(PreNorm(dim=in_channel*2, fn=FeedForward(dim=in_channel*2, hidden_dim=mlp_head, dropout = dropout)))]))
        

        self.channel_DR = nn.Sequential(
            nn.Linear(in_channel*2, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),)

        
    def forward(self, x, mask = None):
        """
        Input:  x -> [batch, inchannel,  patch, patch]
        Output: x -> [batch, outchannel, patch, patch]
        """
        B, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')

        ## 1. Spatial Transformer
        spatial_tokens = repeat(self.spatial_token, '() n d -> b n d', b = B)
        spatial_x = spatial_tokens + self.Zs[0]*x
        for attn, ff in self.spatial_path:
            spatial_x = attn(spatial_x, mask = mask)
            spatial_x = ff(spatial_x)

        ## 2. Spectral Transformer
        spectral_x = rearrange(x, 'b n d -> b d n')
        spectral_cls_tokens = repeat(self.spectral_token, '() n d -> b n d', b = B)
        spectral_x = torch.cat((spectral_cls_tokens, self.Zs[1]*spectral_x), dim = 1)
        for attn, ff in self.spectral_path:
            spectral_x = attn(spectral_x, mask = mask)
            spectral_x = ff(spectral_x)
        spectral_x = repeat(spectral_x, 'b d n -> b n d')
        spectral_x = self.TD(spectral_x)

        ## 3. Fusion Transformer
        fusion_x = torch.cat((spectral_x, self.Zs[2]*spatial_x),dim=1)
        fusion_x = repeat(fusion_x, 'b d n -> b n d')
        for attn, ff in self.fusion_path:
            fusion_x = attn(fusion_x, mask = mask)
            fusion_x = ff(fusion_x)
        fusion_x = self.channel_DR(fusion_x)
        fusion_x = rearrange(fusion_x, 'b n d -> b d n')



        return rearrange(fusion_x, 'b n (h w) -> b n h w', h=H, w=W)






class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out