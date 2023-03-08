import math
from functools import partial

import torch
import torch.nn as nn

from models import logger

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return logger._no_grad_trunc_normal_(tensor, mean, std, a, b)
  

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Stochastic Depth. Not useful when single-block wit.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

def HSlice(s):
    return s if isinstance(s, tuple) else (s, s)

class SubEmbed(nn.Module):
    """ Sub-channel slicing and Linear embeddings
    """
    def __init__(
            self,
            H_size=(64,32),
            h_slice=(64,1),
            in_chans=3,
            embed_dim=384,
    ):
        super().__init__()
        H_size = HSlice(H_size)
        h_slice = HSlice(h_slice)
        self.H_size = H_size
        self.h_slice = h_slice
        self.slice_size = (H_size[0] // h_slice[0], H_size[1] // h_slice[1])
        self.num_slices = self.slice_size[0] * self.slice_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=h_slice, stride=h_slice)
        ''' 
            Linear projection in 'all' recent transformer-based models with a conv2d. 
            Otherwise, use nn.Sequential. Parameters for both models are provided.
        '''
        # self.proj = nn.Sequential(
        #     Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = h_slice[0], s2 = h_slice[1]),
        #     nn.Linear( 3* h_slice[0] * h_slice[1], embed_dim),
        # )          

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x) # Only self.proj(x) if alternative to using conv2d for projection to linear embeddings.
        x = x.flatten(2).transpose(1, 2)  # Comment this when projecting with nn.seq.
        return x

class Projectors(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 num_layers=3, 
                 hidden_dim=1024, 
                 bottleneck_dim=256):
        super().__init__()
        num_layers = max(num_layers, 1)
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=1, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # We use single-head. Otherwise, D = D / H_att
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.scale * (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ WiT based on ViT """
    def __init__(
            self, 
            H_size=(64, 36), 
            h_slice=(64,1), 
            in_chans=3, 
            num_classes=0, 
            embed_dim=384, 
            depth=1,
            num_heads=1, 
            mlp_ratio=1., 
            qkv_bias=False, 
            qk_scale=None, 
            drop_rate=0., 
            attn_drop_rate=0.,
            drop_path_rate=0., 
            norm_layer=nn.LayerNorm, **kwargs):

        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.slice_embed = SubEmbed(
            H_size=H_size, h_slice=h_slice, in_chans=in_chans, embed_dim=embed_dim)
        num_slices = self.slice_embed.num_slices

        self.enc_lid_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # depth decay rule

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        # Linear MLP Head - Identity mostly.
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.enc_lid_token, std=.02) # Added.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x):
        Num_slices = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if Num_slices == N:
            return self.pos_embed
        enc_lid_tokens = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.permute(0, 2, 1), 
            scale_factor=(Num_slices/N), 
            mode='linear',
            recompute_scale_factor=True)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 1).view(1, -1, dim)
        return torch.cat((enc_lid_tokens.unsqueeze(0), patch_pos_embed), dim=1)
   
    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        x = self.slice_embed(x) 
        enc_lid_tokens = self.enc_lid_token.expand(B, -1, -1)
        x = torch.cat((enc_lid_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x)

        return self.pos_drop(x)

    def forward(self, x, loca=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return x[:,1:,:]

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(x)
        return output

def wit(h_slice=(64,1), **kwargs):
    model = TransformerEncoder(
        h_slice=h_slice, 
        embed_dim=384, 
        depth=1, 
        num_heads=1, 
        mlp_ratio=4, #1
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class Attention_Pooling(nn.Module):
    def __init__(
            self, 
            dim, 
            num_heads=8, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_drop=0., 
            proj_drop=0.):
        
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention=False):
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_lid = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_lid = self.proj(x_lid)
        x_lid = self.proj_drop(x_lid)
        
        if attention:
            return x_lid, attn
        else:
            return x_lid

class Block_Pooling(nn.Module):
    def __init__(
                self, 
                dim, 
                num_heads, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop=0., 
                attn_drop=0.,
                init_values=None,
                drop_path=0., 
                act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm, 
                Attention_blck = Attention_Pooling,
                Mlp_block=Mlp):
        
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention_blck(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_lid, attention=False):
        x_all = torch.cat((x_lid,x),dim=1)
        if attention:
            x_all_, lid_attn = self.attn(self.norm1(x_all), attention=True)
            return lid_attn
        else:
            x_all_ = self.attn(self.norm1(x_all))
        x_lid = x_lid + self.drop_path(x_all_)
        x_lid = x_lid + self.drop_path(self.mlp(self.norm2(x_lid)))
        return x_lid

class TransformerPooling(nn.Module):
    def __init__(
            self, 
            in_dim, 
            num_heads, 
            k_nns):
        
        super().__init__()

        self.lid_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.pooling_blocks = nn.ModuleList([
            Block_Pooling(dim=in_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, Attention_blck=Attention_Pooling, Mlp_block=Mlp)
            for i in range(1)])
        trunc_normal_(self.lid_token, std=.02)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(in_dim)
        self.apply(self._init_weights)
        self.k_nns = k_nns
        self.kk = 3
        self.loca36 = self.subcarrier_matching(36, self.kk) 
        self.loca16 = self.subcarrier_matching(16, self.kk) 
        self.embed_dim = in_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, loca=False):
        lid_tokens = self.lid_token.expand(x.shape[0], -1, -1)
        if loca:
            if x.shape[1] == 36:
                matched_carriers = self.loca36
            elif x.shape[1] == 16: 
                matched_carriers = self.loca16                       
            else:
                assert(False)
            x_norm = nn.functional.normalize(x, dim=-1)
            matched_x_norm = x_norm[:, matched_carriers]
            sim_matrix = matched_x_norm @ x_norm.unsqueeze(2).transpose(-2, -1)
            top_idx = sim_matrix.squeeze().topk(k=self.k_nns,dim=-1)[1].view(-1,self.k_nns,1)
            x_loca = x[:,matched_carriers].view(-1,self.kk*2,self.embed_dim) 
            x_loca = torch.gather(x_loca, 1, top_idx.expand(-1, -1, self.embed_dim))
            for i, blk in enumerate(self.pooling_blocks):
                if i == 0:
                    glo_tokens = blk(x, lid_tokens)
                    loca_tokens = blk(x_loca, lid_tokens.repeat(x.shape[1],1,1))
                else:
                    glo_tokens = blk(x, glo_tokens)
                    loca_tokens = blk(x_loca, loca_tokens)
            loca_tokens = loca_tokens.view(x.shape)
            x = self.norm(torch.cat([glo_tokens, loca_tokens], dim=1))
        else:
            for i, blk in enumerate(self.pooling_blocks):
                lid_tokens = blk(x, lid_tokens)
            x = self.norm(torch.cat([lid_tokens, x], dim=1))
        return x
  
    @staticmethod
    def subcarrier_matching(Num_slices, kk):
        carriers_loca = []
        for i in range(Num_slices):
            knn_carriers = torch.zeros(Num_slices)
            if i < kk:
                neighb_indices = [j for j in range(0, (2*kk+1))]
                knn_carriers[neighb_indices] = 1
            elif i >= (Num_slices-kk):
                neighb_indices = [j for j in range(Num_slices-(2*kk+1), Num_slices)]
                knn_carriers[neighb_indices] = 1
            else:
                neighb_indices = [j for j in range(i-kk, i+(kk+1))]
                knn_carriers[neighb_indices] = 1
            knn_carriers[i] = 0
            carriers_loca.append(knn_carriers.nonzero().squeeze())
        return torch.stack(carriers_loca)
