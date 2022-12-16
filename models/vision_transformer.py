# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=384, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, u, v):  # [B,L]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=u.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = u[:, :, None] / dim_t * self.scale
        pos_y = v[:, :, None] / dim_t * self.scale

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)

        return pos


class SinglePositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=384, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):  # [B,L]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x[:, :, None] / dim_t * self.scale

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        return pos_x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        self.qk_scale = qk_scale  # 224 (14**2)
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_scale == 'default':
            scale = self.scale
        else:
            scale = math.log(N, self.qk_scale ** 2 + 1) * self.scale
        attn = (q @ k.transpose(-2, -1)) * scale  # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, nview=5):
        super().__init__()
        self.num_heads = num_heads
        self.nview = nview
        self.eps = 1e-6

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x:[BV,HW,C]
        BV, HW, C = x.shape
        V = self.nview
        B = BV // V
        x = x.reshape(B, V, HW, C).reshape(B, V * HW, C)
        qkv = self.qkv(x).reshape(B, V * HW, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,nh,VHW,C]
        q = q.permute(0, 2, 1, 3).contiguous()  # [B,VHW,nh,C2]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        k = F.elu(k) + 1.0
        q = F.elu(q) + 1.0

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            kv = torch.einsum("nlhd,nlhm->nhmd", k, v)  # [B,nh,C2,C2]
            # Compute the normalizer
            z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps)
            # Finally compute and return the new values
            y = torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z)  # [B,VHW,nh,C2]
        y = y.reshape(B, V, HW, C).reshape(BV, HW, C)
        y = self.proj(y)

        return y


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, nview=5):
        super().__init__()
        self.nview = nview
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, nview=nview)
        # self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.tok_embeddings = nn.Embedding(2, dim)
        # self.epipole_embeddings = PositionEmbeddingSine(dim // 2)
        self.rel_epipole_emb = PositionEmbeddingSine(dim // 4, scale=32 * math.pi)  # 每个patch相对方向大致变化在0.01~0.05
        self.abs_epipole_dis_emb = SinglePositionEmbeddingSine(dim // 4, scale=2 * math.pi)  # 每个极点距离，大致在100~500
        self.abs_epipole_dir_emb = PositionEmbeddingSine(dim // 8, scale=2 * math.pi)  # 每个极点方向

    def forward(self, x, epipole, height, width):
        # x:[BV,1+HW,C], src_epipoles:[B,V-1,2]
        BV, HW, C = x.shape
        B = BV // self.nview
        if epipole is None:
            ref_ids = torch.zeros((B, 1, HW), dtype=torch.long, device=x.device)
            src_ids = torch.ones((B, self.nview - 1, HW), dtype=torch.long, device=x.device)
            tok_ids = torch.cat([ref_ids, src_ids], dim=1)  # [B,V,1+HW]
            tok_ids = tok_ids.reshape(BV, HW)
            tok_emb = self.tok_embeddings(tok_ids)  # [BV,1+HW,C]
        else:
            # 方案1
            # y_, x_ = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=x.device),
            #                          torch.arange(0, width, dtype=torch.float32, device=x.device)])
            # x_, y_ = x_.contiguous(), y_.contiguous()
            # x_ = x_.reshape(1, 1, height, width)
            # y_ = y_.reshape(1, 1, height, width)
            # epipole_map = epipole.reshape(B, self.nview - 1, 2, 1, 1)  # [B, V-1, 2, 1, 1]
            # u = x_ - epipole_map[:, :, 0, :, :]  # [B, V-1, H, W]
            # v = y_ - epipole_map[:, :, 1, :, :]  # [B, V-1, H, W]
            # normed_uv = torch.sqrt(u ** 2 + v ** 2)
            # u, v = u / (normed_uv + 1e-6), v / (normed_uv + 1e-6)
            # u = u.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            # v = v.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            # epi_emb = self.epipole_embeddings(u, v)  # [B(V-1),HW,C]

            # 方案2
            # epipole_map = epipole.reshape(B, self.nview - 1, 2)  # [B, V-1, 2]
            # epipole_map = F.normalize(epipole_map, dim=2)
            # epipole_map = epipole_map.reshape(B, self.nview - 1, 2, 1, 1).repeat(1, 1, 1, height, width)
            # u, v = epipole_map[:, :, 0], epipole_map[:, :, 1]  # [B,V-1,H,W]
            # u = u.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            # v = v.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            # epi_emb = self.epipole_embeddings(u, v)
            # epi_emb = epi_emb.reshape(B, self.nview - 1, HW - 1, C)  # [B,V-1,HW,C]
            # ref_ids = torch.zeros((B, HW), dtype=torch.long, device=x.device)  # [B,1+HW]
            # sep_ids = torch.ones((B, self.nview - 1), dtype=torch.long, device=x.device)  # [B,V-1]
            # ref_emb = self.tok_embeddings(ref_ids).unsqueeze(1)  # [B,1,1+HW,C]
            # sep_emb = self.tok_embeddings(sep_ids).unsqueeze(2)  # [B,V-1,1,C]
            # src_emb = torch.cat([sep_emb, epi_emb], dim=2)  # [B,V-1,1+HW,C]
            # tok_emb = torch.cat([ref_emb, src_emb], dim=1)  # [B,V,1+HW,C]
            # tok_emb = tok_emb.reshape(BV, HW, C)

            # 方案3
            y_, x_ = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=x.device),
                                     torch.arange(0, width, dtype=torch.float32, device=x.device)])
            x_, y_ = x_.contiguous(), y_.contiguous()
            x_ = x_.reshape(1, 1, height, width)
            y_ = y_.reshape(1, 1, height, width)
            epipole_map = epipole.reshape(B, self.nview - 1, 2, 1, 1)  # [B, V-1, 2, 1, 1]
            rel_u = x_ - epipole_map[:, :, 0, :, :]  # [B, V-1, H, W]
            rel_v = y_ - epipole_map[:, :, 1, :, :]  # [B, V-1, H, W]
            normed_uv = torch.sqrt(rel_u ** 2 + rel_v ** 2)
            rel_u, rel_v = rel_u / (normed_uv + 1e-6), rel_v / (normed_uv + 1e-6)
            rel_u = rel_u.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            rel_v = rel_v.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            rel_epi_emb = self.rel_epipole_emb(rel_u, rel_v)  # [B(V-1),HW,C//2]

            epipole_map = F.normalize(epipole_map, dim=2)
            epipole_map = epipole_map.repeat(1, 1, 1, height, width)
            abs_u, abs_v = epipole_map[:, :, 0], epipole_map[:, :, 1]  # [B,V-1,H,W]
            abs_u = abs_u.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            abs_v = abs_v.reshape(B * (self.nview - 1), HW - 1)  # (-1~1)
            abs_epi_dir_emb = self.abs_epipole_dir_emb(abs_u, abs_v)  # [B(V-1),HW,C//4]
            epipole_dis = torch.sqrt(epipole[:, :, 0] ** 2 + epipole[:, :, 1] ** 2) / 512
            epipole_dis = torch.clamp(epipole_dis, 0, 1.0)
            epipole_dis = epipole_dis.reshape(B, self.nview - 1, 1, 1).repeat(1, 1, height, width)  # [B,V-1,H,W]
            epipole_dis = epipole_dis.reshape(B * (self.nview - 1), HW - 1)  # [B(V-1),HW](0~1)
            abs_epi_dis_emb = self.abs_epipole_dis_emb(epipole_dis)  # [B(V-1),HW,C//4]
            abs_epi_emb = torch.cat([abs_epi_dir_emb, abs_epi_dis_emb], dim=-1)

            epi_emb = torch.cat([abs_epi_emb, rel_epi_emb], dim=2)  # [B(V-1),HW,C]
            epi_emb = epi_emb.reshape(B, self.nview - 1, HW - 1, C)

            ref_ids = torch.zeros((B, HW), dtype=torch.long, device=x.device)  # [B,1+HW]
            sep_ids = torch.ones((B, self.nview - 1), dtype=torch.long, device=x.device)  # [B,V-1]
            ref_emb = self.tok_embeddings(ref_ids).unsqueeze(1)  # [B,1,1+HW,C]
            sep_emb = self.tok_embeddings(sep_ids).unsqueeze(2)  # [B,V-1,1,C]
            src_emb = torch.cat([sep_emb, epi_emb], dim=2)  # [B,V-1,1+HW,C]
            tok_emb = torch.cat([ref_emb, src_emb], dim=1)  # [B,V,1+HW,C]
            tok_emb = tok_emb.reshape(BV, HW, C)

        x1 = x + tok_emb

        x2 = x + self.alpha1 * self.attn(x1)
        out = x2 + self.alpha2 * self.mlp(x2)

        return out


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale='default', drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.nview = kwargs.get('nview', 5)
        self.cross_att = kwargs.get('cross_att', False)
        self.patch_size = patch_size
        self.height = -1
        self.width = -1

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if qk_scale != 'default':
            qk_scale = (224 / patch_size) ** 2

        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                                           norm_layer=norm_layer) for i in range(depth)])
        if self.cross_att:
            self.cross_num = kwargs.get('cross_num', 4)
            self.cross_inter = depth // self.cross_num
            self.cross_blocks = nn.ModuleList([CrossBlock(embed_dim, num_heads, mlp_ratio=2., qkv_bias=qkv_bias, drop=0.,
                                                          act_layer=nn.GELU, norm_layer=nn.LayerNorm, nview=5) for _ in range(self.cross_num)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic', align_corners=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        self.height = h // self.patch_size
        self.width = w // self.patch_size
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, h, w)

        return self.pos_drop(x)

    def forward(self, x, src_epipoles=None):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.cross_att and (i + 1) % self.cross_inter == 0:
                x = self.cross_blocks[(i + 1) // self.cross_inter - 1](x, src_epipoles, self.height, self.width)
        x = self.norm(x)
        return x

    def forward_with_last_att(self, x):
        x = self.prepare_tokens(x)
        att = None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, att = blk(x, return_attention=True)
        x = self.norm(x)
        return x, att

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class HRVisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale='default', drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.nview = kwargs.get('nview', 5)
        self.cross_att = kwargs.get('cross_att', False)
        self.patch_size = patch_size
        self.height = -1
        self.width = -1

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if qk_scale != 'default':
            qk_scale = (224 / patch_size) ** 2

        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                                           norm_layer=norm_layer) for i in range(depth)])
        if self.cross_att:
            self.cross_num = kwargs.get('cross_num', 4)
            self.cross_inter = depth // self.cross_num
            self.cross_blocks = nn.ModuleList([CrossBlock(embed_dim, num_heads, mlp_ratio=2., qkv_bias=qkv_bias, drop=0.,
                                                          act_layer=nn.GELU, norm_layer=nn.LayerNorm, nview=5) for _ in range(self.cross_num)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic', align_corners=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        self.height = h // self.patch_size
        self.width = w // self.patch_size
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, h, w)

        return self.pos_drop(x)

    def forward(self, x, src_epipoles=None):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.cross_att and (i + 1) % self.cross_inter == 0:
                x = self.cross_blocks[(i + 1) // self.cross_inter - 1](x, src_epipoles, self.height, self.width)
        x = self.norm(x)
        return x

    def forward_with_last_att(self, x):
        x = self.prepare_tokens(x)
        att = None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, att = blk(x, return_attention=True)
        x = self.norm(x)
        return x, att

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
