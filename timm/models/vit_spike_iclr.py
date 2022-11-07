import torch
import torch.nn as nn

from timm.models.layers import to_2tuple, trunc_normal_, DropPath, HoyerBiAct, HoyerBiAct1d
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
from .helpers import build_model_with_cfg
# from timm.models import create_model

__all__ = ['spikformer']

def compute_non_zero_rate(x):
    x_shape = torch.tensor(list(x.shape))
    all_neural = torch.prod(x_shape)
    z = torch.nonzero(x)
    print("After attention proj the none zero rate is", z.shape[0]/all_neural)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = HoyerBiAct(num_features=hidden_features, spike_type='cw')

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = HoyerBiAct(num_features=out_features, spike_type='cw')

        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B,self.c_hidden,H,W).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B,C,H,W).contiguous()
        x = self.fc2_lif(x)
        return x


class spiking_self_attention(nn.Module):
    # SSA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)

        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = HoyerBiAct1d(num_features=dim, spike_type='cw')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = HoyerBiAct1d(num_features=dim, spike_type='cw')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = HoyerBiAct1d(num_features=dim, spike_type='cw')

        self.attn_lif = HoyerBiAct1d(num_features=dim, spike_type='cw')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = HoyerBiAct(num_features=dim, spike_type='cw')

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.flatten(2)
        B, C, N = x.shape
        x_for_qkv = x

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(2, 3).reshape(B, C, N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(B,C,H,W))

        return x


class encoder_block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = spiking_self_attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class sps(nn.Module):
    #spiking patch splitting
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_chans=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_chans
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.block1_conv = nn.Conv2d(in_chans, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims//8)
        self.block1_lif = HoyerBiAct(num_features=embed_dims//8, spike_type='cw')

        self.block2_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims//4)
        self.block2_lif = HoyerBiAct(num_features=embed_dims//4, spike_type='cw')

        self.block3_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims//2)
        self.block3_lif = HoyerBiAct(num_features=embed_dims//2, spike_type='cw')
        

        self.block4_conv = nn.Conv2d(embed_dims//2, embed_dims//1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims//1)
        # *img_size_h*img_size_w//(16*16*2*2)
        self.block4_lif =HoyerBiAct(num_features=embed_dims//1, spike_type='cw')
        

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = HoyerBiAct(num_features=embed_dims, spike_type='cw')
        # change the position of maxpooling layer, or change the stride of conv2d layer

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.block1_conv(x)
        x = self.block1_mp(x)
        x = self.block1_bn(x).reshape(B, -1, H//2, W//2).contiguous()
        x = self.block1_lif(x).contiguous()

        x = self.block2_conv(x)
        x = self.block2_mp(x)
        x = self.block2_bn(x).reshape(B, -1, H//4, W//4).contiguous()
        x = self.block2_lif(x).contiguous()
        

        x = self.block3_conv(x)
        x = self.block3_mp(x)
        x = self.block3_bn(x).reshape(B, -1, H//8, W//8).contiguous()
        x = self.block3_lif(x).contiguous()
        

        x = self.block4_conv(x)
        x = self.block4_mp(x)
        x = self.block4_bn(x).reshape(B, -1, H//16, W//16).contiguous()
        x = self.block4_lif(x).contiguous()
        

        x_feat = x.reshape(B, -1, H//16, W//16).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(B, -1, H//16, W//16).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)

class spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2],
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        # self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = sps(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_chans=in_chans,
                                 embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        block = nn.ModuleList([encoder_block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.flatten(2).mean(2)

    def forward(self, x):
        # x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _create_vit_snn(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(spikformer, variant, pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_snn(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=12, mlp_ratios=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
         **kwargs)
    return _create_vit_snn('vit_snn', pretrained=pretrained, **model_kwargs)



if __name__ == '__main__':
    H = 128
    W = 128
    x = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'vit_snn',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_chans=3, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()
    model.eval()
    y = model(x)
    print(y.shape)
    print('Test Good!')
