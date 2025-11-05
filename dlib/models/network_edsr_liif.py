import sys
from os.path import join, dirname, abspath
import math
from argparse import Namespace

import torch
from torch import nn
import torch.nn.functional as F


from torch import einsum
from einops import rearrange

# Credit: https://github.com/yinboc/liif/tree/main
# Paper: https://arxiv.org/pdf/2012.09161


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants


__all__ = ['EDSR_LIIF']


# EDSR =========================================================================

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self,
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self,
                 conv,
                 n_feats,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(False),
                 res_scale=1.
                 ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


# @register('edsr-baseline')
def make_edsr_baseline(in_chans: int = 3, n_resblocks=16, n_feats=64,
                       res_scale=1.,
                       scale=2, no_upsampling=False, rgb_range=1.):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    assert no_upsampling, no_upsampling  # to use LIIF instead of
    # upsampling.
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = in_chans
    return EDSR(args)


# @register('edsr')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)


# ==============================================================================


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class EDSR_LIIF(nn.Module):
    def __init__(self,
                 in_chans: int = 3,
                 n_resblocks: int = 16,
                 n_feats: int = 64,
                 res_scale: float = 1.,
                 scale: int = 2,
                 rgb_range: float = 1.,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True
                 ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        assert isinstance(scale, int), type(scale)
        assert scale > 0, scale
        assert isinstance(in_chans, int), type(in_chans)
        assert in_chans > 0, in_chans

        self.scale = scale
        self.in_chans = in_chans

        self.encoder = make_edsr_baseline(in_chans=in_chans,
                                          n_resblocks=n_resblocks,
                                          n_feats=n_feats,
                                          res_scale=res_scale,
                                          scale=scale,
                                          no_upsampling=True,
                                          rgb_range=rgb_range)

        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim = imnet_in_dim * 9
        imnet_in_dim = imnet_in_dim + 2  # attach coord
        if self.cell_decode:
            imnet_in_dim = imnet_in_dim + 2

        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=in_chans,
                         hidden_list=[256, 256, 256, 256]
                         )

    def gen_feat(self, inp):
        return self.encoder(inp)

    def create_hr_coords(self, low_h, low_w):
        h = low_h * self.scale
        w = low_w * self.scale
        shape = [h, w]

        coord_seqs = []
        for i, n in enumerate(shape):
            v0, v1 = -1, 1
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)

        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        # flatten:
        coords = ret.view(-1, ret.shape[-1])  # h*w, 2
        return coords

    def query_rgb(self, feat):
        assert feat.ndim == 4, feat.ndim  # bsz, d, low_h, low_w
        bsz, _, low_h, low_w = feat.shape
        device = feat.device

        hr_coords = self.create_hr_coords(low_h, low_w)  # hr_h * hr_w, 2

        cell = torch.ones_like(hr_coords)
        h = low_h * self.scale
        w = low_w * self.scale
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell = cell.to(device)
        hr_coords = hr_coords.to(device)
        # repeat
        cell = cell.repeat(bsz, 1, 1)  # bs, hr_h * hr_w, 2
        hr_coords = hr_coords.repeat(bsz, 1, 1)

        if self.imnet is None:
            ret = F.grid_sample(feat,
                                hr_coords.flip(-1).unsqueeze(1),
                                mode='nearest',
                                align_corners=False)[:, :, 0, :].permute(
                0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            # bsz, nft * 9, lr_h, lr_w

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2  # lr_h
        ry = 2 / feat.shape[-1] / 2  # lr_w

        feat_coord = make_coord(
            feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(
            0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = hr_coords.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = hr_coords - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = hr_coords.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        # ret: bsz, hr_h*hr_w, self.in_chans
        assert ret.ndim == 3, ret.ndim
        # reshape back to image
        sr_img = ret.permute(0, 2, 1)  # bsz, self.in_chans, hr_h*hr_w
        sr_img = sr_img.view(bsz, self.in_chans, h, w)

        return sr_img

    def forward(self, inp):
        # inp: bsz, c, h, w : low res.
        assert inp.ndim == 4, inp.ndim

        feat = self.gen_feat(inp)
        return self.query_rgb(feat)


if __name__ == '__main__':

    device = torch.device('cuda:0')
    scale = 4
    low_h = 27
    low_w = 27
    c = 3
    b = 2
    x = torch.rand(b, c, low_h, low_w).to(device)
    edsr_model = make_edsr_baseline(in_chans=c,
                                    n_resblocks=16,
                                    n_feats=64,
                                    res_scale=1,
                                    scale=scale,
                                    no_upsampling=True,
                                    rgb_range=1.
                                    ).to(device)
    with torch.no_grad():
        out = edsr_model(x)
        # print(x.shape, out.shape, scale)
        # torch.Size([1, 3, 8, 8]) torch.Size([1, 64, 8, 8]) 8

    # EDSR-LIIF
    model_liff = EDSR_LIIF(in_chans=c,
                           n_resblocks=16,
                           n_feats=64,
                           res_scale=1,
                           scale=scale,
                           rgb_range=1.
                           ).to(device)

    with torch.no_grad():
        out = model_liff(x)
        print(out.shape, x.shape, scale)
        # torch.Size([2, 3, 108, 108]) torch.Size([2, 3, 27, 27]) 4
