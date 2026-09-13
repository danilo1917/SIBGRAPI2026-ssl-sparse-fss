# Adapted from SSL-ALPNet (Ouyang et al., ECCV 2020):
# https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
#
# MIT License
#
# Copyright (c) 2020 Cheng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
from torch import nn
from torch.nn import functional as F


class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode='bilinear'):
        """ALPModule Args: proto_grid:  Grid size when doing multi-prototyping."""
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)

    def forward(self, qry, sup_x, sup_y, mode, thresh,
                isval=False, val_wsize=None, vis_sim=False, **kwargs):
        """Args: mode:       'mask' / 'gridconv' / 'gridconv+' qry:        [way(1), nb(1), nc, h, w] sup_x:      [way(1), shot, nb(1), nc, h, w] sup_y:      [way(1), shot, nb(1), h, w] vis_sim:    visualize raw similarities or not"""
        qry = qry.squeeze(1)          # [way, nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1)   # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0)              # [nshot, 1, h, w]

        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)
            x_norm = torch.max(x_norm, torch.ones_like(x_norm) * eps)   # device-agnostic fix
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask':  # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5)  # nb x C
            proto = proto.mean(dim=0, keepdim=True)  # 1 x C
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps=1e-4) * 20.0
            vis_dict = {'proto_assign': None}
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict

        elif mode == 'gridconv':  # local prototypes only
            input_size = qry.shape
            nch = input_size[1]
            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]  # npro x nc
            pro_n = safe_norm(protos)
            qry_n = safe_norm(qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20
            pred_grid = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float().detach()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()
            return pred_grid, [debug_assign], vis_dict

        elif mode == 'gridconv+':  # local + global prototypes
            input_size = qry.shape
            nch = input_size[1]
            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]

            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5)

            pro_n = safe_norm(torch.cat([protos, glb_proto], dim=0))
            qry_n = safe_norm(qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20
            pred_grid = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()
            return pred_grid, [debug_assign], vis_dict

        else:
            raise NotImplementedError(f"mode '{mode}' not supported")
