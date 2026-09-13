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
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslfss.models.alpnet.alpmodule import MultiProtoAsConv
from sslfss.models.alpnet.backbone.torchvision_backbones import TVDeeplabRes101Encoder

# prototype modes
FG_PROT_MODE = 'gridconv+'  # local + global
BG_PROT_MODE = 'gridconv'   # local only

FG_THRESH = 0.95
BG_THRESH = 0.95


class FewShotSeg(nn.Module):
    """ALPNet Args: in_channels: Number of input channels cfg:         Model configuration dict"""
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_cls()

    def get_encoder(self, in_channels):
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        else:
            raise NotImplementedError(
                f'Backbone {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path))
            print(f'Pre-trained model {self.pretrained_path} loaded.')

    def get_cls(self):
        proto_hw = self.config['proto_grid_size']
        feature_hw = self.config['feature_hw']
        assert self.config['cls_name'] == 'grid_proto'
        self.cls_unit = MultiProtoAsConv(
            proto_grid=[proto_hw, proto_hw],
            feature_hw=feature_hw,
        )

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs,
                isval=False, val_wsize=None, show_viz=False):
        """Args: supp_imgs:  way x shot x [B x 3 x H x W] fore_mask:  way x shot x [B x H x W] back_mask:  way x shot x [B x H x W] qry_imgs:   N x [B x 3 x H x W]"""
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-way not implemented"
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]
        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs]
            + [torch.cat(qry_imgs, dim=0)],
            dim=0,
        )

        img_fts = self.encoder(imgs_concat, low_level=False)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)

        fore_mask = torch.stack(
            [torch.stack(way, dim=0) for way in fore_mask], dim=0)
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
        back_mask = torch.stack(
            [torch.stack(way, dim=0) for way in back_mask], dim=0)

        align_loss = 0
        outputs = []
        bg_sim_maps_all = []
        fg_sim_maps_all = []
        assign_maps_all = []

        for epi in range(1):
            res_fg_msk = torch.stack([
                F.interpolate(fore_mask_w, size=fts_size, mode='bilinear')
                for fore_mask_w in fore_mask
            ], dim=0)
            res_bg_msk = torch.stack([
                F.interpolate(back_mask_w, size=fts_size, mode='bilinear')
                for back_mask_w in back_mask
            ], dim=0)

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

            _raw_score, _, aux_attr = self.cls_unit(
                qry_fts, supp_fts, res_bg_msk,
                mode=BG_PROT_MODE, thresh=BG_THRESH,
                isval=isval, val_wsize=val_wsize, vis_sim=show_viz,
            )
            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            for way, _msk in enumerate(res_fg_msk):
                fg_mode = (
                    FG_PROT_MODE
                    if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask'
                    else 'mask'
                )
                _raw_score, _, aux_attr = self.cls_unit(
                    qry_fts, supp_fts, _msk.unsqueeze(0),
                    mode=fg_mode, thresh=FG_THRESH,
                    isval=isval, val_wsize=val_wsize, vis_sim=show_viz,
                )
                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)  # N x (1+Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            if self.config['align'] and self.training:
                align_loss += self.alignLoss(
                    qry_fts[:, epi], pred, supp_fts[:, :, epi],
                    fore_mask[:, :, epi], back_mask[:, :, epi],
                )

        output = torch.stack(outputs, dim=1).view(-1, *outputs[0].shape[1:])
        assign_maps_t = torch.stack(assign_maps, dim=1)
        bg_sim_maps_t = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps_t = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps_t, fg_sim_maps_t], assign_maps_t

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """Prototype alignment loss."""
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1).unsqueeze(0)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = []

        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2)

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                img_fts = supp_fts[way:way + 1, shot:shot + 1]
                qry_pred_fg_msk = F.interpolate(
                    binary_masks[way + 1].float(), size=img_fts.shape[-2:], mode='bilinear')
                qry_pred_bg_msk = F.interpolate(
                    binary_masks[0].float(), size=img_fts.shape[-2:], mode='bilinear')
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts,
                    sup_y=qry_pred_bg_msk.unsqueeze(-3),
                    mode=BG_PROT_MODE, thresh=BG_THRESH,
                )
                scores.append(_raw_score_bg)

                fg_mode = (
                    FG_PROT_MODE
                    if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask'
                    else 'mask'
                )
                _raw_score_fg, _, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts,
                    sup_y=qry_pred_fg_msk.unsqueeze(-3),
                    mode=fg_mode, thresh=FG_THRESH,
                )
                scores.append(_raw_score_fg)

                supp_pred = F.interpolate(
                    torch.cat(scores, dim=1),
                    size=fore_mask.shape[-2:], mode='bilinear',
                )

                supp_label = torch.full_like(
                    fore_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0

                loss.append(
                    F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255)
                    / n_shots / n_ways
                )

        return torch.sum(torch.stack(loss))
