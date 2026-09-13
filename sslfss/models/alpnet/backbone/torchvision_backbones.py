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
import torch.nn as nn
import torchvision


class TVDeeplabRes101Encoder(nn.Module):
    """FCN-Resnet101 backbone from torchvision deeplabv3."""
    def __init__(self, use_coco_init, aux_dim_keep=64, use_aspp=False):
        super().__init__()
        if use_coco_init:
            print("###### NETWORK: Using ms-coco initialization ######")
            try:
                # torchvision >= 0.13
                from torchvision.models.segmentation import (
                    DeepLabV3_ResNet101_Weights,
                    deeplabv3_resnet101,
                )
                _model = deeplabv3_resnet101(
                    weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
                    progress=True,
                    aux_loss=None,
                )
            except (ImportError, AttributeError):
                # torchvision < 0.13 fallback
                _model = torchvision.models.segmentation.deeplabv3_resnet101(
                    pretrained=True, progress=True, num_classes=21, aux_loss=None)
        else:
            print("###### NETWORK: Training from scratch ######")
            try:
                from torchvision.models.segmentation import deeplabv3_resnet101
                _model = deeplabv3_resnet101(weights=None, progress=True, aux_loss=None)
            except (ImportError, AttributeError):
                _model = torchvision.models.segmentation.deeplabv3_resnet101(
                    pretrained=False, progress=True, num_classes=21, aux_loss=None)

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)
        self.asppconv = nn.Conv2d(256, 256, kernel_size=1, bias=False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256])
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level=False):
        fts = self.backbone(x_in)
        if self.use_aspp:
            high_level_fts = self.aspp_out(fts['out'])
        else:
            high_level_fts = self.localconv(fts['out'])

        if low_level:
            low_level_fts = fts['aux'][:, :self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts
