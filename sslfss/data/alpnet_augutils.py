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
import copy
import numpy as np
import torchvision.transforms as deftfx
import sslfss.data.alpnet_image_transforms as myit

sabs_aug = {
    # turn flipping off as medical data has fixed orientations
    'flip':    {'v': False, 'h': False, 't': False, 'p': 0.25},
    'affine':  {
        'rotate': 5,
        'shift':  (5, 5),
        'shear':  5,
        'scale':  (0.9, 1.2),
    },
    'elastic':     {'alpha': 10, 'sigma': 5},
    'patch':       256,
    'reduce_2d':   True,
    'gamma_range': (0.5, 1.5),
}


def get_geometric_transformer(aug, order=3):
    """order: interpolation degree. Use order=0 for segmentation masks."""
    affine = aug['aug'].get('affine', 0)
    alpha  = aug['aug'].get('elastic', {'alpha': 0})['alpha']
    sigma  = aug['aug'].get('elastic', {'sigma': 0})['sigma']
    flip   = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p': 0.125})

    tfx = []
    if 'flip' in aug['aug']:
        tfx.append(myit.RandomFlip3D(**flip))
    if 'affine' in aug['aug']:
        tfx.append(myit.RandomAffine(
            affine.get('rotate'),
            affine.get('shift'),
            affine.get('shear'),
            affine.get('scale'),
            affine.get('scale_iso', True),
            order=order,
        ))
    if 'elastic' in aug['aug']:
        tfx.append(myit.ElasticTransform(alpha, sigma))

    return deftfx.Compose(tfx)


def get_intensity_transformer(aug):
    """Basic gamma intensity transform."""
    def gamma_transform(img):
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin  = img.min()
            irange = img.max() - cmin + 1e-5
            img = img - cmin + 1e-5
            img = irange * np.power(img / irange, gamma)
            img = img + cmin
        elif gamma_range is False:
            pass
        else:
            raise ValueError(f"Cannot identify gamma range: {gamma_range}")
        return img
    return gamma_transform


def transform_with_label(aug):
    """Returns a callable that jointly augments image + label."""
    geometric_tfx = get_geometric_transformer(aug)
    intensity_tfx = get_intensity_transformer(aug)

    def transform(comp, c_label, c_img, use_onehot, nclass, **kwargs):
        comp = copy.deepcopy(comp)
        assert c_img + 1 == comp.shape[-1], "only single-channel 2D label allowed"

        _label  = comp[..., c_img]
        _h_label = np.float32(np.arange(nclass) == (_label[..., None]))
        comp = np.concatenate([comp[..., :c_img], _h_label], axis=-1)

        comp = geometric_tfx(comp)

        t_label_h = np.rint(comp[..., c_img:])
        assert t_label_h.max() <= 1
        t_img = comp[..., :c_img]
        t_img = intensity_tfx(t_img)

        if use_onehot:
            t_label = t_label_h
        else:
            t_label = np.expand_dims(np.argmax(t_label_h, axis=-1), axis=-1)

        return t_img, t_label

    return transform
