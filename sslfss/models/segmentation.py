import torch.nn as nn
from torch import Tensor
from monai.networks.nets import UNet


class UNetSegmenter(nn.Module):
    """MONAI UNet backbone for MAML-based binary segmentation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=("instance", {"affine": True}),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class UNetBodyHead(nn.Module):
    """Body/head split of UNetSegmenter for ANIL meta-learning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_channels: int = 16,
    ) -> None:
        super().__init__()
        self.body = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=("instance", {"affine": True}),
        )
        self.head = nn.Conv2d(feature_channels, out_channels, kernel_size=1)


    def forward_body(self, x: Tensor) -> Tensor:
        """Return feature maps from the body (no head)."""
        return self.body(x)

    def forward_head(self, features: Tensor) -> Tensor:
        """Return logits from the head given pre-computed feature maps."""
        return self.head(features)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))
