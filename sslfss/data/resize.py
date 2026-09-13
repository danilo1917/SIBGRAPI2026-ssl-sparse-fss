import numpy as np
import torch
import torch.nn.functional as F


def redimensionar_corte(
    arr_2d: np.ndarray,
    target_size: tuple = (128, 128),
    mode: str = "bilinear",
) -> np.ndarray:
    """Resize a 2-D array to ``target_size``."""
    t = torch.tensor(arr_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kwargs = {"size": target_size, "mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    t = F.interpolate(t, **kwargs)
    return t.squeeze().numpy()
