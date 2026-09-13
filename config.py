from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    train_datasets_dir: Path = field(default_factory=lambda: Path("data/raw/train"))
    test_datasets_dir:  Path = field(default_factory=lambda: Path("data/raw/test"))
    processed_dir:      Path = field(default_factory=lambda: Path("data/processed"))
    checkpoints_dir:    Path = field(default_factory=lambda: Path("checkpoints"))
    results_dir:        Path = field(default_factory=lambda: Path("results"))

    n_volumes:        int   = None      # max NIfTI volumes per dataset (None = all)
    target_size:      tuple = (256, 256)
    slice_axis:       int   = 2         # 0=sagittal, 1=coronal, 2=axial
    # Minimum fraction of foreground pixels to keep a slice.
    # Filters trivial edge slices (e.g. 3 px of anatomy).
    # 0.001 on 128×128 → ≥ ~16 foreground pixels required.
    min_fg_fraction:  float = 0.001
    val_frac:         float = 0.20      # fraction of labelled Tr volumes held out for validation

    seed:     int = 42
    val_seed: int = 123   # separate seed for val pseudo-label cache (reproducible independently)

    n_episodes_target:  int   = 36000
    log_interval:       int   = 500
    val_log_episodes:   int   = 200
    grad_clip:          float = 1.0

    test_episodes: int = 0
    eval_mode:     str = "few_shot"
    k_shot:        int = 5

    in_channels:  int = 1
    out_channels: int = 1

    loss_bce_weight:  float = 0.4
    loss_dice_weight: float = 0.6

    dense_label_mode: str = "pseudo"   # "pseudo" | "real"

    run_name: str = "sslfss_maml"

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in (self.processed_dir, self.run_checkpoints_dir, self.run_results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def discover_datasets(base_dir: Path) -> list[Path]:
        """Auto-discover dataset directories under *base_dir*."""
        base = Path(base_dir)
        if not base.exists():
            return []
        return sorted(
            d for d in base.iterdir()
            if d.is_dir() and (d / "imagesTr").is_dir()
        )

    @property
    def run_checkpoints_dir(self) -> Path:
        return Path(self.checkpoints_dir) / self.dense_label_mode

    @property
    def run_results_dir(self) -> Path:
        return Path(self.results_dir) / self.dense_label_mode

    @property
    def checkpoint_path(self) -> Path:
        return self.run_checkpoints_dir / f"{self.run_name}.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.run_checkpoints_dir / f"{self.run_name}_last.pt"