from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """Settings related to data storage and sampling."""

    root_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    dataset_name: str = "msvd"
    download_url: str = (
        "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"
    )
    captions_urls: tuple[str, ...] = (
        "https://www.cs.utexas.edu/~ml/clamp/videoDescription/AllVideoDescriptions.txt",
        "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/AllVideoDescriptions.txt",
        "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/MSRVideoDescriptionCorpus.csv",
        "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/MSRVideoDescriptionCorpus.zip",
        "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/corpus/MSRVideoDescriptionCorpus.csv",
        "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/corpus/MSRVideoDescriptionCorpus.zip",
    )
    captions_filename: str = "MSRVideoDescriptionCorpus.csv"
    num_frames: int = 8
    frame_shape: tuple[int, int] = (224, 224)
    frame_sample_rate: float = 2.0
    max_caption_tokens: int = 64
    min_caption_tokens: int = 3


@dataclass
class ModelConfig:
    """Model checkpoints and training behavior."""

    vit_variant: str = "vit_b_16"
    gpt2_checkpoint: str = "gpt2"
    train_encoder: bool = False
    train_decoder: bool = True
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Hyperparameters for the trainer."""

    batch_size: int = 2
    num_workers: int = 4
    epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "cuda"
    log_every: int = 10
    save_every: int = 1
    seed: int = 42
    output_dir: Path = Path("outputs")
    gpu_ids: tuple[int, ...] = ()


@dataclass
class GenerationConfig:
    """Generation parameters for qualitative evaluation."""

    num_beams: int = 3
    max_length: int = 32
    temperature: float = 1.0


@dataclass
class ProjectConfig:
    """Container for all configuration sections."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def ensure_directories(self) -> None:
        self.dataset.root_dir.mkdir(parents=True, exist_ok=True)
        self.dataset.raw_dir.mkdir(parents=True, exist_ok=True)
        self.dataset.processed_dir.mkdir(parents=True, exist_ok=True)
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
