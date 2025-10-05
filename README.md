# ViT-GPT2 Video Captioning

A from-scratch rewrite of the project with a unified pipeline that downloads the MSVD dataset, prepares manifest files, and trains a ViT encoder–GPT-2 decoder model end-to-end. Feature extraction now happens inline during training: sampled video frames are encoded by a Vision Transformer and fed directly into a GPT-2 decoder with cross-attention.

## Repository Layout

- `environment.txt` – bootstrap script that creates the conda environment and installs dependencies.
- `download_data.py` – downloads the MSVD archive from the official UT Austin mirror.
- `prepare_data.py` – parses captions and builds train/val/test manifests.
- `train.py` – launches training with on-the-fly feature extraction.
- `run_pipeline.py` – end-to-end entry point that downloads, prepares, and trains in one step.
- `src/` – Python modules for configuration, data handling, models, and the trainer.
  - `src/data/metadata.py` – fetches MSVD caption annotations when missing.

## Getting Started

1. **Environment**
   ```bash
   bash environment.txt
   conda activate vit_gpt2_captioning
   ```

2. **Download the dataset**
   ```bash
   python download_data.py
   ```
   The archive is fetched to `data/raw/YouTubeClips.tar`, extracted into `data/raw/YouTubeClips`, and manifests are generated automatically at `data/processed/msvd/{train,val,test}.json`.

3. *(Optional)* **Regenerate manifests**
   ```bash
   python prepare_data.py
   ```
   Use this if you change the dataset root or want to re-split with a different seed. Captions are downloaded automatically when missing.

4. **Train**
   ```bash
   python train.py --epochs 5 --batch-size 2 --device cuda
   ```
   Training outputs checkpoints and qualitative samples under `outputs/`.
   To target specific GPUs, append `--gpu-ids`, e.g. `--gpu-ids 0,1` to use two devices in parallel (default is `0,2`).

### End-to-End Pipeline

To perform download, preparation, and training in a single command, run:

```bash
python run_pipeline.py --epochs 5 --batch-size 2 --device cuda
```

All training CLI flags (`--data-root`, `--output-dir`, `--learning-rate`, `--num-workers`, `--gradient-accumulation`, `--num-frames`, etc.) are also accepted by the pipeline and forwarded to the trainer.

Progress for long-running steps (dataset download, tar extraction, caption parsing, and train/validation epochs) is shown via `tqdm` progress bars.

When executed without overrides, the pipeline assumes an RTX A5000 configuration: `--epochs 20`, `--batch-size 16`, `--learning-rate 3e-5`, `--device cuda`, `--num-workers 8`, `--gradient-accumulation 1`, `--num-frames 16`, and `--gpu-ids 0,2`.
Adjust `--gpu-ids` to select CUDA devices, e.g. `--gpu-ids 0,1` or `--gpu-ids 0,1,2,3` for multi-GPU training.

## Data Preparation Details

The preparation script scans the extracted video directory for clip files (any of `.mp4`, `.avi`, `.flv`, `.mov`, `.mpeg`, `.mpg`). Captions are loaded from the MSVD `video_corpus.csv` metadata and grouped per clip. A deterministic 80/10/10 split is created using a fixed seed.

## Model Architecture

- **Encoder** – `torchvision` ViT-B/16 pretrained on ImageNet. Each video contributes a configurable number of frames. Frames are sampled uniformly, resized to 224×224, normalized with ImageNet statistics, and encoded. The [CLS] token across frames acts as the set of encoder states.
- **Bridge** – LayerNorm + Linear projection adapts ViT embeddings to the GPT-2 hidden size.
- **Decoder** – Hugging Face GPT-2 with cross-attention enabled. The decoder attends over the projected frame embeddings while generating captions. Teacher forcing is used during training, and generation runs through the standard `generate` API for evaluation samples.

## Training Loop

The trainer uses mixed precision when CUDA is available, gradient accumulation, and gradient clipping. After every epoch the script:
- Logs train/val losses and perplexities.
- Writes checkpoints (`model.pt`, `optimizer.pt`, `metrics.json`).
- Generates qualitative captions for a small batch and saves them to `epoch_XXX_samples.json`.

Adjustable CLI options include batch size, epochs, learning rate, gradient accumulation, number of frames per clip, and the dataset/output directories.

## Additional Notes

- Install FFmpeg (the environment script installs it via conda) so that `torchvision.io.read_video` can decode clips.
- Large intermediate assets (videos, manifests, checkpoints) live under `data/` and `outputs/` outside version control by default.
- To resume training or fine-tune, point `train.py` to a processed manifest directory and the desired checkpoint under `outputs/` and load weights manually before calling `trainer.fit()`.
