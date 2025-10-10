# Repository Guidelines

## Project Structure & Module Organization
Core modeling lives in `vit.py`, while shared helpers sit in `data_utils.py`, `video_utils.py`, and `text_processing.py`. Dataset setup scripts (`prepare_msrvtt_dataset.py`, `prepare_msvd_dataset.py`) populate `/Volumes/KINGSTON/dataset/<DATASET>` unless you override `root_folder`. Feature extraction entry points (`feature_extraction_mscoco.py`, `feature_extraction_vizwiz.py`) write frame embeddings inside each dataset’s `features_*` directories.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` — create and activate a local virtualenv.
- `pip install -r environment.txt` — install pinned PyTorch, transformers, and video tooling.
- `python download_datasets.py` — trigger dataset downloads; edit the script to target MSRVTT/MSVD as needed.
- `python feature_extraction_msrvtt.py` — precompute video embeddings before training (swap to MSVD variant as needed).
- `python vit_gpt2_training_v3_msrvtt.py` — launch full training; flip the dataset selector inside the script for MSVD.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case for variables and functions. Keep scripts executable via `if __name__ == "__main__":`. Document tensors and folder expectations with brief docstrings, and mirror existing dataset attribute names (e.g., `MSRVTTDataset.train_annotations`).

## Testing Guidelines
No automated suite ships here. Run targeted smoke checks: inspect dataloaders via `python data_utils.py --inspect-split train`, execute 5–10 batch dry runs of the training script, and confirm `results/test_result_<DATASET>.json` is produced. Track BLEU/CIDEr deltas using the built-in COCO evaluator hooks before landing modeling updates.

## Commit & Pull Request Guidelines
Write imperative, scope-focused commit subjects (e.g., `Fix MSRVTT caption parsing`). In pull requests, note dataset, command, and hardware, link related issues, and attach qualitative caption samples or TensorBoard screenshots for training changes. Rebase onto `main` before seeking review.

## Data & Configuration Notes
Override dataset paths using constructor parameters or environment variables instead of editing hardcoded strings. Do not commit raw data, checkpoints, or `results/` artifacts; update `.gitignore` if you introduce new generated folders.
