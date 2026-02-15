# EmoWOZ QLoRA Hyperparameter Tuning (Controller + Subprocess)

This repo fine-tunes **Llama-2-7b-chat** using **QLoRA (4-bit NF4 + LoRA)** on the **EmoWOZ** dataset, with:
- `controller.py` orchestrating trials as subprocesses (prevents VRAM build-up)
- `train_one_trial.py` training + scoring one trial
- `final_train.py` training the final model with best params
- a notebook-friendly inference snippet (Option A)

## Folder layout
- `scripts/` executable scripts
- `src/` reusable utilities (data prep, judge, model builder)
- `configs/` tuning search space
- `docs/` instructions
- `notebooks/` inference snippet

## Quickstart (Colab recommended)
See: `docs/COLAB_SETUP.md`

## Outputs
- Trial results: `runs/.../result.json`
- Summary: `runs/all_results.csv`
- Best params: `runs/best_params.json`
- Final adapter + tokenizer: `./llama2-emowoz-best/`

## Inference (Option A)
After training completes, run the snippet in `notebooks/inference_cell.py`.

## Workflow Overview

This repository uses a controller + subprocess approach to avoid VRAM accumulation during hyperparameter tuning.

### Scripts
- `train_one_trial.py`: Runs one hyperparameter trial in its own process. Trains briefly and evaluates using a Flan-T5 judge. Writes `result.json` in the trial output folder.
- `controller.py`: Runs multiple trials as subprocesses, selects the best config, then runs final training as a separate subprocess.
- `final_train.py`: Trains the final model using the best hyperparameters and saves the LoRA adapter + tokenizer to `./llama2-emowoz-best`.

### Colab
See `notebooks/colab_pipeline.py` for a copy-paste Colab flow:
- installs and environment checks
- Hugging Face login
- Google Drive mount
- controller run
- inference example
