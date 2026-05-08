# SEVPO

SEVPO implementation and training code for offline safe reinforcement learning experiments.

## Repository Structure

- `code/train/train.py`: training entry point.
- `code/configs/sevpo_config.py`: default SEVPO experiment configuration.
- `code/model/`: agent, dataset, evaluation, network, and wrapper code.
- `code/env/`: task list and toy environment.
- `assets/sevpo-demo.mp4`: demo video.

## Setup

```bash
pip install -r requirements.txt
```

## Train

Run from the `code` directory so local imports resolve correctly:

```bash
cd code
python -m train.train --config=configs/sevpo_config.py:sevpo --env_id=9
```

Use `--project` to set the Weights & Biases project name and `--ratio` to control dataset sampling.
