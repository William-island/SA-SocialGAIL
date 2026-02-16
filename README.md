# SA-SocialGAIL

This repository implements a GNN-based GAIL framework with disentangled latent codes to capture social behaviors such as speed and comfort in dense crowds.


---

## 1. Overview

SA-SocialGAIL builds on adversarial imitation learning (GAIL) with three key ingredients:

- **Graph-based crowd observation**  
  Pedestrians are modeled as nodes in a graph with edges encoding local interactions. A graph neural network encodes the crowd state into features.

- **Disentangled latent codes**  
  Latent variables (e.g. speed) are used to modulate the policy, allowing controllable or interpretable variations in behavior. The implementation supports categorical latent codes defined in YAML configs.

- **Adversarial imitation with auxiliary feature learning**  
  A discriminator is trained on expert vs. policy trajectories, while additional Q-networks with variational information bottleneck (VIB) learn to predict handcrafted features from the GNN representation.

The main algorithm is implemented in  
`core/algos/sa_socialgail.py` as the class **`SA_SOCIAL_GAIL`**, which extends a diversity-aware PPO backbone (`DA_PPO`).

---

## 2. Repository Structure

Only the most relevant files are listed here:

- **Top-level scripts**
  - `sgi_train_imitation.py`: entry point for training SA-SocialGAIL on crowd datasets.
  - `sgi_test_imitation.py`: evaluation and visualization utilities (speed, social comfort, etc.).

- **Core modules (`core/`)**
  - `core/env.py`: wrapper to create `CrowdEnv` and apply action normalization (`NormalizedEnv`).
  - `core/crowd_env/` or `core/datasets/gym_env.py`: implementation of the crowd simulation / replay environment, including:
    - loading preprocessed trajectory datasets (`.pkl`);
    - constructing graph observations around each agent;
    - computing metrics like social comfort, social gradient, Frechet distance, Hausdorff distance, etc.
  - `core/algos/sa_socialgail.py`: main SA-SocialGAIL algorithm, including:
    - policy update (PPO-style);
    - GAIL-style discriminator update;
    - latent-feature Q-nets with VIB loss.
  - `core/network/`: neural network definitions
    - graph-based policies and value functions;
    - discriminators (`GraphDiscrim_Info`, etc.);
    - graph Q networks with VIB (`SGI_GraphQ_VIB`).
  - `core/buffer.py`: rollout buffers, including `RolloutBuffer_for_Latent_Feature` to store both trajectories and latent-related features.
  - `core/trainer.py`: training loop, logging to TensorBoard and saving models.
  - `core/tester.py`: testing utilities for different behavioral metrics.

- **Configs**
  - `configs/algo_sa_socialgail.yaml`: algorithm and environment hyperparameters (steps, rollout length, graph obs length, latent code definition, etc.).
  - `core/crowd_env/datasets/datasets_configs.yaml`: dataset-specific settings (paths, scale factors, map size, etc.), indexed by keys such as `GC`, `synthetic_w_speed`, etc.

---

## 3. Requirements

The project is implemented in Python with PyTorch and PyTorch Geometric, and uses a Gym-style environment.

A non-exhaustive list of main dependencies:

- Python 3.x
- `torch`
- `torch_geometric`
- `gym`
- `numpy`, `scipy`
- `tqdm`
- `tensorboard`
- `pygame`, `imageio`
- `hausdorff`
- `pyyaml`
- `stable-baselines3` (only for optional environment checking in `gym_env.py`)
- `rvo2` (Python-RVO2) for RVO-based utilities

> Note: Exact versions are not strictly enforced here. For reproducible experiments, please set and freeze versions according to your own environment.

---

## 4. Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-account/SA-SocialGAIL.git
cd SA-SocialGAIL

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3. Install common dependencies (example)
pip install torch torchvision torchaudio
pip install torch-geometric
pip install gym numpy scipy tqdm tensorboard pygame imageio hausdorff pyyaml stable-baselines3

# 4. Install RVO2 (Python-RVO2), if needed
# (may need CMake, a C++ compiler, and Cython)
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install .
cd ..
```

You may adapt this installation to your cluster or local environment.

---

## 5. Datasets

The environment expects **preprocessed crowd trajectory datasets** serialized as `.pkl` files. A typical dataset dictionary contains:

- `trajectories`: per-person trajectory data with time stamps and derived velocities;
- `train_ids`: IDs used for training;
- `test_ids`: IDs used for evaluation.

Dataset paths and settings are specified in  
`core/crowd_env/datasets/datasets_configs.yaml`, e.g.:

- `GC`: real-world crowd dataset (e.g., Grand Central-like scenes).
- `synthetic_w_speed`: synthetic dataset with explicit speed-related properties.

Please prepare your datasets and update the corresponding YAML entries:

```yaml
GC:
  dataset_path: ./datasets/gc_interpolated_trajectory.pkl
  scale_factor: 2.0
  # other environment-specific parameters...

synthetic_w_speed:
  dataset_path: ./datasets/synthetic_speed.pkl
  scale_factor: 4.0
  # ...
```

> We do not ship original datasets in this repository. Please follow the dataset licenses and preprocessing steps described in the paper or your own pipeline.

---

## 6. Configurations

Algorithm-related hyperparameters (rollout length, graph observation length, latent code definition, etc.) are defined in:

- `configs/algo_sa_socialgail.yaml`

Key fields include:

- `num_steps`, `rollout_length`, `eval_interval`
- `regions`, `radius`, `graph_obs_past_len`, `padd_to_number`
- `latent_code` block, e.g.:
  ```yaml
  latent_code:
    speed:
      type: categorical
      dim: 2
      feature_dim: 1
      store_dim: 1
  ```
- `reward_us_coef`, `auto_balancing` for latent-related auxiliary rewards.

Dataset-related configurations are merged from `core/crowd_env/datasets/datasets_configs.yaml` according to `--dataset`.

---

## 7. Training

The main training script is `sgi_train_imitation.py`.

### 7.1 Basic usage

```bash
python sgi_train_imitation.py \
    --algo sa_socialgail \
    --dataset GC
```

The script will:

1. Load `configs/algo_sa_socialgail.yaml` according to `--algo`.
2. Load dataset-specific settings from `core/crowd_env/datasets/datasets_configs.yaml` according to `--dataset`.
3. Create training and testing environments via `core/env.py`.
4. Initialize the expert buffer `SerializedBuffer_SA` (pointed to by `args.buffer`).
5. Train the algorithm and periodically evaluate it.

Training logs and checkpoints are stored under:

```text
logs/{dataset}/{algo}/seed{seed}-{timestamp}/
    ├── summary/    # TensorBoard logs
    ├── model/      # saved model checkpoints (.pt)
    └── config.yaml # snapshot of args
```

You can visualize training curves with:

```bash
tensorboard --logdir logs
```

---





