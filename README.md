# Adversarial Skill Embeddings applied to humanoid robotics in Genesis

This repository contains my work on **Adversarial Skill Embeddings (ASE)** applied to **humanoid control** using the **Unitree H1** robot, simulated in the **Genesis** physics engine. The provided low-level policy (`h1-forward-real/model_6000.pt`) can be deployed in `unitree_mujoco` or in real-life (IRL) scenarios, enabling the robot to walk forward and backward. For more information, please see the [internship report repository](https://github.com/Planeurzik/jrl-internship-repo).


![H1 walking](illustration/walking_h1.gif)

## Repository structure

Directory       | Description                                                                 |
 |-----------------|-----------------------------------------------------------------------------|
 | `h1/`           | Contains robot assets.                                                      |
 | `high_level/`   | High-level policy training and deployment.                                  |
 | `illustration/` | Visual assets, including GIFs and images.                                   |
 | `low_level/`    | Low-level policy training, evaluation, and deployment scripts.              |
 | `unitree_sdk2`         | Use it only for sim-to-sim or sim-to-real deployment.                  |

## Getting started

### 1. Install dependencies

You’ll need **Python ≥ 3.12** and the following packages:

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with Genesis, ensure you are using the correct version. This project was developed with Genesis at commit [`9494f35`](https://github.com/Genesis-Sim/Genesis/commit/9494f35) ("init torch contact"). Install it manually if needed.

For MuJoCo compatibility (to observe sim-to-sim gaps), install `mujoco-3.2.7`. Refer to the [unitree_mujoco GitHub repo](https://github.com/unitreerobotics/unitree_mujoco) for details.

### 2. Run evaluation in Genesis

To evaluate the pre-trained low-level policy:

```bash
cd low_level
python h1_eval.py -e h1-forward-real --ckpt 6000
```

This policy was trained with **domain randomization**, **observation noise**, and **privileged observations** for the critic. Modify the code as needed for deployment in MuJoCo or IRL.

To deploy in MuJoCo or IRL:

```bash
cd low_level
python h1_eval.py -e h1-forward-real --ckpt 6000 --env_real_type
```

### 3. Train low/high-level policy

To train a policy:

```bash
cd low_level  # or cd high_level
python h1_train.py
```

Training logs and checkpoints will be saved in the `logs/` directory.

### 4. Customizing Rewards

To add or modify rewards for the high-level policy, edit the `h1_env.py` file. The weights corresponding to the rewards can be adjusted in `h1_train.py`.