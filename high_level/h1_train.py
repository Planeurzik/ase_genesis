import argparse
import os
import pickle
import shutil

import torch

from rl_ase.custom_on_policy_runner import OnPolicyRunner

from datetime import datetime

import genesis as gs

from h1_env import H1Env


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [1024, 512],
            "critic_hidden_dims": [1024, 512],
            "init_noise_std": 1.0,
            "class_name": "Actor",
            "fixed_action_std": False,
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 10,
        "empirical_normalization": None,
        "seed": 1,
        "low_level_policy_model": "../low_level/logs/h1-forward-real/model_6000.pt",
        "low_level_policy_cfgs" : "../low_level/logs/h1-forward-real/cfgs.pkl",
        "high_level_division_frequency_factor": 10,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 19,
        "default_joint_angles": {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_joint": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
        "dof_names": [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ],
        # termination
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 20,
        "termination_if_base_height_less_than": 0.7,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.1],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "feet_names": [
            "left_ankle",
            "right_ankle",
        ],
        "joint_limits_torque" : {
            "left_hip_yaw_joint": 150.0, "left_hip_roll_joint": 150.0,
            "left_hip_pitch_joint": 150.0, "left_knee_joint": 150.0,
            "left_ankle_joint": 40.0, "right_hip_yaw_joint": 150.0,
            "right_hip_roll_joint": 150.0, "right_hip_pitch_joint": 150.0,
            "right_knee_joint": 150.0, "right_ankle_joint": 40.0,
            "torso_joint": 150.0, "left_shoulder_pitch_joint": 40.0,
            "left_shoulder_roll_joint": 40.0, "left_shoulder_yaw_joint": 18.0,
            "left_elbow_joint": 18.0, "right_shoulder_pitch_joint": 40.0,
            "right_shoulder_roll_joint": 40.0, "right_shoulder_yaw_joint": 18.0,
            "right_elbow_joint": 18.0,
            },
        "joint_limits" : {
            "left_hip_yaw_joint":[-0.2 , 0.2],
            "left_hip_roll_joint":[-0.43 , 0.43],
            "left_hip_pitch_joint":[-3.14 , 0.6 ],
            "left_knee_joint":[-0.26 , 2.05],
            "left_ankle_joint":[-0.87 , 0.52],
            "right_hip_yaw_joint":[-0.2 , 0.2],
            "right_hip_roll_joint":[-0.43 , 0.43],
            "right_hip_pitch_joint":[-3.14 , 0.6 ], 
            "right_knee_joint":[-0.26 , 2.05],
            "right_ankle_joint":[-0.87 , 0.52], 
            "torso_joint":[-0.3 , 0.3],
            "left_shoulder_pitch_joint":[-0.5 , 0.5], 
            "left_shoulder_roll_joint":[-0.34 , 0.5 ],
            "left_shoulder_yaw_joint":[-0.5 , 0.5],
            "left_elbow_joint":[-1.5, -0.5],
            "right_shoulder_pitch_joint":[-0.7 , 0.7], 
            "right_shoulder_roll_joint":[-0.5 ,  0.34],
            "right_shoulder_yaw_joint":[-0.5 , 0.5], 
            "right_elbow_joint":[-1.25, -0.5 ]
            },
        "kp": {
            "left_hip_yaw_joint": 100.0, "left_hip_roll_joint": 100.0,
            "left_hip_pitch_joint": 100.0, "left_knee_joint": 100.0,
            "left_ankle_joint": 20.0, "right_hip_yaw_joint": 100.0,
            "right_hip_roll_joint": 100.0, "right_hip_pitch_joint": 100.0,
            "right_knee_joint": 100.0, "right_ankle_joint": 20.0,
            "torso_joint": 100.0, "left_shoulder_pitch_joint": 100.0,
            "left_shoulder_roll_joint": 100.0, "left_shoulder_yaw_joint": 100.0,
            "left_elbow_joint": 100.0, "right_shoulder_pitch_joint": 100.0,
            "right_shoulder_roll_joint": 100.0, "right_shoulder_yaw_joint": 100.0,
            "right_elbow_joint": 100.0,
            },
        "kd": {
            "left_hip_yaw_joint": 10.0, "left_hip_roll_joint": 10.0,
            "left_hip_pitch_joint": 10.0, "left_knee_joint": 10.0,
            "left_ankle_joint": 4.0, "right_hip_yaw_joint": 10.0,
            "right_hip_roll_joint": 10.0, "right_hip_pitch_joint": 10.0,
            "right_knee_joint": 10.0, "right_ankle_joint": 4.0,
            "torso_joint": 10.0, "left_shoulder_pitch_joint": 5.0,
            "left_shoulder_roll_joint": 5.0, "left_shoulder_yaw_joint": 5.0,
            "left_elbow_joint": 5.0, "right_shoulder_pitch_joint": 5.0,
            "right_shoulder_roll_joint": 5.0, "right_shoulder_yaw_joint": 5.0,
            "right_elbow_joint": 5.0,
            },
        "randomize_cfg":{
            "randomize":1.0,
            "friction_ratio_shift":0.4,
            "mass_shift": 10,
            "com_shift":0.01,
            "kp_shift":0.2,
            "kd_shift": 0.2,
            "time_delay": 4 # number of timesteps of delay, uniformly chosen for each env
            },
    }
    obs_cfg = {
        "num_obs": 63,
        "obs_scales": {
            "lin_vel": 5.3,
            "ang_vel": 1.37,
            "dof_pos": 10.0,
            "dof_vel": 0.77,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 1.0,
        "feet_height_target": 0.2,
        "reward_scales": {
            "tracking_lin_vel": 0.9,
            "tracking_ang_vel": 0.0,#0.3,
            #"distance_to_target": 0.9,
            #"death": 4.0,
            #"lin_vel_z": 0.0,#-1.0,
            #"base_height": 0.0,#-50.0,
            "action_rate": -0.0005,
            #"similar_to_default": 0.0,#-0.01,
            #"feet_height": 0.0,
            "imitation": 0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.1, 0.4],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0.0, 0.0],#[-0.4, 0.4],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=8000)
    parser.add_argument("--show_viewer", action="store_true", default=False)
    parser.add_argument("-l", "--load", type=int, default=None)
    args = parser.parse_args()

    gs.init(logging_level="warning",backend=gs.gpu)

    if args.exp_name == "h1-walking":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        args.exp_name = f"{args.exp_name}_{timestamp}"

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(log_dir) and args.load is None:
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.load is not None:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    else:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    env = H1Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
        low_level_policy_model=train_cfg["low_level_policy_model"],
        low_level_policy_cfgs=train_cfg["low_level_policy_cfgs"],
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    if args.load is not None:
        resume_path = os.path.join(log_dir, f"model_{args.load}.pt")
        print(f"Loading from {resume_path}")
        runner.load(resume_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
