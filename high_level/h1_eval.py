import argparse
import os
import pickle

import torch

import torch.nn.functional as F

from rl_ase.custom_on_policy_runner import OnPolicyRunner
from rl_ase.custom_actor_critic import ActorCritic

from rl_ase.utils import _sample_latents

import genesis as gs

import numpy as np

from h1_env import H1Env

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1-walking")
    parser.add_argument("--ckpt", type=int, default=50)
    parser.add_argument("--env_real_type", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gs.init(backend=gs.gpu, logging_level="warning")

    collect_data = False
    data = []
    len_data = 30000

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    reward_cfg["reward_scales"] = {}

    env = H1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not collect_data,
        low_level_policy_model=train_cfg["low_level_policy_model"],
        low_level_policy_cfgs=train_cfg["low_level_policy_cfgs"],
        recording=False,
        cam_duration=500,
        env_real_type=args.env_real_type,
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    high_level_policy, low_level_policy = runner.get_inference_policy(device=gs.device)
    obs, _ = env.reset()
    count = 0
    z_mean = high_level_policy(obs)
    z_variables = _sample_latents(z_mean.shape[0], z_mean.shape[1], device = z_mean.device, mean=z_mean)
    with torch.no_grad():
        if collect_data:
            for i in tqdm(range(len_data)):
                z_mean = high_level_policy(obs)
                z_variables = _sample_latents(z_mean.shape[0], z_mean.shape[1], device = z_mean.device, mean=z_mean)
                
                comm = env.commands[0].clone().cpu().tolist()
                x_comm = comm[0]
                ang_comm = comm[2]
                x_vel = env.base_lin_vel[0][0].item()
                ang_vel = env.base_ang_vel[0][2].item()
                data.append([x_comm, ang_comm, x_vel, ang_vel])
                
                #print("commands:", x_comm, ang_comm)
                #print("velocities", x_vel, ang_vel)
                #print("z_mean:", z_mean)

                new_obs = torch.cat([obs, z_variables], dim=-1).to(device)
                actions = low_level_policy.act_inference(new_obs)
                obs, rews, dones, infos = env.step(actions)
        else:
            while True:
                if count % int(train_cfg["high_level_division_frequency_factor"]) == 0:
                    z_mean = high_level_policy(obs)
                    z_variables = _sample_latents(z_mean.shape[0], z_mean.shape[1], device = z_mean.device, mean=z_mean)

                comm = env.commands[0].clone().cpu().tolist()
                x_comm = comm[0]
                ang_comm = comm[2]
                print("commands:", x_comm, ang_comm)
                #print("velocities", x_vel, ang_vel)
                #print("z_mean:", z_mean)

                new_obs = torch.cat([obs, z_variables], dim=-1).to(device)
                actions = low_level_policy.act_inference(new_obs)
                obs, rews, dones, infos = env.step(actions)
                count += 1
    if collect_data:
        data = np.array(data)
        np.save("data.npy", data)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
