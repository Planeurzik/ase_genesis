import argparse
import os
import pickle
from importlib import metadata

import torch

from rl_ase.custom_on_policy_runner import OnPolicyRunner

from rl_ase.encoder_discriminator import EncoderDiscriminator

import genesis as gs

from h1_env import H1Env

from tqdm import tqdm
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1-walking")
    parser.add_argument("--ckpt", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gs.init(backend=gs.gpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(os.path.join(log_dir, "cfgs.pkl"), "rb"))
    reward_cfg["reward_scales"] = {}

    encoder_discriminator = EncoderDiscriminator(
        input_dim=12+4*env_cfg["num_actions"],
        hidden_units=train_cfg["encoder_discriminator_hidden_units"],
        encoder_output_dim=env_cfg["latent_dim"],
    ).to(device)

    env = H1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        encoder_discriminator=encoder_discriminator,
        resample_z=True,
        show_viewer=False,
        recording=False,
        cam_duration=500,
        z_timesteps_uniformely=False,
        collecting_data=True
    )

    """
    total_mass = env.robot.get_mass()
    print(total_mass)
    missing_mass = 58 - total_mass 
    link_masses = [(l, l.get_mass()) for l in env.robot.links]

    print("Updated link masses:")
    new_total = 0.0
    for link, original_mass in link_masses:
        ratio = original_mass / total_mass if total_mass > 0 else 0
        new_mass = original_mass + missing_mass * ratio
        new_total += new_mass
        print(f"Link: {link.name:30s} Old: {original_mass:.4f}  New: {new_mass:.4f}")

    print(f"\n✅ Adjusted Total Mass: {new_total:.4f} kg (Target: 58 kg)")
    exit(0)
    """
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    nb_data = 300000

    with torch.no_grad():
        for _ in tqdm(range(nb_data)):
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
        array_to_save = env.velocity_save
        array_to_save = np.array(array_to_save)
        np.save("real_data_vel_angle.npy", array_to_save)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
