import torch
import math
import genesis as gs
import numpy as np
from geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, xyz_to_quat
import colorsys

from random import randint

import datetime

from rl_ase.utils import _sample_latents

from h1_gen_env import H1RobotGenesis
from real_env import H1RobotReal

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class H1Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, encoder_discriminator, resample_z=True, show_viewer=False, recording=False, cam_duration=500, z_timesteps_uniformely=False, collecting_data=False, env_real_type=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.recording = recording

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.latent_dim = env_cfg["latent_dim"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"] # there is a 1 step latency on real robot
        self.frequency = 40

        self.dt = 1/self.frequency

        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.encoder_discriminator = encoder_discriminator

        self.resample_z = resample_z

        self.z_timesteps_uniformely = z_timesteps_uniformely

        self.env_real_type = env_real_type

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        if self.env_real_type:
            self.robot = H1RobotReal(env_cfg, obs_cfg)
        else:
            self.robot = H1RobotGenesis(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, show_viewer=show_viewer, recording=recording, cam_duration=cam_duration)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        self.beta = env_cfg["beta"]
        self.kappa = env_cfg["kappa"]
        self.keep_z_constant = env_cfg["keep_z_constant"]
        self.current_z_constant = self._sample_z_timestep()
        self.z_count = 0
        self.z_values = self.sample_latents(self.num_envs, self.latent_dim, self.device)
        self.robot._update_sphere()

        self.state_buf = torch.zeros((self.num_envs, 6+2*self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_state_buf = torch.zeros_like(self.state_buf)

        self.encoder_out = torch.zeros((self.num_envs, self.latent_dim), device=self.device, dtype=gs.tc_float)
        self.discrim_out = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)

        self.collecting_data = collecting_data
        if self.collecting_data:
            self.velocity_save = []
        
        self.gyro = torch.zeros_like(self.base_ang_vel)
        self.rpy = torch.zeros_like(self.base_ang_vel)
        
        noise_cfg = {"noise_level":1.0,"gravity":0.05, "base_ang_vel":0.2, "dof_pos":0.01, "dof_vel":1.5, "last_actions":0.00}
        noise_std = [
            noise_cfg["gravity"]*torch.ones_like(self.projected_gravity),
            noise_cfg["base_ang_vel"]*torch.ones_like(self.base_ang_vel)*self.obs_scales["ang_vel"],
            noise_cfg["dof_pos"]*torch.ones_like(self.dof_pos)*self.obs_scales["dof_pos"],
            noise_cfg["dof_vel"]*torch.ones_like(self.dof_vel)*self.obs_scales["dof_vel"],
            noise_cfg["last_actions"]*torch.ones_like(self.actions),
            0.0*torch.ones_like(self.z_values),
        ]
        self.noise_std_vec = noise_cfg["noise_level"]*torch.cat(noise_std,axis=1)

        self.num_save_actions = 4
        self.action_array = torch.zeros((self.num_envs, self.num_save_actions, self.num_actions), device = self.device, dtype=gs.tc_float)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):

        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        self.action_array = torch.roll(self.action_array, 1, dims=1)

        self.action_array[:, 0, :] = self.actions

        if self.env_real_type:
            exec_actions = self.actions
        else:
            exec_actions = self.action_array[:,self.num_save_actions-1,:]
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos)#, self.motor_dofs)
        self.robot.step()

        self.z_count += 1

        if self.z_count > self.current_z_constant and self.resample_z:
            self.z_count = 0
            self.current_z_constant = self._sample_z_timestep()
            self.z_values = self.sample_latents(self.num_envs, self.latent_dim, self.device)
            self.robot._update_sphere()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)

        _, self.gyro[:], self.rpy[:] = self.robot.get_imu_infos()

        imu_quat = xyz_to_quat(self.rpy,rpy=True)
        inv_imu_quat = inv_quat(imu_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_imu_quat)

        #self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position()#self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity()#self.motor_dofs)

        if self.collecting_data:
            tmp_appending = torch.cat([
                self.z_values[0],
                self.base_lin_vel[0,0].unsqueeze(0),
                self.base_ang_vel[0,2].unsqueeze(0)
            ])

            self.velocity_save.append(tmp_appending.cpu().numpy())

        # print(1/torch.std(self.dof_pos), 1/torch.std(self.dof_vel), 1/torch.std(self.base_ang_vel), 1/torch.std(self.base_lin_vel))

        # print(torch.max(torch.max(torch.abs(self.dof_vel), dim=0)[0]))

        #self.feet_pos[:] = torch.stack([
        #    transform_by_quat(self.robot.get_link(name + "_link").get_pos(), inv_base_quat) for name in self.env_cfg["feet_names"]
        #], dim=-1)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # Compute states before rewards
        self.state_buf = torch.cat(
            [
                self.base_lin_vel,# * self.obs_scales["lin_vel"],
                self.gyro, #self.base_ang_vel,# * self.obs_scales["ang_vel"],
                (self.dof_pos - self.default_dof_pos),# * self.obs_scales["dof_pos"],  # 12
                self.dof_vel,# * self.obs_scales["dof_vel"],  # 12
            ],
            axis=-1,
        )

        self.discrim_out = self.encoder_discriminator(
            torch.cat([self.state_buf, self.last_state_buf], dim=-1),
            discriminator=True,
        )

        self.encoder_out = self.encoder_discriminator(
            torch.cat([self.state_buf, self.last_state_buf], dim=-1),
            discriminator=False,
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        #self.reset_buf = False
        self.term_buf = (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]) | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]) | (self.base_pos[:, 2] < self.env_cfg["termination_if_base_height_less_than"])
        
        #contactswg = self.plane.get_links_net_contact_force()[:,0]
        #self.flying_buf = torch.all(contactswg==0, dim=-1) & (self.episode_length_buf>10)
        #self.term_buf |= self.flying_buf
        
        self.reset_buf |= self.term_buf

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
                [
                self.projected_gravity,  # 3
                #self.commands * self.commands_scale,  # 3
                #self.base_lin_vel * self.obs_scales["lin_vel"],
                self.gyro * self.obs_scales["ang_vel"],  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 19
                self.z_values,  # 64
            ],
            axis=-1,
        )

        # self.extras["observations"]["last_state_buf"] = self.last_state_buf

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_state_buf[:] = self.state_buf[:]

        self.extras["observations"]["critic"] = torch.cat([self.obs_buf[:,:3], self.base_lin_vel * self.obs_scales["lin_vel"], self.obs_buf[:,3:]], dim=-1)

        if not self.env_real_type:
            self.obs_buf += self.noise_std_vec*2*(-0.5+torch.rand_like(self.obs_buf))

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_state(self):
        return self.state_buf

    def get_observations(self):
        self.extras["observations"]["critic"] = torch.cat([self.obs_buf[:,:3], self.base_lin_vel * self.obs_scales["lin_vel"], self.obs_buf[:,3:]], dim=-1)
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.extras["observations"]["critic"]

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.robot.reset(envs_idx)
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            #dofs_idx_local=self.motor_dofs,
            #zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx],
                           #zero_velocity=False,
                           envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx],
                            #zero_velocity=False,
                            envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        #self.robot.zero_all_dofs_velocity(envs_idx)

        self.gyro[envs_idx] = 0
        self.rpy[envs_idx] = 0

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.action_array[envs_idx] = torch.zeros((len(envs_idx), self.num_save_actions, self.num_actions), device = self.device, dtype=gs.tc_float)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.current_z_constant = self._sample_z_timestep()
        self.z_values[envs_idx] = self.sample_latents(self.num_envs, self.latent_dim, self.device)[envs_idx]
        self.robot._update_sphere()
        
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_imitation(self):

        dot_product = torch.sum(self.encoder_out * self.z_values, dim=1)

        discrim_out = self.discrim_out.squeeze(-1)

        r = -torch.log(1-discrim_out + 1e-7) + (self.beta*self.kappa*dot_product)
        return r
    
    """ def _reward_shoulder_elbow(self):
        return torch.sum(torch.square(self.last_actions[:,-8:] - self.actions[:,-8:]), dim=1) """

    #def _reward_tracking_lin_vel(self):
    #    # Tracking of linear velocity commands (xy axes)
    #    lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #    return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    #def _reward_tracking_ang_vel(self):
    #    # Tracking of angular velocity commands (yaw)
    #    ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #    return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    #def _reward_lin_vel_z(self):
    #    # Penalize z axis base linear velocity
    #    return torch.square(self.base_lin_vel[:, 2])

    """ def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1) """

    #def _reward_similar_to_default(self):
    #    # Penalize joint poses far away from default pose
    #    return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    #def _reward_base_height(self):
    #    # Penalize base height away from target
    #    return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    #def _reward_feet_height(self):
    #    feet_height = torch.mean(self.feet_pos[:, 2, :], dim=1)
    #    return torch.square(feet_height - self.reward_cfg["feet_height_target"])
    
    #def _reward_contacts(self):
    #    contactswg = self.plane.get_links_net_contact_force()[:,0]
    #    no_contact_bool = torch.all(contactswg==0, dim=-1) & (self.episode_length_buf>10)
    #    return no_contact_bool.float()

    #def _reward_death(self):
    #    return -1.0*self.term_buf

    def _sample_z_timestep(self):
        if self.z_timesteps_uniformely:
            return randint(1, self.keep_z_constant)
        else:
            return self.keep_z_constant
    
    def sample_latents(self, num_envs, latent_dim, device):
        z_random = _sample_latents(num_envs, latent_dim, device)
        return z_random
