import torch
import genesis as gs
import numpy as np
import torch.nn.functional as F
from geom import quat_to_xyz, xyz_to_quat, transform_by_quat, inv_quat, transform_quat_by_quat

import datetime
import time
from realrobot.interface import RealRobotInterface

class H1RobotReal:
    def __init__(self, env_cfg, obs_cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_envs = 1

        self.fps = 40
        self.dt = 1/self.fps  # control frequency on real robot is 50hz
        self.it = 0 # number of steps elapsed
        self.last_time = time.time()

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # names to indices
        self.motor_dofs = [self.get_joint_id(name) for name in self.env_cfg["dof_names"]]


        joint_limits_torque = np.array([env_cfg["joint_limits_torque"][name] for name in self.env_cfg["dof_names"]])
        self.joint_limits_torque = torch.Tensor(joint_limits_torque).to(self.device)

        state_path = "../state"
        action_path = "../action"
        print("initializing interface")
        self.interface = RealRobotInterface(state_path, action_path)
        print("initialized interface")
        self.rpy = torch.zeros((1,3), device=self.device)
        self.gyro = torch.zeros((1,3), device=self.device)
        self.acc = torch.zeros((1,3), device=self.device)
        self.mot_pos = torch.zeros((1,19), device=self.device)
        self.mot_vel = torch.zeros((1,19), device=self.device)
        self.pos = torch.zeros((1,3), device=self.device) # mock pose is at 0,0 and the robot dont move
        self.pos[0,0]=10
        self.pos[0,2]=0.95
        self.vel= torch.zeros((1,3), device=self.device) # mock pose is at 0,0 and the robot dont move
        self.ang = torch.zeros((1,3),device = self.device) # mock pose is at 0,0 and the robot dont move
        self.action = torch.zeros((1,19), device=self.device)
        self.command= torch.zeros((1,2), device=self.device)
        self.done= False

    def control_dofs_position(self, actions):
        self.action[:] = actions
        #print("mot pos",self.mot_pos.pow(2).sum())

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return xyz_to_quat(self.rpy, rpy=True)

    def get_vel(self):
        return self.vel

    def get_ang(self):
        return self.ang

    def get_dofs_position(self):
        return self.mot_pos

    def get_dofs_velocity(self):
        return self.mot_vel

    def get_reset(self):
        return self.reset

    def set_dofs_position(self, position, envs_idx=None):
        return

    def set_dofs_velocity(self,velocity, envs_idx=None):
        return

    def set_pos(self, pose,  envs_idx=None):
        return

    def set_vel(self,vel, envs_idx=None):
        return

    def set_ang(self, ang, envs_idx=None):
        return

    def set_pos_and_quat(self, pose, quat, envs_idx=None):
        return

    def set_quat(self, quat, envs_idx=None):
        return

    def reset(self, envs_idx):
        self.done= True

    def get_link_pos(self,name):
        raise NotImplementedError()

    def get_link_vel(self,name):
        raise NotImplementedError()

    def get_link_quat(self,name):
        raise NotImplementedError()


    def get_imu_infos(self):
        # return acc, gyro and rpy from the imu link
        return self.acc, self.gyro, self.rpy


    def get_joint_id(self, name):
        return self.env_cfg["dof_names"].index(name)

    def flying(self):
        return False

    def debug_directions(self, commands):
        if self.it%50==0:
            print(commands)
        return

    def get_command(self):
        return self.command

    def update_state(self):
        state_dict = self.interface.get_last_state_dict()
        self.rpy[0,:] = torch.Tensor(state_dict["rpy"]).to(device=self.device)
        self.gyro[0,:] = torch.Tensor(state_dict["gyro"]).to(device=self.device)
        self.acc[0,:] = torch.Tensor(state_dict["acc"]).to(device=self.device)
        self.mot_pos[0,:] = torch.Tensor(state_dict["mot_pos"]).to(device=self.device)[:19]
        self.mot_vel[0,:] = torch.Tensor(state_dict["mot_vel"]).to(device=self.device)[:19]
        self.done = state_dict["reset"]
        self.command[0,:] = torch.Tensor(state_dict["command"]).to(device=self.device)

    def step(self):
        self.interface.act(self.action)
        difftime = self.dt-(time.time()-self.last_time)
        if difftime < 0:
            print(" negative difftime !!! " )
        time.sleep(max(0,difftime))
        self.last_time = time.time()
        self.it += 1
        self.update_state()
    

    def _update_sphere(self):
        return