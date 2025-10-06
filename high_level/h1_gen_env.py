import torch
import math
import genesis as gs
import numpy as np
import cv2
import torch.nn.functional as F
from geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import datetime
import time

class H1RobotGenesis:
    def __init__(self, num_envs, env_cfg, obs_cfg, show_viewer=False, recording=False, cam_duration=800):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_envs = num_envs

        self.env_cfg = env_cfg
        self.randomize_cfg = env_cfg["randomize_cfg"]

        self.recording = recording

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]

        self.simulate_action_latency = False  # there is a 1 step latency on real robot
        self.frequency = 40 

        self.dt = 1/self.frequency

        self.recording = recording

        self.show_viewer = show_viewer

        self.it = 0

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(self.frequency),
                camera_pos=(1.0, 4.5, 3.5),
                camera_lookat=(0.0, -2.0, 0.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(),#rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        if self.recording:
            self.cam = self.scene.add_camera(
            res    = (1440, 1080),
            pos    = (1.0, 4.5, 3.5),
            lookat = (0.0, -2.0, 0.0),
            fov    = 40,
            GUI    = False
            )

        self.cam_count = 0
        self.cam_duration = cam_duration
        
        #self.plain = self.scene.add_entity(gs.morphs.Terrain(pos = (self.shift[0].item(),self.shift[1].item(),0.0),horizontal_scale=self.horizontal_scale, height_field = terrain), material = gs.materials.Rigid(coup_friction = 0.5, coup_softness=0.002))
        #self.plain = self.scene.add_entity(gs.morphs.Mesh(file="./terrain_city.stl",pos = (self.shift[0].item(),self.shift[1].item(),-0.1), fixed=True))#,material = gs.materials.Rigid(coup_friction = 0.5, coup_softness=0.002)))
        #self.plain = self.scene.add_entity(gs.morphs.Terrain(pos = (-6,-4-12,-0.05),horizontal_scale=horizontal_scale, subterrain_types = "flat_terrain"))
        self.plain = self.scene.add_entity(gs.morphs.Plane(), material = gs.materials.Rigid(coup_friction=0.4))
        #forcefield = gs.engine.force_fields.Turbulence(strength=100.0,frequency=3)
                # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="../h1/h1.xml",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )

        if self.show_viewer:
            # Added by Philippe, colored sphere
            self.sphere_pos = self.base_init_pos.cpu().numpy() + np.array([0, 0, 1])
        

        #imu

        #self.imu = self.scene.add_sensor(
        #    gs.sensors.imu.IMUOptions(
        #        entity_idx=self.robot.idx,
        #        link_idx_local=self.robot.get_link("imu_link").idx_local,
        #    )
        #)

        # build
        self.scene.build(n_envs=num_envs, env_spacing=(0.0, 0.0), n_envs_per_row=1)
        if self.recording:
            self.cam.start_recording()

        self.motor_dofs = []
        for i,name in enumerate(self.env_cfg["dof_names"]):
            #print(name)
            #name = name.replace("_joint", "")
            self.motor_dofs.append(self.get_joint_id(name))
        


        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"][name] for name in self.env_cfg["dof_names"]], self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"][name] for name in self.env_cfg["dof_names"]], self.motor_dofs)
        self.base_kp = torch.Tensor([self.env_cfg["kp"][name] for name in self.env_cfg["dof_names"]]).to(self.device)
        self.base_kd = torch.Tensor([self.env_cfg["kd"][name] for name in self.env_cfg["dof_names"]]).to(self.device)
        self.kp = torch.ones((self.num_envs, len(self.motor_dofs)), device = self.device)
        self.kd = torch.ones((self.num_envs, len(self.motor_dofs)), device = self.device)

        joint_limits_torque = np.array([env_cfg["joint_limits_torque"][name] for name in self.env_cfg["dof_names"]])
        self.joint_limits_torque = torch.Tensor(joint_limits_torque).to(self.device)
        self.robot.set_dofs_force_range(
                lower = -joint_limits_torque,
                upper = joint_limits_torque,
                dofs_idx_local = self.motor_dofs,
                )
        self.robot.set_friction(0.5)

        # domain randomization
        self.randomize_strength = self.randomize_cfg["randomize"]
        self.mass_shift =(-0.5 + torch.rand(self.num_envs,1, device = self.device))
        self.com_shift =(-0.5 + torch.rand(self.num_envs, 1,3, device = self.device))
        self.friction_ratio_shift = (-0.5+torch.rand(self.num_envs, self.robot.n_links, device = self.device))
        self.kp_shift = (-0.5+torch.rand_like(self.kp, device = self.device))
        self.kd_shift = (-0.5+torch.rand_like(self.kd, device = self.device))
        self.apply_shifts(torch.arange(self.num_envs,device = self.device))
        self.actions = torch.zeros_like(self.kp, device = self.device)

    def apply_shifts(self,envs_idx):
        self.robot.set_mass_shift(
            mass_shift=self.randomize_strength*self.randomize_cfg["mass_shift"]*self.mass_shift[envs_idx],
            ls_idx_local=[self.robot.get_link("torso_link").idx_local,],envs_idx=envs_idx,
        )
        self.robot.set_COM_shift(
            com_shift=self.randomize_strength*self.randomize_cfg["com_shift"]*self.com_shift[envs_idx],
            ls_idx_local= [self.robot.get_link("torso_link").idx_local,], envs_idx = envs_idx
        )
        self.robot.set_friction_ratio(
            friction_ratio=1+self.randomize_strength*self.randomize_cfg["friction_ratio_shift"]*self.friction_ratio_shift[envs_idx],
            ls_idx_local=np.arange(0, self.robot.n_links),envs_idx = envs_idx
        )
        self.kp[envs_idx] = self.base_kp[None,:]*(1+self.randomize_strength*self.randomize_cfg["kp_shift"]*self.kp_shift[envs_idx])
        self.kd[envs_idx] = self.base_kd[None,:]*(1+self.randomize_strength*self.randomize_cfg["kd_shift"]*self.kd_shift[envs_idx])


    def randomize_shifts(self,envs_idx):
        self.mass_shift[envs_idx] =(-0.5 + torch.rand_like(self.mass_shift[envs_idx]))
        self.friction_ratio_shift[envs_idx] =(-0.5 + torch.rand_like(self.friction_ratio_shift[envs_idx]))
        self.com_shift[envs_idx]=(-0.5 + torch.rand_like(self.com_shift[envs_idx]))
        self.kp_shift[envs_idx]=(-0.5 + torch.rand_like(self.kp_shift[envs_idx]))
        self.kd_shift[envs_idx]=(-0.5 + torch.rand_like(self.kd_shift[envs_idx]))

    def step_corrector(self):
        error = self.actions-self.get_dofs_position()
        force = error*self.kp-self.get_dofs_velocity()*self.kd
        #self.robot.control_dofs_position(actions, self.motor_dofs)
        self.robot.control_dofs_force(force, self.motor_dofs)


    def control_dofs_position(self, actions):
        self.actions = actions
        #self.robot.control_dofs_position(actions, self.motor_dofs)
        #print(self.motor_dofs)
        #print(self.env_cfg["dof_names"])
        #print("kpkd")
        #print(self.robot.get_dofs_control_force(self.motor_dofs),force, error, sep="\n")

    def get_pos(self):
        return self.robot.get_pos()

    def get_quat(self):
        return self.robot.get_quat()


    def get_imu_infos(self):
        # return acc, gyro, rpy from the imu link
        imu_link = self.robot.get_link("imu_link")
        imu_acc_wf =  self.robot.get_links_acc(ls_idx_local = [self.robot.get_link("imu_link").idx_local])[:,0,:]
        imu_acc_wf[:,2]+= 9.81 # we add gravity
        imu_angvel_wf = imu_link.get_ang()
        imu_quat = imu_link.get_quat()
        inv_imu_quat = inv_quat(imu_quat)
        imu_acc_l = transform_by_quat(imu_acc_wf, inv_imu_quat)
        imu_angvel_l = transform_by_quat(imu_angvel_wf, inv_imu_quat)

        rpy = quat_to_xyz(imu_quat, rpy=True)
        return imu_acc_l, imu_angvel_l, rpy

    def get_vel(self):
        return self.robot.get_vel()

    def get_ang(self):
        return self.robot.get_ang()

    def get_reset(self):
        return False

    def get_dofs_position(self):
        return self.robot.get_dofs_position(self.motor_dofs)

    def get_dofs_velocity(self):
        return self.robot.get_dofs_velocity(self.motor_dofs)

    def set_dofs_position(self, position, envs_idx=None):
        self.robot.set_dofs_position(
                position=position,
                dofs_idx_local=self.motor_dofs,
                zero_velocity=True,
                envs_idx=envs_idx,
                )
    def set_dofs_velocity(self,velocity, envs_idx=None):
        self.robot.set_dofs_velocity(
            velocity=velocity,
            dofs_idx_local=self.motor_dofs,
            envs_idx=envs_idx,
            )


    def set_pos(self, pose,  envs_idx=None):
        self.robot.set_pos(pose, zero_velocity=True, envs_idx=envs_idx)

    def set_vel(self,vel, envs_idx=None):
        self.robot.set_dofs_velocity(
            velocity=vel,
            dofs_idx_local=[0,1,2],
            envs_idx=envs_idx,
            )


    def set_ang(self, ang, envs_idx=None):
        self.robot.set_dofs_velocity(
            velocity=ang,
            dofs_idx_local=[3,4,5],
            envs_idx=envs_idx,
            )



    def set_pos_and_quat(self, pose, quat, envs_idx=None):
        self.set_pos(pose, envs_idx=envs_idx)
        self.set_quat(quat, envs_idx=envs_idx)

    def set_quat(self, quat, envs_idx=None):
        self.robot.set_quat(quat, zero_velocity=True, envs_idx=envs_idx)

    def reset(self, envs_idx):
        self.robot.zero_all_dofs_velocity(envs_idx)
        self.randomize_shifts(envs_idx)
        self.apply_shifts(envs_idx)

    def get_link_pos(self,name):
        link = self.robot.get_link(name)
        return link.get_pos()

    def get_link_vel(self,name):
        link = self.robot.get_link(name)
        return link.get_vel()

    def get_link_quat(self,name):
        link = self.robot.get_link(name)           
        return link.get_quat()


    def get_joint_id(self, name):
        return self.robot.get_joint(name).dof_idx_local

    """ def flying(self):
        contactswg = self.plain.get_links_net_contact_force()[:,0] # TODO
        flying_buf = torch.all(contactswg==0, dim=-1) 
        return flying_buf """

    def debug_directions(self, commands):
        if self.it%25==0 and self.show_viewer:
            commands = torch.cat([commands,torch.zeros((self.num_envs, 1), device= self.device)], axis=1).cpu().numpy() # add z axis
            self.scene.clear_debug_objects()
            for i in range(self.num_envs):
                self.scene.draw_debug_arrow(self.get_pos()[i].cpu().numpy(), vec=commands[i], radius=0.01, color=(1.0, 0.0, 0.0, 0.5))
        return


    def step(self):
        #self.scene.step()
        self.step_corrector()
        self.scene.step()
        self.it += 1
        if self.cam_count < self.cam_duration and self.recording:
            pos_cam = self.get_pos()[0].cpu().numpy()
            self.cam.set_pose(
                    pos = pos_cam + np.array([1.0, 4.5, 3.5]),
                    lookat = pos_cam,
                    )
            self.cam.render()
            self.cam_count += 1
        if self.cam_count == self.cam_duration and self.recording:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            path_video = f"videos/{current_datetime}.mp4"

            self.cam.stop_recording(save_to_filename=path_video, fps=int(50))
            print(f"Recorded at {path_video}")
            self.cam_count += 1
            