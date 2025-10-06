import torch
from typing import Optional, Tuple, Dict, List

def build_mirror_index_map(env) -> List[int]:
    dof_names = env.env_cfg["dof_names"]
    mirror_joints = env.env_cfg["mirror_joints"]

    joint_to_idx = {name: i for i, name in enumerate(dof_names)}
    index_map = list(range(len(dof_names)))
    for left, right in mirror_joints.items():
        if left in joint_to_idx and right in joint_to_idx:
            i, j = joint_to_idx[left], joint_to_idx[right]
            index_map[i], index_map[j] = j, i
    return index_map

def build_negation_mask(env, keys_to_negate: List[str]) -> List[bool]:
    dof_names = env.env_cfg["dof_names"]
    return [any(k in name for k in keys_to_negate) for name in dof_names]


def mirror_obs_actions(
    obs: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    env,
    obs_type: str = "policy",
    keys_to_negate: List[str] = ["roll", "yaw"],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    dof_names = env.env_cfg["dof_names"]
    keys_to_negate = [
        "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "torso_joint",
    ]

    device = obs.device if obs is not None else actions.device

    index_map = build_mirror_index_map(env)
    index_map_tensor = torch.tensor(index_map, device=device)
    negate_mask = build_negation_mask(env, keys_to_negate)
    negate_tensor = torch.tensor([-1.0 if m else 1.0 for m in negate_mask], device=device)

    mirrored_obs = None
    mirrored_actions = None

    if obs is not None:
        mirrored_obs = obs.clone()
        # Expected obs layout: [3 (ang vel), 3 (gravity), 3 (cmd), 19 (pos), 19 (vel), 19 (actions)]
        dof_pos_start = 9
        dof_vel_start = dof_pos_start + len(dof_names)
        actions_start = dof_vel_start + len(dof_names)

        # Mirror dof_pos
        mirrored_obs[:, dof_pos_start : dof_pos_start + len(dof_names)] = (
            obs[:, dof_pos_start : dof_pos_start + len(dof_names)][:, index_map_tensor] * negate_tensor
        )
        # Mirror dof_vel
        mirrored_obs[:, dof_vel_start : dof_vel_start + len(dof_names)] = (
            obs[:, dof_vel_start : dof_vel_start + len(dof_names)][:, index_map_tensor] * negate_tensor
        )
        # Mirror actions (last part of obs_buf)
        mirrored_obs[:, actions_start : actions_start + len(dof_names)] = (
            obs[:, actions_start : actions_start + len(dof_names)][:, index_map_tensor] * negate_tensor
        )

    if actions is not None:
        mirrored_actions = actions.clone()
        mirrored_actions = actions[:, index_map_tensor] * negate_tensor

    return mirrored_obs, mirrored_actions
