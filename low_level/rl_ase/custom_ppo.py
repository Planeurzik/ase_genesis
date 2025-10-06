# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain

import numpy as np

from rl_ase.custom_actor_critic import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rl_ase.custom_rollout_storage import RolloutStorage
from rsl_rl.utils import string_to_callable

from rl_ase.utils import _sample_latents


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        encoder_discriminator = None,
        latent_dim = 64,
        num_actions = 19,
        enc_disc_nb_updates = 10,
        enc_disc_batch_size = 4096,
        kappa = 1.0,
        w_div = 0.01,
        w_gp = 5.0,
        dataset = "recordings/clean_dataset_small.npy",
        scales = None,
    ):
        # device-related parameters
        self.device = device

        # Added by Philippe
        self.encoder_discriminator = encoder_discriminator
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.enc_disc_nb_updates = enc_disc_nb_updates
        self.enc_disc_batch_size = enc_disc_batch_size
        self.w_div = w_div
        self.scales = scales
        
        # Loading the dataset here Philippe
        print("Dataset file:", dataset)
        self.dataset = np.load(dataset)
        self.dataset = torch.from_numpy(self.dataset).to(self.device)
        
        self.w_gp = w_gp

        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_encoder_discriminator = optim.Adam(self.encoder_discriminator.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Added by Philippe
        self.kappa = kappa

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Added by Philippe
        # self.transition.last_state_buf  = infos["observations"]["last_state_buf"].to(self.device).clone()

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_diversity = 0
        mean_encoder_loss = 0
        mean_discriminator_loss = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Added by Philippe

            encoder_loss = self._update_encoder(critic_obs_batch)

            discriminator_loss = self._update_discriminator(critic_obs_batch)

            #diversity_loss = self._compute_diversity_loss(obs_batch, mu_batch)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() #+ self.w_div * diversity_loss

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                #symmetry_loss = mse_loss(
                #    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                #)
                symmetry_loss = mse_loss(
                    mean_actions_batch, actions_mean_symm_batch.detach()
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            # encoder_loss.backward()
            # discriminator_loss.backward()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            #mean_diversity += diversity_loss.item()
            mean_encoder_loss += encoder_loss.item()
            mean_discriminator_loss += discriminator_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        #mean_diversity /= num_updates
        mean_encoder_loss /= num_updates
        mean_discriminator_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            #"diversity": mean_diversity,
            "encoder": mean_encoder_loss,
            "discriminator": mean_discriminator_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

    def _compute_diversity_loss(self, obs, mu):
        """Compute the diversity loss as described in equation (15).
        
        This encourages the policy to produce different behaviors for different latents
        by penalizing when policies conditioned on different latents behave similarly.
        
        Args:
            obs: Observations tensor [batch_size, obs_dim]
            ase_latents: Current latent variables z [batch_size, latent_dim]
            
        Returns:
            diversity_loss: The computed diversity loss term
        """
        # Uncomment to disable
        return torch.tensor(0.0, device=obs.device)
        n = obs.shape[0]
        obs = obs[:, :obs.shape[1] - self.latent_dim]
        z_original = obs[:, obs.shape[1] - self.latent_dim :]
        
        # Sample one different set of latent from p(z)
        z_random = _sample_latents(n, self.latent_dim, obs.device)
        
        # Get the actor outputs for both sets of latents
        # First, run the policy with z1 latents

        obs2 = torch.cat((obs, z_random), dim=-1)
        
        # Then, run the policy with z2 latents
        mu2 = self.policy.act_inference(obs2)

        clipped_mu1 = torch.clamp(mu, -1.0, 1.0)
        clipped_mu2 = torch.clamp(mu2, -1.0, 1.0)

        kl_div = clipped_mu1 - clipped_mu2

        a_diff = torch.mean(torch.square(kl_div), dim=-1)

        # Compute distance between latents: D_z(z₁, z₂)
        z_dist = torch.sum(z_original * z_random, dim=-1)
        z_dist = 0.5*(1 - z_dist)

        z_dist = torch.clamp(z_dist, min=0.1)
        # Compute the ratio and apply the formula from equation (15)
        ratio = a_diff / (z_dist)
        ratio = torch.clamp(ratio, max=10.0)
        diversity_loss = torch.mean(torch.square((ratio - 1.0)),dim=-1)
        #log_ratio = torch.log(a_diff + 1e-5) - torch.log(z_dist + 1e-5)
        #diversity_loss = torch.mean(log_ratio, dim=-1)
        
        return diversity_loss
    
    def _update_encoder(self, critic_obs):

        total_loss = torch.tensor(0.0, device=self.device)
        input_size = critic_obs.shape[0]-1
        batch_size = min(self.enc_disc_batch_size, input_size)

        temp = critic_obs[:, 3:critic_obs.shape[1] - self.latent_dim - self.num_actions]
        states = torch.zeros_like(temp)
        states[:, :3] = temp[:, :3]/self.scales["lin_vel"]
        states[:, 3:6] = temp[:, 3:6]/self.scales["ang_vel"]
        states[:, 6:6+self.num_actions] = temp[:, 6:6+self.num_actions]/self.scales["dof_pos"]
        states[:, 6+self.num_actions:6+2*self.num_actions] = temp[:, 6+self.num_actions:6+2*self.num_actions]/self.scales["dof_vel"]
        last_states = states[:-1]
        current_states = states[1:]

        z = critic_obs[1:, critic_obs.shape[1] - self.latent_dim:]

        if states.shape[1] != 6+2*self.num_actions:
            print(f"State shape is not correct {states.shape}. Exiting...")
            exit(0)

        for n in range(self.enc_disc_nb_updates):

            indices = torch.randint(0, input_size, (batch_size,), device=self.device)

            last_states_batch = last_states[indices]
            current_states_batch = current_states[indices]

            z_batch = z[indices]
            input = torch.cat((current_states_batch, last_states_batch), dim=-1)
            encoder_out = self.encoder_discriminator(input, discriminator=False)

            loss = -self.kappa * torch.sum(z_batch*encoder_out, dim=-1)
            loss = torch.mean(loss)

            self.optimizer_encoder_discriminator.zero_grad()
            loss.backward()
            self.optimizer_encoder_discriminator.step()

            total_loss += loss / self.enc_disc_nb_updates
        
            
        
        return total_loss
    
    def _update_discriminator(self, critic_obs):
        temp = critic_obs[:, 3:critic_obs.shape[1] - self.latent_dim - self.num_actions]
        states = torch.zeros_like(temp)
        states[:, :3] = temp[:, :3]/self.scales["lin_vel"]
        states[:, 3:6] = temp[:, 3:6]/self.scales["ang_vel"]
        states[:, 6:6+self.num_actions] = temp[:, 6:6+self.num_actions]/self.scales["dof_pos"]
        states[:, 6+self.num_actions:6+2*self.num_actions] = temp[:, 6+self.num_actions:6+2*self.num_actions]/self.scales["dof_vel"]
        last_states = states[:-1]
        current_states = states[1:]

        if states.shape[1] != 6+2*self.num_actions:
            print(f"State shape is not correct {states.shape}. Exiting...")
            exit(0)

        last_states_dataset = self.dataset[:-1]
        current_states_dataset = self.dataset[1:]

        batch_size = min(self.enc_disc_batch_size, last_states.shape[0], last_states_dataset.shape[0])

        total_loss = torch.tensor(0.0, device=self.device)

        # Added to remove transitions between skills !!!!
        all_indices = torch.arange(last_states_dataset.shape[0], device=self.device)
        valid_indices = all_indices#all_indices[all_indices % 180 != 179]

        for n in range(self.enc_disc_nb_updates):

            indices_obs = torch.randint(0, last_states.shape[0], (batch_size,), device=self.device)
            indices_dataset = torch.randint(0, valid_indices.shape[0], (batch_size,), device=self.device)

            indices_dataset = valid_indices[indices_dataset]

            last_states_batch = last_states[indices_obs]
            current_states_batch = current_states[indices_obs]
            last_states_dataset_batch = last_states_dataset[indices_dataset]
            current_states_dataset_batch = current_states_dataset[indices_dataset]

            #print("Policy", torch.mean(current_states_batch[:, self.num_actions:], dim=0))
            #print("Dataset", torch.mean(current_states_dataset_batch[:, self.num_actions:], dim=0))
            #if n==9:
            #    exit(0)

            policy_s_s_prime = torch.cat([current_states_batch, last_states_batch], dim=-1)  # (batch, 2 * state_dim)
            expert_s_s_prime = torch.cat([current_states_dataset_batch, last_states_dataset_batch], dim=-1)  # (batch, 2 * state_dim)

            # ----- Discriminator -----
            policy_preds = self.encoder_discriminator(policy_s_s_prime, discriminator=True)
            expert_preds = self.encoder_discriminator(expert_s_s_prime, discriminator=True)

            # ----- Losses -----
            loss_policy = F.binary_cross_entropy(policy_preds, torch.zeros_like(policy_preds))
            loss_expert = F.binary_cross_entropy(expert_preds, torch.ones_like(expert_preds))
            loss_gan = loss_policy + loss_expert

            # ----- Gradient penalty -----
            alpha = torch.rand(policy_s_s_prime.size(0), 1, device=self.device)
            interpolated = alpha * expert_s_s_prime + (1 - alpha) * policy_s_s_prime
            interpolated.requires_grad_(True)

            interpolated_preds = self.encoder_discriminator(interpolated, discriminator=True)

            gradients = torch.autograd.grad(
                outputs=interpolated_preds,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interpolated_preds),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss_gp = self.w_gp * gradient_penalty

            loss = loss_gan + loss_gp

            self.optimizer_encoder_discriminator.zero_grad()
            loss.backward()
            self.optimizer_encoder_discriminator.step()
            
            total_loss += loss/self.enc_disc_nb_updates
            
        return total_loss