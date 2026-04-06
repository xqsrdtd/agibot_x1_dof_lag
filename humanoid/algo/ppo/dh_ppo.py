# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor_critic_dh import ActorCriticDH
from .rollout_storage import RolloutStorage

class DHPPO:
    actor_critic: ActorCriticDH
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 lin_vel_idx = 45,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 # Diffusion prior regularization (weak RL transition)
                 use_diffusion_reg: bool = False,
                 diffusion_reg_coef: float = 0.0,
                 diffusion_adv_weighted: bool = False,
                 diffusion_adv_weight_mode: str = "positive",
                 diffusion_prior_ckpt_path: str = "",
                 # DPPO-style: finetune diffusion UNet online (still driven by denoise loss * diffusion_reg_coef)
                 diffusion_finetune: bool = False,
                 diffusion_finetune_lr: float = 1e-6,
                 diffusion_finetune_max_grad_norm: float = 0.5,
                 # DPPO-style: pull Gaussian policy mean toward diffusion-sampled action (no grad through sampling).
                 diffusion_actor_align_coef: float = 0.0,
                 diffusion_align_inference_steps: int = 15,
                 # predict_action is expensive (DDPM loop); run align only every N minibatches (1 = every).
                 diffusion_align_every: int = 1,
                 # Mix diffusion-sampled actions into env rollout (per-env Bernoulli). Log-prob still uses Gaussian π(a|s).
                 diffusion_rollout_prob: float = 0.0,
                 diffusion_rollout_inference_steps: int = 8,
                 # Mixed behavior: scale PPO surrogate & entropy on diffusion-rollout rows (Gaussian ratio is biased there).
                 diffusion_rollout_ppo_surrogate_scale: float = 1.0,
                 # When finetuning diffusion: adv-weighted denoise on executed actions from diffusion rollout only.
                 diffusion_rollout_denoise_coef: float = 0.0,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic: ActorCriticDH = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(),
                                    lr=learning_rate)
        self.state_estimator_optimizer = optim.Adam(self.actor_critic.state_estimator.parameters(),
                                        lr=learning_rate)
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
        self.num_short_obs = self.actor_critic.num_short_obs
        self.lin_vel_idx = lin_vel_idx

        # Diffusion prior regularization setup
        self.use_diffusion_reg = bool(use_diffusion_reg)
        self.diffusion_reg_coef = float(diffusion_reg_coef)
        self.diffusion_adv_weighted = bool(diffusion_adv_weighted)
        self.diffusion_adv_weight_mode = str(diffusion_adv_weight_mode)
        self.diffusion_prior_ckpt_path = str(diffusion_prior_ckpt_path) if diffusion_prior_ckpt_path is not None else ""
        self.diffusion_finetune = bool(diffusion_finetune)
        self.diffusion_finetune_lr = float(diffusion_finetune_lr)
        self.diffusion_finetune_max_grad_norm = float(diffusion_finetune_max_grad_norm)
        self.diffusion_actor_align_coef = float(diffusion_actor_align_coef)
        self.diffusion_align_inference_steps = max(1, int(diffusion_align_inference_steps))
        self.diffusion_align_every = max(1, int(diffusion_align_every))
        self.diffusion_rollout_prob = float(diffusion_rollout_prob)
        self.diffusion_rollout_inference_steps = max(1, int(diffusion_rollout_inference_steps))
        self.diffusion_rollout_ppo_surrogate_scale = float(diffusion_rollout_ppo_surrogate_scale)
        self.diffusion_rollout_denoise_coef = float(diffusion_rollout_denoise_coef)
        self.diffusion_prior = None
        self.diffusion_optimizer = None
        if self.use_diffusion_reg and self.diffusion_prior_ckpt_path and (
            self.diffusion_reg_coef != 0.0
            or self.diffusion_actor_align_coef != 0.0
            or self.diffusion_rollout_prob > 0.0
            or self.diffusion_rollout_denoise_coef != 0.0
        ):
            self.diffusion_prior = self._load_diffusion_prior_h1(
                self.diffusion_prior_ckpt_path, finetune=self.diffusion_finetune
            )
            if self.diffusion_finetune and self.diffusion_prior is not None:
                self.diffusion_optimizer = optim.Adam(
                    self.diffusion_prior.parameters(), lr=self.diffusion_finetune_lr
                )

    def _load_diffusion_prior_h1(self, ckpt_path: str, finetune: bool = False):
        """
        Load a horizon=1 no-lagtok diffusion prior trained by diffusion_policy-main.
        This method is intentionally specialized to match the Phase-0 checkpoint format.
        """
        import dill
        import sys
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        # Ensure `diffusion_policy` package is importable even when running from
        # `agibot_x1_train-18DOFs/` (not from `diffusion_policy-main/`).
        diffusion_policy_root = "/home/wuyou/humanoid_rl/diffusion_policy-main"
        if diffusion_policy_root not in sys.path:
            sys.path.insert(0, diffusion_policy_root)

        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy

        # Phase-0 constants (x1 short obs, no-lagtok)
        horizon = 1
        obs_dim = 325
        action_dim = 18
        n_obs_steps = 1
        n_action_steps = 1

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )

        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_dim * n_obs_steps,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        diffusion_prior = DiffusionUnetLowdimPolicy(
            model=model,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            obs_as_local_cond=False,
            obs_as_global_cond=True,
            pred_action_steps_only=False,
            oa_step_convention=True,
        )

        payload = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
        state_dicts = payload.get("state_dicts", {})
        if "ema_model" in state_dicts:
            state_dict = state_dicts["ema_model"]
        elif "model" in state_dicts:
            state_dict = state_dicts["model"]
        else:
            raise KeyError(f"Checkpoint missing diffusion policy weights: keys={list(state_dicts.keys())}")

        diffusion_prior.load_state_dict(state_dict, strict=False)
        diffusion_prior.to(self.device)
        if finetune:
            diffusion_prior.train()
            for p in diffusion_prior.parameters():
                p.requires_grad_(True)
        else:
            diffusion_prior.eval()
            for p in diffusion_prior.parameters():
                p.requires_grad_(False)
        return diffusion_prior

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, None, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Build Gaussian policy; optionally replace some envs' actions with diffusion samples.
        # Stored log π_G(a|s) is used in PPO ratio; diffusion rows get optional surrogate down-weighting in update().
        self.actor_critic.prepare_action_distribution(obs)
        actions = self.actor_critic.distribution.sample()
        rollout_fd = torch.zeros(obs.shape[0], 1, device=obs.device, dtype=torch.float32)
        if self.diffusion_prior is not None and self.diffusion_rollout_prob > 0.0:
            obs_seq = obs[:, -self.num_short_obs :].unsqueeze(1)
            was_training = self.diffusion_prior.training
            self.diffusion_prior.eval()
            saved_steps = self.diffusion_prior.num_inference_steps
            self.diffusion_prior.num_inference_steps = self.diffusion_rollout_inference_steps
            try:
                with torch.no_grad():
                    out = self.diffusion_prior.predict_action({"obs": obs_seq})
            finally:
                self.diffusion_prior.num_inference_steps = saved_steps
            if was_training:
                self.diffusion_prior.train()
            a_diff = out["action"]
            if a_diff.dim() == 3:
                a_diff = a_diff.squeeze(1)
            a_diff = a_diff.to(dtype=actions.dtype)
            use_diff = torch.rand(obs.shape[0], device=obs.device, dtype=torch.float32) < self.diffusion_rollout_prob
            mask = use_diff.unsqueeze(-1).expand_as(actions)
            actions = torch.where(mask, a_diff, actions)
            rollout_fd = use_diff.float().unsqueeze(-1)
        self.transition.rollout_from_diffusion = rollout_fd
        self.transition.actions = actions.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_state_estimator_loss = 0
        mean_diffusion_loss = 0
        mean_diffusion_align_loss = 0.0
        align_loss_sum = 0.0
        align_loss_count = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        mb_idx = 0
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, rollout_fd_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                state_estimator_input = obs_batch[:,-self.num_short_obs:]
                est_lin_vel = self.actor_critic.state_estimator(state_estimator_input)
                ref_lin_vel = critic_obs_batch[:,self.lin_vel_idx:self.lin_vel_idx+3].clone()
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss (per-sample reweight on diffusion rollout rows: Gaussian log-ratio is not the true mix policy).
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                adv_flat = torch.squeeze(advantages_batch)
                ppo_w = 1.0 - rollout_fd_batch.squeeze(-1) * (1.0 - self.diffusion_rollout_ppo_surrogate_scale)
                surrogate = -adv_flat * ratio * ppo_w
                surrogate_clipped = -adv_flat * torch.clamp(ratio, 1.0 - self.clip_param,
                                                             1.0 + self.clip_param) * ppo_w
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # update all actor_critic.parameters()
                ent_w = (entropy_batch * ppo_w).mean()
                loss = (surrogate_loss + 
                        self.value_loss_coef * value_loss - 
                        self.entropy_coef * ent_w +
                        torch.nn.MSELoss()(est_lin_vel, ref_lin_vel))

                diffusion_loss_value = 0.0
                diffusion_loss = None
                diffusion_align_value = 0.0

                obs_seq = None
                if self.diffusion_prior is not None and (
                    self.diffusion_reg_coef != 0.0
                    or self.diffusion_actor_align_coef > 0.0
                    or self.diffusion_rollout_denoise_coef != 0.0
                ):
                    obs_short_batch = obs_batch[:, -self.num_short_obs:]
                    obs_seq = obs_short_batch.unsqueeze(1)  # (B, 1, 325)

                if self.diffusion_prior is not None and self.diffusion_reg_coef != 0.0:
                    if self.diffusion_finetune:
                        self.diffusion_prior.train()
                    else:
                        self.diffusion_prior.eval()
                    action_seq = mu_batch.unsqueeze(1)       # (B, 1, 18)
                    if self.diffusion_adv_weighted:
                        # DPPO-v0 style: weight denoise loss by (normalized) PPO advantage.
                        per_sample_loss = self.diffusion_prior.compute_loss(
                            {"obs": obs_seq, "action": action_seq},
                            return_per_sample=True
                        )
                        adv = torch.squeeze(advantages_batch, dim=-1).detach()
                        if self.diffusion_adv_weight_mode == "positive":
                            weights = torch.clamp(adv, min=0.0)
                            weights = weights / (weights.mean() + 1e-8)
                        elif self.diffusion_adv_weight_mode == "abs":
                            weights = torch.abs(adv)
                            weights = weights / (weights.mean() + 1e-8)
                        else:
                            # signed-normalized keeps direction but may reduce stability.
                            weights = adv / (torch.mean(torch.abs(adv)) + 1e-8)
                        diffusion_loss = torch.mean(per_sample_loss * weights)
                    else:
                        diffusion_loss = self.diffusion_prior.compute_loss(
                            {"obs": obs_seq, "action": action_seq}
                        )
                    loss = loss + self.diffusion_reg_coef * diffusion_loss
                    diffusion_loss_value = float(diffusion_loss.detach().cpu().item())

                # Distill actor mean toward diffusion sample (gradients only on actor; sampling path detached).
                do_align = (
                    self.diffusion_prior is not None
                    and self.diffusion_actor_align_coef > 0.0
                    and obs_seq is not None
                    and (mb_idx % self.diffusion_align_every == 0)
                )
                if do_align:
                    was_training = self.diffusion_prior.training
                    self.diffusion_prior.eval()
                    saved_steps = self.diffusion_prior.num_inference_steps
                    self.diffusion_prior.num_inference_steps = self.diffusion_align_inference_steps
                    try:
                        with torch.no_grad():
                            out = self.diffusion_prior.predict_action({"obs": obs_seq})
                    finally:
                        self.diffusion_prior.num_inference_steps = saved_steps
                    if was_training:
                        self.diffusion_prior.train()
                    a_diff = out["action"]
                    if a_diff.dim() == 3:
                        a_diff = a_diff.squeeze(1)
                    a_diff = a_diff.detach()
                    align_loss = F.mse_loss(mu_batch, a_diff)
                    loss = loss + self.diffusion_actor_align_coef * align_loss
                    diffusion_align_value = float(align_loss.detach().cpu().item())
                    align_loss_sum += diffusion_align_value
                    align_loss_count += 1

                # Adv-weighted denoise on diffusion-rollout transitions (targets executed a); gradients to UNet when finetuning.
                if (
                    self.diffusion_rollout_denoise_coef != 0.0
                    and self.diffusion_finetune
                    and obs_seq is not None
                    and rollout_fd_batch.sum().item() > 0.0
                ):
                    was_tr_dn = self.diffusion_prior.training
                    self.diffusion_prior.train()
                    action_exec = actions_batch.unsqueeze(1).detach()
                    per_r = self.diffusion_prior.compute_loss(
                        {"obs": obs_seq, "action": action_exec},
                        return_per_sample=True,
                    )
                    if per_r.dim() > 1:
                        per_r = per_r.view(per_r.shape[0], -1).mean(dim=-1)
                    m = rollout_fd_batch.squeeze(-1)
                    if self.diffusion_adv_weighted:
                        adv = torch.squeeze(advantages_batch, dim=-1).detach()
                        if self.diffusion_adv_weight_mode == "positive":
                            w_adv = torch.clamp(adv, min=0.0)
                            w_adv = w_adv / (w_adv.mean() + 1e-8)
                        elif self.diffusion_adv_weight_mode == "abs":
                            w_adv = torch.abs(adv)
                            w_adv = w_adv / (w_adv.mean() + 1e-8)
                        else:
                            w_adv = adv / (torch.mean(torch.abs(adv)) + 1e-8)
                        w_eff = m * w_adv
                    else:
                        w_eff = m
                    denom = w_eff.sum() + 1e-8
                    rollout_dn = (per_r * w_eff).sum() / denom
                    loss = loss + self.diffusion_rollout_denoise_coef * rollout_dn
                    if not was_tr_dn:
                        self.diffusion_prior.eval()

                # Gradient step
                self.optimizer.zero_grad()
                if self.diffusion_optimizer is not None:
                    self.diffusion_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                if self.diffusion_optimizer is not None:
                    nn.utils.clip_grad_norm_(
                        self.diffusion_prior.parameters(), self.diffusion_finetune_max_grad_norm
                    )
                    self.diffusion_optimizer.step()
                self.optimizer.step()

                state_estimator_loss = torch.nn.MSELoss()(est_lin_vel, ref_lin_vel)

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_state_estimator_loss += state_estimator_loss.item()
                mean_diffusion_loss += diffusion_loss_value
                mb_idx += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_state_estimator_loss /= num_updates
        mean_diffusion_loss /= num_updates
        # Report mean align loss only over minibatches where predict_action ran (else TB looks near-zero).
        mean_diffusion_align_loss = (
            align_loss_sum / max(1, align_loss_count) if align_loss_count > 0 else 0.0
        )
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_state_estimator_loss,
            mean_diffusion_loss,
            mean_diffusion_align_loss,
        )
