import torch
import copy
import torch.nn.functional as F
from collections import defaultdict
from fmif import model_utils as mu
import numpy as np


def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * mu.MASK_TOKEN_INDEX


def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _sample_categorical_gradient(categorical_probs, temp = 1.0):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    output = torch.nn.functional.softmax((torch.log(categorical_probs)-torch.log(gumbel_norm))/temp, 2)
    return output

def calculate_gae(rewards, values, next_values, gamma=0.99, lam=0.95):
    """
    Calculate Generalized Advantage Estimation (GAE).

    Args:
        rewards (np.ndarray): Array of rewards.
        values (np.ndarray): Array of value estimates.
        next_values (np.ndarray): Array of value estimates for the next state.
        dones (np.ndarray): Array of done flags (1 if done, 0 otherwise).
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.

    Returns:
        np.ndarray: Array of advantage estimates.
    """
    advantages = torch.zeros_like(values)
    gae = 0

    for t in reversed(range(len(rewards))):
        nd = 0.0 if t == len(rewards) - 1 else 1.0
        delta = rewards[t] + gamma * next_values[t] *nd - values[t]
        gae = delta + gamma * lam * nd * gae
        advantages[t] = gae
    return advantages




class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self.num_tokens = 22
        self.neg_infinity = -1000000.0

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_aatypes(self, aatypes_1, t, res_mask): #, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)

        if self._cfg.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t) # (B, N) # t=1 is clean data

            aatypes_t[corruption_mask] = mu.MASK_TOKEN_INDEX

            aatypes_t = aatypes_t * res_mask + mu.MASK_TOKEN_INDEX * (1 - res_mask)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._cfg.interpolant_type}")

        return aatypes_t.long()

    def corrupt_batch(self, batch, t=None):
        noisy_batch = copy.deepcopy(batch)
        X, S, mask, chain_M, residue_idx, chain_encoding_all = noisy_batch
        noisy_batch = {}
        noisy_batch['X'] = X
        noisy_batch['S'] = S
        noisy_batch['mask'] = mask
        noisy_batch['chain_M'] = chain_M
        noisy_batch['residue_idx'] = residue_idx
        noisy_batch['chain_encoding_all'] = chain_encoding_all
        aatypes_1 = S
        num_batch, num_res = aatypes_1.shape
        
        if t is None:
            t = self.sample_t(num_batch)[:, None]
        else:
            t = torch.ones((num_batch, 1), device=self._device) * t
        noisy_batch['t'] = t
        res_mask = mask * chain_M
        aatypes_t = self._corrupt_aatypes(aatypes_1, t, res_mask)
        noisy_batch['S_t'] = aatypes_t
        return noisy_batch


    def sample(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            cls=None, w=None,
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()

        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
                if cls is not None:
                    uncond = (2 * torch.ones(X.shape[0], device=X.device)).long()
                    cond = (cls * torch.ones(X.shape[0], device=X.device)).long()
                    model_out_uncond = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all, cls=uncond, t=None)
                    model_out_cond = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all, cls=cond, t=None)
                    model_out = (1+w) * model_out_cond - w * model_out_uncond
                else:
                    model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all, t=None)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                            dim=-1, keepdim=True)
            unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
            
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
            _x = _sample_categorical(q_xs)
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())
            t_1 = t_2

        t_1 = ts[-1]

        return pred_aatypes_1, prot_traj, clean_traj
        
    @torch.no_grad()
    def get_likelihood(self, model, x0, X, mask, chain_M, residue_idx, chain_encoding_all, eps=1e-5, n_samples=1):
        num_batch, num_res = x0.shape
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        aatypes_t_1 = x0
        likelihoods = []
        for _ in range(n_samples):
            log_p_sample_list = []
            t_1 = ts[0]
            for t_2 in ts[1:]:
                d_t = t_2 - t_1
                
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                multiplier = (move_chance_t - move_chance_s)/move_chance_t # [bsz, 1]
                xt = self._corrupt_aatypes(aatypes_t_1, torch.ones((num_batch, 1), device=self._device) * t_1, mask)

                model_out = model(X, xt, mask, chain_M, residue_idx, chain_encoding_all, t=None)
                log_p_x0 = model_out.gather(-1, x0[..., None]).squeeze(-1) # [bsz, seq_len]
                log_p_x0 = log_p_x0 * multiplier
                t_1 = t_2
                log_p_sample_list.append(log_p_x0.sum(dim=-1))
            log_p_sample = torch.stack(log_p_sample_list, dim=0).sum(dim=0)
            likelihoods.append(log_p_sample)
        return torch.stack(likelihoods, dim=0).mean(dim=0)

    def sample_rl_gae_buffer(
        self,
        model,
        value_model,
        reward_model,
        X, mask, chain_M, residue_idx, chain_encoding_all,
        truncate_steps, gumbel_softmax_temp, old_model, old_statics=None, multiple_reward_sampling=0, use_pretrain_for_clip=False
    ):
        assert old_statics is not None
        with torch.no_grad():
            num_batch, num_res = mask.shape
            aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
            num_timesteps = self._cfg.num_timesteps
            ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
            t_1 = ts[0]
            aatypes_t_1 = aatypes_0
            last_x_list = []
            x_argmaxs = []
            move_chance_t_list = []
            copy_flag_list = []
            value_record_list = []
            value_record_masked = []
            rewards = []
            rl_logits = []
            used_for_reward_loss = []
            old_logp = []

            buffer_s0 = []
            buffer_s1 = []
            buffer_s2 = []
            buffer_s3 = []
            buffer_s4 = []
            buffer_s5 = []

            buffer_v = []
            buffer_a = []
            buffer_r = []
            buffer_logp = []
            buffer_t1 = []
            buffer_t2 = []
            buffer_copy_flag = []

            for _ts, t_2 in enumerate(ts[1:]):
                d_t = t_2 - t_1
                packed_s = [X, aatypes_t_1.detach(),  mask, chain_M, residue_idx, chain_encoding_all]
                buffer_s0.append(X.clone())
                buffer_s1.append(aatypes_t_1.detach())
                buffer_s2.append(mask)
                buffer_s3.append(chain_M)
                buffer_s4.append(residue_idx)
                buffer_s5.append(chain_encoding_all)

                model_out = model(X, aatypes_t_1.detach(),  mask, chain_M, residue_idx, chain_encoding_all, t=None) 
                if use_pretrain_for_clip:
                    old_model_out = old_model(X, aatypes_t_1.detach(),  mask, chain_M, residue_idx, chain_encoding_all, t=None)
                    old_pred_logits_1 = old_model_out # [bsz, seqlen, 22]
                    
                    old_pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                    old_pred_logits_1 = old_pred_logits_1 / self._cfg.temp - torch.logsumexp(old_pred_logits_1 / self._cfg.temp, 
                                                                                    dim=-1, keepdim=True)
                    unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                    old_pred_logits_1[unmasked_indices] = self.neg_infinity
                    old_pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                    move_chance_t = 1.0 - t_1
                    move_chance_s = 1.0 - t_2
                    old_q_xs = old_pred_logits_1.exp() * d_t
                    old_q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                value_out = value_model(X, aatypes_t_1.detach(),  mask, chain_M, residue_idx, chain_encoding_all)
                value_record_list.append(value_out)
                buffer_v.append(value_out)
                
                pred_logits_1 = model_out # [bsz, seqlen, 22]
                pred_logits_wo_mask = pred_logits_1.clone()
                pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
                pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)

                pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                                dim=-1, keepdim=True)
                pred_logits_t = pred_logits_1.clone()
                unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                pred_logits_1[unmasked_indices] = self.neg_infinity
                pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                q_xs = pred_logits_1.exp() * d_t
                q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                _x = _sample_categorical(q_xs)
                
                a = _x.unsqueeze(-1)
                buffer_a.append(a)

                if use_pretrain_for_clip:
                    buffer_logp.append(old_q_xs.log().gather(-1, a).squeeze(-1))
                else:
                    buffer_logp.append(q_xs.log().gather(-1, a).squeeze(-1))

                
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                buffer_copy_flag.append(copy_flag)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

                if multiple_reward_sampling>0:
                    true_rewards = []
                    for _ in range(multiple_reward_sampling):
                        _x_arg2max = torch.multinomial(q_xs[:,:,:-1],dim=-1)
                        aatypes_t_3 = aatypes_t_1 * copy_flag + _x_arg2max * (1 - copy_flag)
                        true_reward = reward_model(X, aatypes_t_3, mask, chain_M, residue_idx, chain_encoding_all)
                        true_rewards.append(true_reward)
                    true_reward = torch.stack(true_rewards).mean(dim=0)
                else:
                    _x_arg2max = q_xs[:, :, :-1].argmax(dim=-1)
                    aatypes_t_3 = aatypes_t_1 * copy_flag + _x_arg2max * (1 - copy_flag)
                    true_reward = reward_model(X, aatypes_t_3, mask, chain_M, residue_idx, chain_encoding_all)
                    
                aatypes_t_1 = aatypes_t_2.long()
                
                
                rewards.append(true_reward)
                buffer_r.append(true_reward)
                buffer_t1.append(t_1)
                buffer_t2.append(t_2)
                t_1 = t_2

            rewards = torch.stack(rewards)

            likelihood = self.get_likelihood(old_model, aatypes_t_1, X, mask, chain_M, residue_idx, chain_encoding_all)
            aatypes_t_1_argmax = F.one_hot(aatypes_t_1, num_classes=self.num_tokens).float()

            rewards[-1] = rewards[-1] + np.abs(self._cfg.reward_scale)* (( torch.clamp((likelihood - (old_statics[0]))/(old_statics[1]+1e-8), -4.0, 0.0)) )
            value_record = torch.stack(value_record_list)
            if self._cfg.reward_scale<0.0:
                rewards[:-1] = torch.zeros_like(rewards[:-1])
            else:
                rewards[1:] = rewards[1:] - rewards[:-1]
            value_record = torch.zeros_like(value_record)
            gae = calculate_gae(rewards, value_record, torch.cat( (value_record[1:], torch.zeros_like(value_record[0]).unsqueeze(0)), dim=0), gamma=0.99, lam=0.95)
            buffer_r = gae.clone()
            
            buffer_s0 = torch.stack(buffer_s0)
            buffer_s1 = torch.stack(buffer_s1)
            buffer_s2 = torch.stack(buffer_s2)
            buffer_s3 = torch.stack(buffer_s3)
            buffer_s4 = torch.stack(buffer_s4)
            buffer_s5 = torch.stack(buffer_s5)
            buffer_v = torch.stack(buffer_v)
            buffer_a = torch.stack(buffer_a)
            buffer_t1 = torch.stack(buffer_t1)
            buffer_t2 = torch.stack(buffer_t2)
            buffer_logp = torch.stack(buffer_logp)
            buffer_copy_flag = torch.stack(buffer_copy_flag)

            buffer_s0 = buffer_s0.flatten(0,1)
            buffer_s1 = buffer_s1.flatten(0,1)
            buffer_s2 = buffer_s2.flatten(0,1)
            buffer_s3 = buffer_s3.flatten(0,1)
            buffer_s4 = buffer_s4.flatten(0,1)
            buffer_s5 = buffer_s5.flatten(0,1)
            buffer_v = buffer_v.flatten(0,1)
            buffer_a = buffer_a.flatten(0,1)
            buffer_r = buffer_r.flatten(0,1)
            buffer_t1 = buffer_t1[:,None].repeat(1, num_batch)
            buffer_t2 = buffer_t2[:,None].repeat(1, num_batch)
            buffer_logp = buffer_logp.flatten(0,1)
            
            buffer_t1 = buffer_t1.flatten(0,1)
            buffer_t2 = buffer_t2.flatten(0,1)
            buffer_t1 = buffer_t1.to(X.device)
            buffer_t2 = buffer_t2.to(X.device)

            buffer_copy_flag = buffer_copy_flag.flatten(0,1)
            

            return aatypes_t_1_argmax, buffer_s0, buffer_s1, buffer_s2, buffer_s3, buffer_s4, buffer_s5 , buffer_v, buffer_a, buffer_r, buffer_logp, buffer_t1, buffer_t2, buffer_copy_flag

    def sample_gradient(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            truncate_steps, gumbel_softmax_temp
        ):
        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        aatypes_0 = F.one_hot(aatypes_0, num_classes=self.num_tokens).float()
        
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0
        last_x_list = []
        move_chance_t_list = []
        copy_flag_list = []

        for _ts, t_2 in enumerate(ts[1:]):
            d_t = t_2 - t_1
            model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            pred_logits_1 = model_out.clone()
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)

            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)
            if aatypes_t_1.ndim > 2 and aatypes_t_1.shape[-1] == self.num_tokens:
                aatypes_t_1_argmax = aatypes_t_1.argmax(dim=-1)
            else:
                aatypes_t_1_argmax = aatypes_t_1
            unmasked_indices = (aatypes_t_1_argmax != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1_argmax[unmasked_indices]] = 0
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
            if _ts < num_timesteps - truncate_steps:
                _x = _sample_categorical(q_xs)
                _x = F.one_hot(_x, num_classes=self.num_tokens).float()
                copy_flag = (aatypes_t_1.argmax(dim=-1) != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype).unsqueeze(-1)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)
                aatypes_t_2 = aatypes_t_2.detach()
                aatypes_t_1 = aatypes_t_1.detach()
            else:
                q_xs = q_xs + 1e-8
                _x = _sample_categorical_gradient(q_xs, gumbel_softmax_temp)
                copy_flag = 1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

            last_x_list.append(aatypes_t_1)
            move_chance_t_list.append(move_chance_t + self._cfg.min_t)
            copy_flag_list.append(copy_flag)
            aatypes_t_1 = aatypes_t_2
            t_1 = t_2

        last_x_list.append(aatypes_t_1)
        move_chance_t_list.append(1.0 - t_1 + self._cfg.min_t)
        copy_flag_list.append(1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))

        aatypes_t_1_argmax = aatypes_t_1[:, :, :-1].argmax(dim=-1) # to avoid the mask token
        aatypes_t_1_argmax = F.one_hot(aatypes_t_1_argmax, num_classes=self.num_tokens).float()
        return aatypes_t_1 + (aatypes_t_1_argmax - aatypes_t_1).detach(), last_x_list, move_chance_t_list, copy_flag_list


    def sample_controlled_CG(self,
                              model,
                              X, mask, chain_M, residue_idx, chain_encoding_all,
                              guidance_scale, reward_model):
        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                            dim=-1, keepdim=True)
            unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t

            x_onehot = F.one_hot(aatypes_t_1, num_classes=self.num_tokens).float()
            x_grad = self.compute_gradient_CG(model, x_onehot, reward_model, X, mask, chain_M, residue_idx, chain_encoding_all)
            guidance = guidance_scale * (x_grad - x_grad[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
            q_xs = q_xs * guidance.exp()
            _x = _sample_categorical(q_xs)
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())
            t_1 = t_2

        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj

    def compute_gradient_CG(self, model, x_onehot, reward_model,
                             X, mask, chain_M, residue_idx, chain_encoding_all):
        x_onehot.requires_grad_(True)
        expected_x0 = model(X, x_onehot, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
        scores = reward_model(X, expected_x0, mask, chain_M, residue_idx, chain_encoding_all)
        scores = scores.mean()
        scores.backward()
        x_grad = x_onehot.grad.clone()
        return x_grad


    def sample_controlled_SMC(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, alpha
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())
            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                            dim=-1, keepdim=True)
            unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
            
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
            _x = _sample_categorical(q_xs)
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

            '''
            Calcualte exp(v_{t-1}(x_{t-1})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_2, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_2.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_2 + (1 - copy_flag) *  one_hot_x0
            reward_num = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            '''
            Calcualte exp(v_{t}(x_{t})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_1 + (1 - copy_flag) *  one_hot_x0
            reward_den = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) # Now calculate exp( (v_{t-1}(x_{t-1) -v_{t}(x_{t}) /alpha) 
            ratio = ratio.detach().cpu().numpy()
            final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
            aatypes_t_2 = aatypes_t_2[final_sample_indices]
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())
            t_1 = t_2

        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj



    def sample_controlled_TDS(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, alpha, guidance_scale
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())
            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                            dim=-1, keepdim=True)
            unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t
            x_onehot = F.one_hot(aatypes_t_1, num_classes=self.num_tokens).float()
            x_grad = self.compute_gradient_CG(model, x_onehot, reward_model, X, mask, chain_M, residue_idx, chain_encoding_all)
            guidance = guidance_scale * (x_grad - x_grad[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))

            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
            q_xs = q_xs * guidance.exp()
            _x = _sample_categorical(q_xs)
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)
            prob_multiplier = (1 - copy_flag) * torch.gather(guidance.exp(), 2, _x.unsqueeze(-1)).squeeze(-1) + copy_flag * torch.ones_like(_x)

            '''
            Calcualte exp(v_{t-1}(x_{t-1})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_2, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_2.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_2 + (1 - copy_flag) *  one_hot_x0
            reward_num = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            '''
            Calcualte exp(v_{t}(x_{t})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_1 + (1 - copy_flag) *  one_hot_x0
            reward_den = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) / prob_multiplier.prod(dim=-1) # Now calculate exp( (v_{t-1}(x_{t-1) -v_{t}(x_{t}) /alpha) 
            ratio = ratio.detach().cpu().numpy()
            final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
            aatypes_t_2 = aatypes_t_2[final_sample_indices]
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())
            t_1 = t_2

        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj


def fm_model_step(model, noisy_batch, cls=None):
    loss_mask = noisy_batch['mask'] * noisy_batch['chain_M']
    if torch.any(torch.sum(loss_mask, dim=-1) < 1):
        raise ValueError('Empty batch encountered')
    
    # Model output predictions.
    X = noisy_batch['X']
    aatypes_t_1 = noisy_batch['S_t']
    mask = noisy_batch['mask']
    chain_M = noisy_batch['chain_M']
    residue_idx = noisy_batch['residue_idx']
    chain_encoding_all = noisy_batch['chain_encoding_all']
    pred_logits = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all, cls=cls)

    return pred_logits
