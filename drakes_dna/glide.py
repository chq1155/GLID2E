import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import oracle
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import argparse
import wandb
import os
import datetime
import time
import psutil
import json
import gc
from utils import str2bool, set_seed


def check_nan_inf(tensor, name="tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains NaN or Inf")
        return True
    return False

def safe_log(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps))

def safe_exp(x, max_val=20.0):
    return torch.exp(torch.clamp(x, max=max_val))

def safe_normalize(x, eps=1e-6):
    mean_x = x.mean()
    std_x = x.std()
    if std_x < eps:
        return torch.zeros_like(x)
    return (x - mean_x) / (std_x + eps)

class PerformanceMonitor:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.epoch_metrics = []
        self.process = psutil.Process()
        
    def get_memory_usage(self):
        cpu_memory = self.process.memory_info().rss / 1024 / 1024
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
        return cpu_memory, gpu_memory
    
    def start_epoch(self, epoch_num):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        cpu_mem, gpu_mem = self.get_memory_usage()
        
        self.current_epoch = {
            'epoch': epoch_num,
            'start_time': time.time(),
            'start_cpu_memory': cpu_mem,
            'start_gpu_memory': gpu_mem,
            'peak_cpu_memory': cpu_mem,
            'peak_gpu_memory': gpu_mem
        }
        
    def update_peak_memory(self):
        cpu_mem, gpu_mem = self.get_memory_usage()
        self.current_epoch['peak_cpu_memory'] = max(self.current_epoch['peak_cpu_memory'], cpu_mem)
        self.current_epoch['peak_gpu_memory'] = max(self.current_epoch['peak_gpu_memory'], gpu_mem)
        
    def end_epoch(self):
        self.update_peak_memory()
        
        end_time = time.time()
        cpu_mem, gpu_mem = self.get_memory_usage()
        
        self.current_epoch.update({
            'end_time': end_time,
            'duration': end_time - self.current_epoch['start_time'],
            'end_cpu_memory': cpu_mem,
            'end_gpu_memory': gpu_mem,
            'cpu_memory_delta': cpu_mem - self.current_epoch['start_cpu_memory'],
            'gpu_memory_delta': gpu_mem - self.current_epoch['start_gpu_memory']
        })
        
        self.epoch_metrics.append(self.current_epoch.copy())
        
    def get_average_metrics(self, num_epochs=5):
        if len(self.epoch_metrics) < num_epochs:
            return None
            
        metrics = self.epoch_metrics[:num_epochs]
        
        avg_metrics = {
            'num_epochs_averaged': num_epochs,
            'avg_duration': np.mean([m['duration'] for m in metrics]),
            'avg_peak_cpu_memory': np.mean([m['peak_cpu_memory'] for m in metrics]),
            'avg_peak_gpu_memory': np.mean([m['peak_gpu_memory'] for m in metrics]),
            'avg_cpu_memory_delta': np.mean([m['cpu_memory_delta'] for m in metrics]),
            'avg_gpu_memory_delta': np.mean([m['gpu_memory_delta'] for m in metrics]),
            'std_duration': np.std([m['duration'] for m in metrics]),
            'std_peak_cpu_memory': np.std([m['peak_cpu_memory'] for m in metrics]),
            'std_peak_gpu_memory': np.std([m['peak_gpu_memory'] for m in metrics]),
            'total_training_time': sum([m['duration'] for m in metrics])
        }
        
        return avg_metrics
    
    def save_report(self, save_path, num_epochs=5):
        avg_metrics = self.get_average_metrics(num_epochs)
        if avg_metrics is None:
            print(f"Not enough epochs to generate report (need {num_epochs}, got {len(self.epoch_metrics)})")
            return
            
        report = {
            'summary': avg_metrics,
            'per_epoch_details': self.epoch_metrics[:num_epochs],
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        report_path = os.path.join(save_path, 'performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        readable_path = os.path.join(save_path, 'performance_summary.txt')
        with open(readable_path, 'w') as f:
            f.write("=== Training Performance Report ===\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Epochs analyzed: {num_epochs}\n\n")
            
            f.write("=== Average Metrics (First 5 Epochs) ===\n")
            f.write(f"Average epoch duration: {avg_metrics['avg_duration']:.2f} ± {avg_metrics['std_duration']:.2f} seconds\n")
            f.write(f"Total training time: {avg_metrics['total_training_time']:.2f} seconds\n")
            f.write(f"Average peak CPU memory: {avg_metrics['avg_peak_cpu_memory']:.2f} ± {avg_metrics['std_peak_cpu_memory']:.2f} MB\n")
            f.write(f"Average peak GPU memory: {avg_metrics['avg_peak_gpu_memory']:.2f} ± {avg_metrics['std_peak_gpu_memory']:.2f} MB\n")
            f.write(f"Average CPU memory delta: {avg_metrics['avg_cpu_memory_delta']:.2f} MB\n")
            f.write(f"Average GPU memory delta: {avg_metrics['avg_gpu_memory_delta']:.2f} MB\n\n")
            
            f.write("=== System Information ===\n")
            f.write(f"CPU cores: {report['system_info']['cpu_count']}\n")
            f.write(f"Total system memory: {report['system_info']['total_memory_gb']:.2f} GB\n")
            f.write(f"GPU available: {report['system_info']['gpu_available']}\n")
            f.write(f"GPU count: {report['system_info']['gpu_count']}\n")
            f.write(f"GPU name: {report['system_info']['gpu_name']}\n\n")
            
            f.write("=== Per-Epoch Details ===\n")
            for i, epoch_data in enumerate(report['per_epoch_details']):
                f.write(f"Epoch {i}:\n")
                f.write(f"  Duration: {epoch_data['duration']:.2f}s\n")
                f.write(f"  Peak CPU Memory: {epoch_data['peak_cpu_memory']:.2f} MB\n")
                f.write(f"  Peak GPU Memory: {epoch_data['peak_gpu_memory']:.2f} MB\n")
                f.write(f"  CPU Memory Delta: {epoch_data['cpu_memory_delta']:.2f} MB\n")
                f.write(f"  GPU Memory Delta: {epoch_data['gpu_memory_delta']:.2f} MB\n\n")
        
        print(f"Performance report saved to: {report_path}")
        print(f"Human-readable summary saved to: {readable_path}")

def get_ema_avg_fn(decay=0.999):
    """Get the function applying exponential moving average (EMA) across a single param."""

    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update

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
        delta = rewards[t] + gamma * next_values *nd - values[t]
        gae = delta + gamma * lam * nd * gae
        advantages[t] = gae
    return advantages

def calculate_return_to_go(rewards, gamma=1.0):
    """
    Calculate Return to Go.

    Args:
        rewards (np.ndarray): Array of rewards.
        gamma (float): Discount factor.

    Returns:
        np.ndarray: Array of Return to Go estimates.
    """
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards) - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns

def fine_tune(new_model,  new_model_y, new_model_y_eval, old_model, value_model, value_model_ema, args, eps=1e-5):
    monitor = PerformanceMonitor()

    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')

    new_model.config.finetuning.truncate_steps = args.truncate_steps
    new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
    dt = (1 - eps) / args.total_num_steps
    new_model.train()
    old_model.eval()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.num_epochs)
    value_optim = torch.optim.Adam(value_model.parameters(), lr=args.learning_rate)
    value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optim, args.num_epochs)
    batch_losses = []
    batch_rewards = []

    # static old model
    logps = []
    for _step in range(args.num_accum_steps*10):
        sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list, logp_list, raw_x_list = new_model._sample_finetune_rl(eval_sp_size=args.batch_size, copy_flag_temp=args.copy_flag_temp) # [bsz, seqlen, 4]
        logps.append(logp_list.detach())
    
    logps = torch.cat(logps, 0)
    logps_mean, logps_std = logps.mean(), logps.std()

    import matplotlib.pyplot as plt
    plt.hist(logps.cpu().numpy().flatten(), bins=100)
    plt.savefig('./logps_hist.png')
    plt.close()

    for epoch_num in range(args.num_epochs):
        monitor.start_epoch(epoch_num)
        
        rewards = []
        rewards_eval = []
        losses = []
        reward_losses = []
        value_losses = []
        kl_losses = []
        ratios = []
        tot_grad_norm = 0.0
        new_model.train()
        value_model.train()
        replay_buffer_x = []
        replay_buffer_a = []
        replay_buffer_cond = []
        replay_buffer_r = []
        replay_buffer_logp = []
        replay_buffer_t = []
        
        for _step in range(args.num_accum_steps):
            sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list, logp_list, old_logp_list, raw_x_list, reward_list, input_xs, value_list,t_list = new_model._sample_finetune_rl_old(eval_sp_size=args.batch_size, copy_flag_temp=args.copy_flag_temp, old_model=old_model, reward_model=new_model_y, value_model=value_model) # [bsz, seqlen, 4]
            sample2 = torch.transpose(sample, 1, 2)
            preds = new_model_y(sample2).squeeze(-1) # [bsz, 3]
            reward = preds[:, 0]

            sample_argmax = torch.argmax(sample, 2)
            sample_argmax = 1.0 * F.one_hot(sample_argmax, num_classes= 4)
            sample_argmax = torch.transpose(sample_argmax, 1, 2)

            preds_argmax = new_model_y(sample_argmax).squeeze(-1)
            reward_argmax = preds_argmax[:, 0]
            rewards.append(reward_argmax.detach().cpu().numpy())
            
            preds_eval = new_model_y_eval(sample_argmax).squeeze(-1)
            reward_argmax_eval = preds_eval[:, 0]
            rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())
            
            monitor.update_peak_memory()
            
            total_kl = []
            
            if epoch_num < args.alpha_schedule_warmup:
                current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
            else:
                current_alpha = args.alpha

            
            value_list = torch.stack(value_list, 0)
            raw_x_list = torch.stack(raw_x_list, 0)
            condt_list = torch.stack(condt_list, 0)
            reward_list = torch.stack(reward_list, 0)
            t_list = torch.stack(t_list, 0)
            reward_list[-1] += current_alpha * torch.clamp((old_logp_list-logps_mean+args.ab_K*logps_std).detach()/(logps_std+1e-6), min=-args.range_min, max=args.range_max)
            if args.use_reward_shaping:
                reward_list[1:] = reward_list[1:]-reward_list[:-1]
            else:
                reward_list[:-1] = torch.zeros_like(reward_list[:-1])
            return_list = calculate_gae(reward_list, value_list[:-1], value_list[-1], gamma=0.99, lam=0.95)
            replay_buffer_x.append(torch.stack(input_xs, 0).flatten(0, 1))
            replay_buffer_a.append(raw_x_list.flatten(0, 1))
            replay_buffer_cond.append(condt_list.flatten(0, 1))
            replay_buffer_r.append(return_list.flatten(0, 1))
            replay_buffer_logp.append(torch.stack(logp_list, 0).flatten(0, 1))
            replay_buffer_t.append(t_list.flatten(0, 1))
            
        replay_buffer_x = torch.cat(replay_buffer_x, 0)
        replay_buffer_a = torch.cat(replay_buffer_a, 0)
        replay_buffer_cond = torch.cat(replay_buffer_cond, 0)
        replay_buffer_r = torch.cat(replay_buffer_r, 0)
        replay_buffer_logp = torch.cat(replay_buffer_logp, 0)
        replay_buffer_t = torch.cat(replay_buffer_t, 0)
        

        perm = torch.randperm(replay_buffer_x.size(0))

        for i in range(0, replay_buffer_x.size(0), args.batch_size*16):
            indices = perm[i:i+args.batch_size*16]
            batch_x = replay_buffer_x[indices]
            batch_a = replay_buffer_a[indices]
            batch_cond = replay_buffer_cond[indices]
            batch_r = replay_buffer_r[indices]
            batch_logp = replay_buffer_logp[indices]
            batch_t = replay_buffer_t[indices]
            
            if (check_nan_inf(batch_x, "batch_x") or 
                check_nan_inf(batch_a, "batch_a") or 
                check_nan_inf(batch_r, "batch_r") or 
                check_nan_inf(batch_logp, "batch_logp")):
                print(f"Skipping batch {i} due to NaN in input")
                continue
            
            log_p_x0 = new_model.forward(batch_x, batch_cond)
            
            if check_nan_inf(log_p_x0, "log_p_x0"):
                print(f"Skipping batch {i} due to NaN in model output")
                continue
            
            sigma_t, _ = new_model.noise(batch_t)
            sigma_s, _ = new_model.noise(batch_t - (1-1e-5)/new_model.config.sampling.steps)

            move_chance_t = 1 - torch.exp(-sigma_t)
            move_chance_s = 1 - torch.exp(-sigma_s)
            move_chance_t = move_chance_t[:, None]
            move_chance_s = move_chance_s[:, None]
            
            q_xs = safe_exp(log_p_x0) * (move_chance_t - move_chance_s)
            q_xs[:, :, new_model.mask_index] = move_chance_s[:, 0]
            
            log_p_x0 = safe_log(q_xs).gather(2, batch_a.unsqueeze(-1)).squeeze(-1)
            
            if check_nan_inf(log_p_x0, "computed_log_p_x0"):
                print(f"Skipping batch {i} due to NaN in log probability")
                continue
            
            normed_batch_r = safe_normalize(batch_r)
            
            log_p_x0_old = batch_logp
            
            log_diff =log_p_x0 - log_p_x0_old
            ratio = safe_exp(log_diff)
            
            if check_nan_inf(ratio, "ratio"):
                print(f"Skipping batch {i} due to NaN in ratio")
                continue
            
            ratioA = ratio * normed_batch_r.unsqueeze(-1)
            ratioB = torch.clamp(ratio, 1-0.05, 1+0.05) * normed_batch_r.unsqueeze(-1)
            clip_ratio = ((ratio-1.0).abs()>0.05).float().mean().detach().cpu().numpy()
            ratios.append(clip_ratio)
            
            reward_loss = -torch.min(ratioA, ratioB).mean() + args.beta * log_p_x0.mean(dim=1).mean()
            
            if check_nan_inf(reward_loss, "reward_loss"):
                print(f"Skipping batch {i} due to NaN in reward_loss")
                continue
            
            loss = reward_loss
            
            value_pred = value_model.value_forward(batch_x, batch_cond).squeeze()
            
            if check_nan_inf(value_pred, "value_pred"):
                print(f"Skipping batch {i} due to NaN in value_pred")
                continue
            
            value_loss = F.smooth_l1_loss(value_pred, batch_r + value_pred.detach())
            
            if check_nan_inf(value_loss, "value_loss"):
                print(f"Skipping batch {i} due to NaN in value_loss")
                continue
            
            value_losses.append(value_loss.cpu().detach().numpy())
            loss = loss + value_loss
            
            if check_nan_inf(loss, "total_loss"):
                print(f"Skipping batch {i} due to NaN in total_loss")
                continue
            
            if not torch.isfinite(loss):
                print(f"Skipping batch {i} due to infinite loss: {loss.item()}")
                continue
            
            loss.backward()
            
            has_nan_grad = False
            for name, param in new_model.named_parameters():
                if param.grad is not None and check_nan_inf(param.grad, f"grad_{name}"):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Skipping optimization step {i} due to NaN gradients")
                optim.zero_grad()
                value_optim.zero_grad()
                continue
            
            reward_losses.append(reward_loss.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())
            
            norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
            value_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.gradnorm_clip)
            tot_grad_norm += norm
            
            optim.step()
            value_optim.step()
            optim.zero_grad()
            value_optim.zero_grad()
            
            monitor.update_peak_memory()
            
        rewards = np.array(rewards)
        rewards_eval = np.array(rewards_eval)
        losses = np.array(losses)
        reward_losses = np.array(reward_losses)
        value_losses = np.array(value_losses)
        ratios = np.array(ratios)

        scheduler.step()
        value_scheduler.step()
        
        monitor.end_epoch()
        
        epoch_duration = monitor.epoch_metrics[-1]['duration']
        peak_cpu_mem = monitor.epoch_metrics[-1]['peak_cpu_memory']
        peak_gpu_mem = monitor.epoch_metrics[-1]['peak_gpu_memory']
        
        print("Epoch %d"%epoch_num, "Mean reward %f"%np.mean(rewards), "Mean reward eval %f"%np.mean(rewards_eval), 
        "Mean grad norm %f"%tot_grad_norm, "Mean loss %f"%np.mean(losses), "Mean reward loss %f"%np.mean(reward_losses),  "median_reward_eval %f" % np.median(rewards_eval), "Mean value loss %f"%np.mean(value_losses),
        "Mean ratios %f"%np.mean(ratios), f"Time: {epoch_duration:.2f}s", f"Peak CPU: {peak_cpu_mem:.2f}MB", f"Peak GPU: {peak_gpu_mem:.2f}MB"
        )
        
        if args.name != 'debug':
            wandb.log({"epoch": epoch_num, "mean_reward": np.mean(rewards), "mean_reward_eval": np.mean(rewards_eval), "median_reward_eval": np.median(rewards_eval),
            "mean_grad_norm": tot_grad_norm, "mean_loss": np.mean(losses), "mean reward loss": np.mean(reward_losses), "mean value loss": np.mean(value_losses),
            "mean_ratios": np.mean(ratios), "epoch_duration": epoch_duration, "peak_cpu_memory": peak_cpu_mem, "peak_gpu_memory": peak_gpu_mem}) 
            
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch_num} Mean reward {np.mean(rewards)} Mean reward eval {np.mean(rewards_eval)} Mean grad norm {tot_grad_norm} Mean loss {np.mean(losses)} Mean reward loss {np.mean(reward_losses)} Time: {epoch_duration:.2f}s Peak CPU: {peak_cpu_mem:.2f}MB Peak GPU: {peak_gpu_mem:.2f}MB\n")
        
        if (epoch_num+1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")
            
        if epoch_num == 4: 
            monitor.save_report(save_path, num_epochs=5)
            print("Performance report generated after 5 epochs!")
            
    if args.name != 'debug':
        wandb.finish()

    return batch_losses

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default='') # TODO
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=1000)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
argparser.add_argument('--gradnorm_clip', type=float, default=1.0)
argparser.add_argument('--batch_size', type=int, default=8)
argparser.add_argument('--name', type=str, default='debug')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=50)
argparser.add_argument('--alpha', type=float, default=0.001)
argparser.add_argument('--beta', type=float, default=0.001)
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--use_reward_shaping", type=str2bool, default=False)
argparser.add_argument("--range_min", type=float, default=-3.0)
argparser.add_argument("--range_max", type=float, default=3.0)
argparser.add_argument("--ab_K", type=float, default=1.0)
args = argparser.parse_args()
print(args)

# pretrained model path
CKPT_PATH = os.path.join(args.base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
log_base_dir = os.path.join(args.base_path, 'mdlm/glide')

# reinitialize Hydra    
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration
initialize(config_path="configs_gosai", job_name="load_model")
cfg = compose(config_name="config_gosai.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# initialize a log file
if args.name == 'debug':
    print("Debug mode")
    save_path = os.path.join(log_base_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'log.txt')
else:
    run_name = f'alpha{args.alpha}_beta{args.beta}_range{args.range_min, args.range_max}_abK{args.ab_K}_accum{args.num_accum_steps}_bsz{args.batch_size}_shaping{args.use_reward_shaping}_temp{args.gumbel_temp}_seed{args.seed}_{args.name}_{curr_time}'
    save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project='glide', name=run_name, config=args, dir=save_path)
    log_path = os.path.join(save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

# Initialize the model
new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
for param in old_model.parameters():
    param.requires_grad = False
reward_model = oracle.get_gosai_oracle(mode='train').to(new_model.device)
reward_model_eval = oracle.get_gosai_oracle(mode='eval').to(new_model.device)
reward_model.eval()
reward_model_eval.eval()
value_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)

value_model_ema = torch.optim.swa_utils.AveragedModel(value_model, new_model.device,
            get_ema_avg_fn(0.9))
value_model_ema.eval()
fine_tune(new_model, reward_model, reward_model_eval, old_model, value_model, value_model_ema, args)
