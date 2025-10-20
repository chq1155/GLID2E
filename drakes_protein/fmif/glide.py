import argparse
import os.path
import random
import string
import datetime
import pickle
import wandb
from protein_oracle.utils import str2bool
import pyrosetta
pyrosetta.init(extra_options="-out:level 100")
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from biotite.sequence.io import fasta
import shutil
from protein_oracle.data_utils import ALPHABET
import pandas as pd
import numpy as np
import torch
import einops

runid = ''.join(random.choice(string.ascii_letters) for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


def cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, save_path, base_path):
    with torch.no_grad():
        results_list = []
        run_name = save_path.split('/')
        if run_name[-1] == '':
            run_name = run_name[-2]
        else:
            run_name = run_name[-1]
        
        sc_output_dir = os.path.join(base_path, 'sc_tmp', run_name, 'sc_output', batch["protein_name"][0][:-4])
        the_pdb_path = os.path.join(pdb_path, batch['WT_name'][0])
        # fold the ground truth sequence
        os.makedirs(os.path.join(sc_output_dir, 'true_seqs'), exist_ok=True)
        true_fasta = fasta.FastaFile()
        true_detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(S[0]) if mask_for_loss[0][_ix] == 1])
        true_fasta['true_seq_1'] = true_detok_seq
        true_fasta_path = os.path.join(sc_output_dir, 'true_seqs', 'true.fa')
        true_fasta.write(true_fasta_path)
        true_folded_dir = os.path.join(sc_output_dir, 'true_folded')
        if os.path.exists(true_folded_dir):
            shutil.rmtree(true_folded_dir)
        os.makedirs(true_folded_dir, exist_ok=False)
        true_folded_output = the_folding_model.fold_fasta(true_fasta_path, true_folded_dir)
        
        true_folded_pdb_path = os.path.join(true_folded_dir, 'folded_true_seq_1.pdb')
        true_folded_pose = pyrosetta.pose_from_file(true_folded_pdb_path)
        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(true_folded_pose))
        packer.apply(true_folded_pose)
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.apply(true_folded_pose)
        true_folded_relax_path = os.path.join(sc_output_dir, 'true_folded_relax', 'folded_true_relax_seq_1.pdb')
        os.makedirs(os.path.join(sc_output_dir, 'true_folded_relax'), exist_ok=True)
        true_folded_pose.dump_pdb(true_folded_relax_path)

        true_pose = pyrosetta.pose_from_file(the_pdb_path)
        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(true_pose))
        packer.apply(true_pose)
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.apply(true_pose)

        os.makedirs(os.path.join(sc_output_dir, 'true_relax'), exist_ok=True)
        true_pose.dump_pdb(os.path.join(sc_output_dir, 'true_relax', 'true_relax_seq_1.pdb'))

        foldtrue_true_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, true_folded_pose)

        for _it, ssp in enumerate(S_sp):
            os.makedirs(sc_output_dir, exist_ok=True)
            os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
            codesign_fasta = fasta.FastaFile()
            detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            codesign_fasta['codesign_seq_1'] = detok_seq
            codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
            codesign_fasta.write(codesign_fasta_path)

            folded_dir = os.path.join(sc_output_dir, 'folded')
            if os.path.exists(folded_dir):
                shutil.rmtree(folded_dir)
            os.makedirs(folded_dir, exist_ok=False)

            folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            gen_folded_pdb_path = os.path.join(folded_dir, 'folded_codesign_seq_1.pdb')
            pose = pyrosetta.pose_from_file(gen_folded_pdb_path)
            scorefxn = pyrosetta.create_score_function("ref2015_cart")
            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
            packer.apply(pose)
            relax = FastRelax()
            relax.set_scorefxn(scorefxn)
            relax.apply(pose)

            os.makedirs(os.path.join(sc_output_dir, 'folded_relax'), exist_ok=True)
            pose.dump_pdb(os.path.join(sc_output_dir, 'folded_relax', 'folded_codesign_relax_seq_1.pdb'))
            gen_true_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, pose)
            gen_foldtrue_bbrmsd = pyrosetta.rosetta.core.scoring.bb_rmsd(true_folded_pose, pose)
            seq_revovery = (S_sp[_it] == S[0]).float().mean().item()
            resultdf = pd.DataFrame(columns=['gen_true_bb_rmsd', 'gen_foldtrue_bb_rmsd', 'foldtrue_true_bb_rmsd', 'seq_recovery'])
            resultdf.loc[0] = [gen_true_bbrmsd, gen_foldtrue_bbrmsd, foldtrue_true_bbrmsd, seq_revovery]
            resultdf['seq'] = detok_seq
            resultdf['true_seq'] = true_detok_seq
            resultdf['protein_name'] = batch['protein_name'][0]
            resultdf['WT_name'] = batch['WT_name'][0]
            resultdf['num'] = _it
            resultdf['pdbpath'] = sc_output_dir
            results_list.append(resultdf)

    return results_list

def parse_df(results_df):
    avg_rmsd = results_df['bb_rmsd'].mean()
    success_rate = results_df['bb_rmsd'].apply(lambda x: 1 if x < 2 else 0).mean()
    return avg_rmsd, success_rate, np.format_float_positional(avg_rmsd, unique=False, precision=3), np.format_float_positional(success_rate, unique=False, precision=3)

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


def main(args, log_path, save_path):
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import os.path
    from protein_oracle.utils import set_seed
    from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
    from protein_oracle.model_utils import ProteinMPNNOracle
    from fmif.model_utils import ProteinMPNNFMIF,ProteinMPNNFMIF_T
    from fmif.fm_utils import Interpolant, _sample_categorical_gradient
    import fmif.model_utils as mu
    from tqdm import tqdm
    from multiflow.models import folding_model
    from types import SimpleNamespace

    scaler = torch.cuda.amp.GradScaler()
    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    pdb_path = os.path.join(args.base_path, 'proteindpo_data/AlphaFold_model_PDBs')
    max_len = 75  # Define the maximum length of proteins
    dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    set_seed(args.seed, use_cuda=True)

    # make a dict of pdb filename: index
    for batch in loader:
        pdb_structures = batch[0]
        pdb_filenames = batch[1]
        pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
        break

    dpo_dict_path = os.path.join(args.base_path, 'proteindpo_data/processed_data')
    dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_wt.pkl'), 'rb'))
    dpo_train_dataset = ProteinDPODataset(dpo_train_dict, pdb_idx_dict, pdb_structures)
    
    
    
    _batch_size = args.batch_size // args.resample if args.resample > 0 else args.batch_size
    loader_train = DataLoader(dpo_train_dataset, batch_size=_batch_size, shuffle=True, drop_last=True)
    print("dataset length ",len(dpo_train_dataset))
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
    dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
    loader_test = DataLoader(dpo_test_dataset, batch_size=1, shuffle=False)

    new_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        augment_eps=args.backbone_noise
                        )
    new_fmif_model.to(device)
    new_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'pmpnn/outputs/pretrained_if_model.pt'))['model_state_dict'], strict=False)
    new_fmif_model.finetune_init()

    old_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        augment_eps=args.backbone_noise
                        )
    old_fmif_model.to(device)
    old_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'pmpnn/outputs/pretrained_if_model.pt'))['model_state_dict'], strict=False)
    old_fmif_model.finetune_init()
    old_fmif_model.eval()
    
    noise_interpolant = Interpolant(args)
    noise_interpolant.set_device(device)

    # load the reward model
    reward_model = ProteinMPNNOracle(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    reward_model.to(device)
    reward_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_ft.pt'))['model_state_dict'])
    reward_model.finetune_init()


    value_model = ProteinMPNNOracle(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    value_model.to(device)
    value_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_ft.pt'))['model_state_dict'],
        strict=False)
    value_model.finetune_init()
    value_model.train()

    # create moving average model for value_model
    # Compute exponential moving averages of the weights and buffers
    value_model_ema = torch.optim.swa_utils.AveragedModel(value_model, device,
            get_ema_avg_fn(0.0))
    value_model_ema.eval()

    for param in old_fmif_model.parameters():
        param.requires_grad = False
    # new_fmif_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model.eval()

    # reward model for evaluation
    reward_model_eval = ProteinMPNNOracle(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    reward_model_eval.to(device)
    reward_model_eval.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_eval.pt'))['model_state_dict'])
    reward_model_eval.finetune_init()
    for param in reward_model_eval.parameters():
        param.requires_grad = False
    reward_model_eval.eval()

    folding_cfg = {
        'seq_per_sample': 1,
        'folding_model': 'esmf',
        'own_device': False,
        'pmpnn_path': './ProteinMPNN/',
        'pt_hub_dir': os.path.join(args.base_path, '.cache/torch/'),
        'colabfold_path': os.path.join(args.base_path, 'colabfold-conda/bin/colabfold_batch') # for AF2
    }
    folding_cfg = SimpleNamespace(**folding_cfg)
    the_folding_model = folding_model.FoldingModel(folding_cfg)

    new_fmif_model.train()
    optim = torch.optim.Adam(new_fmif_model.parameters(), lr=args.lr, weight_decay=args.wd)
    optim_value = torch.optim.Adam(value_model.parameters(), lr=1e-3, weight_decay=args.wd)

    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs, eta_min=1e-8)
    learning_rate_scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(optim_value, T_max=args.num_epochs, eta_min=1e-8)
    torch.autograd.set_detect_anomaly(True)
    wandb_step = 0
    import einops
    old_statics = (None, None)


    likelihoods = []
    for _step, batch in tqdm(enumerate(loader_train)):
        X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
        x0 = noise_interpolant.sample(old_fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)[0]
        likelihood = noise_interpolant.get_likelihood(old_fmif_model, x0, X, mask, chain_M, residue_idx, chain_encoding_all)
        likelihoods.append(likelihood)
    likelihood = torch.cat(likelihoods, dim=0)
    likelihood = likelihood.reshape(-1)
    print(likelihood.mean(), likelihood.std())
    likelihood_stats = [likelihood.mean().item(), likelihood.std().item()]
    for epoch_num in range(1, args.num_epochs+1):
        rewards = []
        rewards_argmax = []
        rewards_eval = []
        rewrads_epoch = []
        rewards_argmax_epoch = []
        rewards_eval_epoch = []
        record_value_losses = []
        record_value_losses_epoch = []
        losses = []
        reward_losses = []
        kl_losses = []
        tot_grad_norm = 0.0
        new_fmif_model.train()
        train_sp_acc, train_sp_weights = 0., 0.
        results_merge = []
        clip_ratio = []
        clip_ratio_epoch = []

        kl_losses_epoch = []
        rewards_epoch = []
        rewards_argmax_epoch = []
        rewards_eval_epoch = []
        clip_ratio_epoch= []
        record_value_losses_epoch=[]
        reward_losses_epoch=[]

        for _step, batch in tqdm(enumerate(loader_train)):
            X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
            if args.resample>0:
                
                inputs = [X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt]
                inputs = [einops.repeat(x,'b ... -> (r b) ...', r=args.resample) for x in inputs]
                X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = inputs
            
            S_sp, buffer_s0, buffer_s1, buffer_s2, buffer_s3, buffer_s4, buffer_s5 , buffer_v, buffer_a, buffer_r, buffer_logp, buffer_t1, buffer_t2, buffer_copy_flag = noise_interpolant.sample_rl_gae_buffer(new_fmif_model, value_model, 
            reward_model, X, mask, chain_M, residue_idx, chain_encoding_all, args.truncate_steps, args.gumbel_softmax_temp, old_fmif_model, likelihood_stats)
            
            with torch.no_grad():
                S_sp_argmax = S_sp.argmax(dim=-1)
                print(S_sp_argmax[0])
                assert 0
                dg_pred_argmax = reward_model(X, S_sp_argmax, mask, chain_M, residue_idx, chain_encoding_all)
                rewards_argmax.append(dg_pred_argmax.detach().cpu().numpy())
                dg_pred_eval = reward_model_eval(X, S_sp_argmax, mask, chain_M, residue_idx, chain_encoding_all)
                rewards_eval.append(dg_pred_eval.detach().cpu().numpy())
            rewards.append(dg_pred_argmax.detach().cpu().numpy())
            total_kl = []
            assert len(last_x_list) == args.num_timesteps

            total_value_loss = []
            total_reward_loss = []
            for timestep in range(args.num_timesteps - 1):
                last_x = last_x_list[timestep]
                move_chance_t = move_chance_t_list[timestep]
                copy_flag = copy_flag_list[timestep]
                t_1 = ts[timestep]
                t_2 = ts[timestep+1]
                d_t = t_2 -t_1
                log_p_x0_out = new_fmif_model(X, last_x.detach(), mask, chain_M, residue_idx, chain_encoding_all, t=None)#t_1 * torch.ones((X.shape[0])).to(X.device))
                log_p_x0 = log_p_x0_out * 1.0
                log_p_x0[:, :, mu.MASK_TOKEN_INDEX] = -1e9
                log_p_x0 = F.log_softmax(log_p_x0 / args.temp, dim=-1)[:, :, :-1]
                
                with torch.no_grad():
                    log_p_x0_old_out = old_fmif_model(X, last_x, mask, chain_M, residue_idx, chain_encoding_all, t=None)
                    log_p_x0_old = log_p_x0_old_out.clone()
                    log_p_x0_old[:, :, mu.MASK_TOKEN_INDEX] = -1e9
                    log_p_x0_old = F.log_softmax(log_p_x0_old / args.temp, dim=-1)[:, :, :-1]
                    p_x0_old = log_p_x0_old.exp()

                if args.use_diff_value:
                    pred_logits_1 = log_p_x0_out.clone()
                    pred_logits_wo_mask = pred_logits_1.clone()
                    pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
                    pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)

                    pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = -1e9
                    pred_logits_1 = pred_logits_1 / 0.1 - torch.logsumexp(pred_logits_1 / 0.1, 
                                                                                    dim=-1, keepdim=True)
                    pred_logits_t = pred_logits_1.clone()
                    aatypes_t_1 = last_x
                    aatypes_t_1_argmax = last_x.argmax(dim=-1)
                    unmasked_indices = (aatypes_t_1_argmax != mu.MASK_TOKEN_INDEX)
                    pred_logits_1[unmasked_indices] = -1e9
                    pred_logits_1[unmasked_indices, aatypes_t_1_argmax[unmasked_indices]] = 0
                    # move_chance_t = 1.0 - t_1
                    move_chance_s = 1.0 - t_2
                    q_xs = pred_logits_1.exp() * d_t
                    q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
                    
                    q_xs = q_xs + 1e-8
                    _x = _sample_categorical_gradient(q_xs, 0.5)
                    copy_flag = 1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1)
                    aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)
                    aatypes_t_1 = aatypes_t_2

                    aatypes_t_1_argmax = aatypes_t_1[:, :, :-1].argmax(dim=-1) # to avoid the mask token
                    aatypes_t_1_argmax = F.one_hot(aatypes_t_1_argmax, num_classes=22).float()
                    used_x = aatypes_t_1 + (aatypes_t_1_argmax - aatypes_t_1).detach()
                else:
                    used_x = None
                p_x0 = log_p_x0.exp()
                p_x0_old = log_p_x0_old.exp()

                kl_div = copy_flag.detach() * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t.detach() # [bsz, seq_len, num_tokens]
                kl_div = kl_div * last_x[:, :, :-1].detach()
                mask_for_kl = mask*chain_M
                kl_div = (kl_div.sum(dim=-1) * mask_for_kl.detach()).sum(-1)
                
                tmp_loss = 0.0
                value_target = estimated_v[timestep]
                argx = x_argmaxs[timestep]
                
                us = log_p_x0_out.gather(-1, argx.unsqueeze(-1)).squeeze(-1)
                us_old = log_p_x0_old_out.gather(-1, argx.unsqueeze(-1)).squeeze(-1) 
                
                value_target = value_target.detach()
                value_loss = F.mse_loss(value_model(X, last_x.detach(),  mask, chain_M, residue_idx, chain_encoding_all), value_target,
                reduction='none')
                total_value_loss.append(value_loss.detach())
                tmp_loss = value_loss.mean()
                gae_o = estimated_v[timestep].detach()
                gae = gae_o
                gae = (gae_o- gae_o.mean()) / (gae_o.std() + 1e-5)
                gae = gae.detach()
                if timestep >= 0:
                    ratio = ((us-us_old)*mask).sum(dim=-1).exp()
                    ratioA = ratio * gae
                    ratioB = torch.clamp(ratio, 1-0.2, 1+0.2) * gae
                    if args.use_reward_clip:
                        reward_loss = - torch.min(ratioA, ratioB) / (mask.sum(dim=-1)+1e-8)  + args.entropy_scale*((us*mask).sum(dim=-1)) 
                    elif args.use_diff_value:
                        reward_loss =  - value_model_ema(X, used_x, mask, chain_M, residue_idx, chain_encoding_all)
                        if timestep == 0:
                            print(reward_loss.shape, gae.shape)
                    else:
                        reward_loss = - (us*mask).sum(dim=-1)*gae + args.entropy_scale*((us*mask).sum(dim=-1)) 
                    total_reward_loss.append(reward_loss.detach())
                    clip_ratio.append(  ((torch.abs(ratio.detach()-1.0).ge(0.2)).sum()/torch.numel(ratio.detach())).cpu().numpy()) 
                    tmp_loss = tmp_loss + reward_loss.mean()
                tmp_loss = tmp_loss 
                tmp_loss.backward()
                        
                total_kl.append(kl_div.detach())
                
            
            kl_loss = torch.stack(total_kl, dim=1).sum(1).mean()
            value_loss = torch.stack(total_value_loss, dim=1).sum(1).mean()
            reward_loss = torch.stack(total_reward_loss, dim=1).sum(1).mean()
            reward_losses.append(reward_loss.detach().cpu().numpy())
            record_value_losses.append(value_loss.detach().cpu().numpy())
            kl_losses.append(kl_loss.detach().cpu().numpy())

            S_sp_argmax = S_sp.argmax(dim=-1)
            true_false_sp = (S_sp_argmax == S).float()
            mask_for_loss = mask*chain_M
            train_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
            train_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.gradient_norm)
            optim_value.step()
            optim_value.zero_grad()
            value_model_ema.update_parameters(value_model)

            if (_step + 1) % args.accum_steps == 0 or _step == len(loader_train) - 1:
                norm = torch.nn.utils.clip_grad_norm_(new_fmif_model.parameters(), args.gradient_norm)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()
                if args.wandb_name != 'debug':
                    rewards = np.hstack(rewards)
                    rewards_argmax = np.hstack(rewards_argmax)
                    rewards_eval = np.hstack(rewards_eval)
                    tot_grad_norm = tot_grad_norm / (_step + 1) * args.accum_steps
                    wandb.log({"train/mean_reward": np.mean(rewards), "train/positive_reward_prop": np.mean(rewards>0), 
                        "train/mean_reward_argmax": np.mean(rewards_argmax), "train/positive_reward_argmax_prop": np.mean(rewards_argmax>0),
                        "train/mean_reward_eval": np.mean(rewards_eval), "train/positive_reward_eval_prop": np.mean(rewards_eval>0),
                        "train/mean_grad_norm": tot_grad_norm, 
                        "train/mean_value_losses": np.mean(record_value_losses),
                        'train/clip_ratio':np.mean(clip_ratio) ,
                        'train/kl_loss':np.mean(kl_losses),
                        'train/reward_loss': np.mean(reward_losses) }, step=wandb_step)
                    wandb_step +=  1
                    rewards_epoch.extend(rewards)
                    rewards_argmax_epoch.extend(rewards_argmax)
                    rewards_eval_epoch.extend(rewards_eval)
                    clip_ratio_epoch.extend(clip_ratio)
                    record_value_losses_epoch.extend(record_value_losses)
                    reward_losses_epoch.extend(reward_losses)
                    kl_losses_epoch.extend(kl_losses)

                    rewards = []
                    rewards_argmax = []
                    rewards_eval = []
                    clip_ratio = []
                    record_value_losses = []
                    reward_losses = []
                    kl_losses = []
                    tot_grad_norm = 0.0
        learning_rate_scheduler.step()
        learning_rate_scheduler_value.step()
        
        rewards_epoch.extend(rewards)
        rewards_argmax_epoch.extend(rewards_argmax)
        rewards_eval_epoch.extend(rewards_eval)
        clip_ratio_epoch.extend(clip_ratio)
        record_value_losses_epoch.extend(record_value_losses)
        reward_losses_epoch.extend(reward_losses)
        kl_losses_epoch.extend(kl_losses)

        reward_losses_epoch = np.hstack(reward_losses_epoch)
        rewards = np.hstack(rewards_epoch)
        rewards_argmax = np.hstack(rewards_argmax_epoch)
        rewards_eval = np.hstack(rewards_eval_epoch)
        clip_ratio_epoch = np.hstack(clip_ratio_epoch)
        record_value_losses_epoch = np.hstack(record_value_losses_epoch)
        kl_losses_epoch = np.hstack(kl_losses_epoch)

        tot_grad_norm = tot_grad_norm / (_step + 1) * args.accum_steps
        
        print("Epoch %d"%epoch_num, "Mean reward %f"%np.mean(rewards), np.max(rewards),np.min(rewards),np.std(rewards), "Positive reward prop %f"%np.mean(rewards>0), 
            "Mean reward argmax %f"%np.mean(rewards_argmax), "Positive reward argmax prop %f"%np.mean(rewards_argmax>0),
            "Mean reward eval %f"%np.mean(rewards_eval), "Positive reward eval prop %f"%np.mean(rewards_eval>0),
            "Mean grad norm %f"%tot_grad_norm,
            "Mean record value losses %f"%np.mean(record_value_losses_epoch), 
            "Mean reward losses %f"%np.mean(reward_losses_epoch),
            "Mean kl losses %f"%np.mean(kl_losses_epoch))
        print("clip ratio", np.mean(clip_ratio_epoch))
        
        train_sp_accuracy = train_sp_acc / train_sp_weights
        train_sp_accuracy_ = np.format_float_positional(np.float32(train_sp_accuracy), unique=False, precision=3)
        print("Train SP accuracy %s"%train_sp_accuracy_)
        if args.wandb_name != 'debug':
            wandb.log({"train/SP_accuracy": train_sp_accuracy}, step=wandb_step)
        with open(log_path, 'a') as f:
            f.write("Train SP accuracy %s"%train_sp_accuracy_ + "\n")

        if epoch_num % args.save_model_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_fmif_model.state_dict(), model_path)
            torch.save(value_model.state_dict(), os.path.join(save_path, f'value_model_{epoch_num}.ckpt'))
            print(f"Model saved at epoch {epoch_num}")

        if epoch_num % args.eval_every_n_epochs == 0:
            repeat_num = args.num_samples_per_eval
            new_fmif_model.eval()
            with torch.no_grad():
                rewards = []
                rewards_eval = []
                test_sp_acc, test_sp_weights = 0., 0.
                results_merge = []
                for _step, batch in tqdm(enumerate(loader_test)):
                    X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                    X = X.repeat(repeat_num, 1, 1, 1)
                    mask = mask.repeat(repeat_num, 1)
                    chain_M = chain_M.repeat(repeat_num, 1)
                    residue_idx = residue_idx.repeat(repeat_num, 1)
                    chain_encoding_all = chain_encoding_all.repeat(repeat_num, 1)
                    S_sp, _, _ = noise_interpolant.sample(new_fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
                    dg_pred = reward_model(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all)
                    rewards.append(dg_pred.detach().cpu().numpy())
                    dg_pred_eval = reward_model_eval(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all)
                    rewards_eval.append(dg_pred_eval.detach().cpu().numpy())
                    true_false_sp = (S_sp == S).float()
                    mask_for_loss = mask*chain_M
                    test_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                    test_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    results_list = cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, save_path, args.base_path)
                    results_merge.extend(results_list)

                rewards = np.hstack(rewards)
                rewards_eval = np.hstack(rewards_eval)
                print("Test Mean reward %f"%np.mean(rewards), "Test Positive reward prop %f"%np.mean(rewards>0),
                      "Test Mean reward eval %f"%np.mean(rewards_eval), "Test Positive reward eval prop %f"%np.mean(rewards_eval>0))
                if args.wandb_name != 'debug':
                    wandb.log({"test/mean_reward": np.mean(rewards), "test/positive_reward_prop": np.mean(rewards>0), 
                        "test/mean_reward_eval": np.mean(rewards_eval), "test/positive_reward_eval_prop": np.mean(rewards_eval>0)}, step=wandb_step)
                with open(log_path, 'a') as f:
                    f.write("Test Mean reward %f"%np.mean(rewards) + " Test Positive reward prop %f"%np.mean(rewards>0) + 
                        " Test Mean reward eval %f"%np.mean(rewards_eval) + " Test Positive reward eval prop %f"%np.mean(rewards_eval>0) + "\n")
                test_sp_accuracy = test_sp_acc / test_sp_weights
                test_sp_accuracy_ = np.format_float_positional(np.float32(test_sp_accuracy), unique=False, precision=3)
                results_merge = pd.concat(results_merge)
                avg_rmsd = results_merge['gen_true_bb_rmsd'].mean()
                mid_rmsd = results_merge['gen_true_bb_rmsd'].median()
                rmsd_rate = results_merge['gen_true_bb_rmsd'].apply(lambda x: 1 if x < 2 else 0).mean()
                avg_rmsd_ = np.format_float_positional(avg_rmsd, unique=False, precision=3)
                mid_rmsd_ = np.format_float_positional(mid_rmsd, unique=False, precision=3)
                rmsd_rate_ = np.format_float_positional(rmsd_rate, unique=False, precision=3)
                results_merge['rewards'] = rewards_eval
                results_merge['success'] = (results_merge['rewards'] > 0) & (results_merge['gen_true_bb_rmsd'] < 2)
                success_rate = results_merge['success'].mean()
                success_rate_ = np.format_float_positional(success_rate, unique=False, precision=3)
                print("Test success rate %s"%success_rate_, "Test SP accuracy %s"%test_sp_accuracy_, "Test gen_true_rmsd avg %s"%avg_rmsd_, "Test gen_true_rmsd mid%s"%mid_rmsd_, "Test gen_true_success_rate %s"%rmsd_rate_)
                if args.wandb_name != 'debug':
                    wandb.log({"test/success_rate": success_rate, "test/SP_accuracy": test_sp_accuracy, "test/gen_true_rmsd": avg_rmsd, "test/gen_true_rmsd_mid": mid_rmsd, "test/gen_true_success_rate": rmsd_rate}, step=wandb_step)
                with open(log_path, 'a') as f:
                    f.write("Test success rate %s"%success_rate_ + " Test SP accuracy %s"%test_sp_accuracy_ + " Test gen_true_rmsd %s"%avg_rmsd_ + " Test gen_true_rmsd mid %s"%mid_rmsd_ + " Test gen_true_success_rate %s"%rmsd_rate_ + "\n")
    
    if args.wandb_name != 'debug':
        wandb.finish()
        
    return



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--base_path", type=str, default="", help="base path for data and model")  # TODO
    argparser.add_argument("--num_epochs", type=int, default=201, help="number of epochs to train for") 
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=100, help="save model weights every n epochs")
    argparser.add_argument("--batch_size", type=int, default=64, help="number of sequences for one batch")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")   # 48
    argparser.add_argument("--dropout", type=float, default=0.0, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")  
    argparser.add_argument("--gradient_norm", type=float, default=1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
    argparser.add_argument("--wandb_name", type=str, default="debug", help="wandb run name")
    argparser.add_argument("--lr", type=float, default=3e-5)
    argparser.add_argument("--wd", type=float, default=1e-4)
    argparser.add_argument("--min_t", type=float, default=1e-2)
    argparser.add_argument("--schedule", type=str, default='linear') 
    argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    argparser.add_argument("--temp", type=float, default=0.1)
    argparser.add_argument("--noise", type=float, default=1.0)
    argparser.add_argument("--interpolant_type", type=str, default='masking')
    argparser.add_argument("--num_timesteps", type=int, default=50)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--eval_every_n_epochs", type=int, default=100)
    argparser.add_argument("--num_samples_per_eval", type=int, default=30)
    
    argparser.add_argument("--resample", type=int, default=0)
    argparser.add_argument("--accum_steps", type=int, default=1)
    argparser.add_argument("--truncate_steps", type=int, default=25)
    argparser.add_argument("--truncate_kl", type=str2bool, default=False)
    argparser.add_argument("--alpha", type=float, default=0.0003)
    argparser.add_argument("--gumbel_softmax_temp", type=float, default=0.5)
    argparser.add_argument("--reward_scale", type=float, default = 1.0)
    argparser.add_argument("--use_reward_clip", type=str2bool, default=False)
    argparser.add_argument("--entropy_scale", type=float, default=1e-3)
    
    argparser.add_argument("--use_diff_value", type=str2bool, default=False)
    args = argparser.parse_args()    
    print(args) 
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_for_outputs = os.path.join(args.base_path, 'protein_rewardbp')

    # initialize a log file
    if args.wandb_name == 'debug':
        print("Debug mode")
        save_path = os.path.join(path_for_outputs, args.wandb_name)
        os.makedirs(save_path, exist_ok=True)
        log_path = os.path.join(save_path, 'log.txt')
    else:
        run_name = f'alpha{args.alpha}_accum{args.accum_steps}_bsz{args.batch_size}_resample{args.resample}_entropy_scale{args.entropy_scale}_reward{args.reward_scale}_clip{args.gradient_norm}_seed{args.seed}_{args.wandb_name}_{curr_time}'
        save_path = os.path.join(path_for_outputs, run_name)
        os.makedirs(save_path, exist_ok=True)
       
        wandb.login()
        wandb.init(project='protein_glid2e', name=run_name, config=args, dir=save_path)
        log_path = os.path.join(save_path, 'log.txt')
    main(args, log_path, save_path)
