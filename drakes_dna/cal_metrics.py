import os

import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import dataloader_gosai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import oracle
from scipy.stats import pearsonr
import torch
from tqdm import tqdm
import diffusion_gosai_cfg
from utils import set_seed
set_seed(0, use_cuda=True)
plt.rcParams['figure.dpi'] = 200


base_path = '' # TODO
save_path = '' # TODO
# our model

import argparse
import glob
import re
def find_dirs_by_regex(base_path, ckpt_pattern):
    matched_dirs = []
    for root, dirs, files in os.walk(base_path):
        for _dir in dirs:
            matched_dirs.append(os.path.join(root, _dir))
    matched_dirs.sort()
    return matched_dirs
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='') # TODO
parser.add_argument('--search_ckpt', action='store_true', default=True, help='Enable ckpt search mode')
parser.add_argument('--ckpt_pattern', type=str, default='mdlm/glide/alpha0.1_beta0.001_range\(4.0\,\ 0.0\)_abK*_accum4_bsz8_shapingTrue_temp1.0_seed*', 
help='Glob pattern for ckpt directories, e.g. "mdlm/*"')
args = parser.parse_args()

if args.search_ckpt and args.ckpt_pattern is not None:
    # Search for all matching ckpts
    ckpt_results = find_dirs_by_regex(base_path, args.ckpt_pattern)
    print(f"Found {len(ckpt_results)} checkpoints matching pattern {args.ckpt_pattern}")
    highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999, highexp_set_sp_clss_999, highexp_preds_999, highexp_seqs_999 = oracle.cal_highexp_kmers(return_clss=True)
    
    for ckpt_dir in ckpt_results:
        print(f"Evaluating {ckpt_dir}")
        seed = re.search(r'seed(\d+)', ckpt_dir).group(1)
        abk = re.search(r'abK(\d+\.\d+)', ckpt_dir).group(1)
        print(seed, abk)
        os.makedirs(os.path.join(save_path, f'seed{seed}_abK{abk}'), exist_ok=True)
        OUT_PATH = os.path.join(os.path.join(save_path, f'seed{seed}_abK{abk}'), f'ckpt_model.csv')
        for i in range(49,1000,50):  
            try:
                csv_results = [i]
                CKPT_PATH = os.path.join(ckpt_dir,f'model_{i}.ckpt')
                print(CKPT_PATH)
                NUM_SAMPLE_BATCHES = 50
                NUM_SAMPLES_PER_BATCH = 64

                # reinitialize Hydra
                GlobalHydra.instance().clear()
                initialize(config_path="configs_gosai", job_name="load_model")

                cfg = compose(config_name="config_gosai.yaml")

                old_path = os.path.join(base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
                old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(old_path, config=cfg)
                old_model.eval()
                cfg.eval.checkpoint_path = CKPT_PATH

                model = diffusion_gosai_update.Diffusion(cfg, eval=False).cuda()
                model.load_state_dict(torch.load(cfg.eval.checkpoint_path))
                model.eval()

                all_detoeknized_samples = []
                all_raw_samples = []
                for _ in tqdm(range(NUM_SAMPLE_BATCHES)):
                    samples = model._sample(eval_sp_size=NUM_SAMPLES_PER_BATCH)
                    all_raw_samples.append(samples)
                    detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
                    all_detoeknized_samples.extend(detokenized_samples)
                all_raw_samples = torch.concat(all_raw_samples)
                
                # save all_raw_samples
                torch.save(all_raw_samples, os.path.join(save_path, f'seed{seed}_abK{abk}', f'all_raw_samples_{i}.ckpt'))
                # save all_detoeknized_samples
                torch.save(all_detoeknized_samples, os.path.join(save_path, f'seed{seed}_abK{abk}', f'all_detoeknized_samples_{i}.ckpt'))
                
                
                from grelu.interpret.motifs import scan_sequences
                import grelu
                
                model_logl = old_model.get_likelihood(all_raw_samples, num_steps=128, n_samples=1)

                reward_model_bs = oracle.get_gosai_oracle(mode='train')
                reward_model_bs.eval()

                compare = np.concatenate((
                                        model_logl.detach().cpu().numpy(),
                                        ), axis= 0)
                print(f"epoch {i}","likelihood eval median: ",np.median(compare.reshape(-1, NUM_SAMPLE_BATCHES * NUM_SAMPLES_PER_BATCH), axis=-1))
                csv_results.append(np.median(compare.reshape(-1, NUM_SAMPLE_BATCHES * NUM_SAMPLES_PER_BATCH), axis=-1))
                highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999, highexp_set_sp_clss_999, highexp_preds_999, highexp_seqs_999 = oracle.cal_highexp_kmers(return_clss=True)

                generated_preds = oracle.cal_gosai_pred_new(all_detoeknized_samples, mode='eval')

                compare = np.concatenate((
                                            generated_preds[:,0],
                                        ), axis= 0)
                print(f"epoch {i}","Pred-Activity based on Eval Oracle: ",np.median(compare.reshape(-1, NUM_SAMPLE_BATCHES * NUM_SAMPLES_PER_BATCH), axis=-1))
                csv_results.append(np.median(compare.reshape(-1, NUM_SAMPLE_BATCHES * NUM_SAMPLES_PER_BATCH), axis=-1))
                generated_preds_atac = oracle.cal_atac_pred_new(all_detoeknized_samples)

                print( f"epoch {i}",  "ATCC", (generated_preds_atac[:,1]>0.5).sum()/NUM_SAMPLE_BATCHES / NUM_SAMPLES_PER_BATCH)
                csv_results.append((generated_preds_atac[:,1]>0.5).sum()/NUM_SAMPLE_BATCHES / NUM_SAMPLES_PER_BATCH)
                def compare_kmer(kmer1, kmer2, n_sp1, n_sp2, title):
                    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
                    counts = np.zeros((len(kmer_set), 2))
                    for _i, kmer in enumerate(kmer_set):
                        if kmer in kmer1:
                            counts[_i][1] = kmer1[kmer] * n_sp2 / n_sp1
                        if kmer in kmer2:
                            counts[_i][0] = kmer2[kmer]
                    print(f"epoch {i}","3-mer ",pearsonr(counts[:, 0], counts[:, 1]))
                    csv_results.append(pearsonr(counts[:, 0], counts[:, 1]))
                generated_kmer = oracle.count_kmers(all_detoeknized_samples)
                compare_kmer(highexp_kmers_999, generated_kmer, n_highexp_kmers_999, len(all_detoeknized_samples), title=r"Finetuned")

                with open(OUT_PATH, 'a') as f:
                    f.write(','.join(map(str, csv_results)) + '\n')
                # ... (rest of your evaluation code)
            except Exception as e:
                print(f"Error evaluating {model_ckpt_path}: {e}")
    # Exit after search mode
    exit(0)