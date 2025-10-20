# Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design

The repository contains the code for the `GLID2E` method presented in the paper: *[GLID2E: Lightweight Policy-Based Fine-Tuning for
Discrete Diffusion in Biological Sequence Design]()(2025)*.
`GLID2E` is a fine-tuning method for reward optimization or alignment in discrete diffusion models, employing a reinforcement learning approach to fine-tune pretrained discrete diffusion models for generating functional biological sequence.


![img](main_fig1.png)

## Data
All data can be downloaded from this link(thanks to DRAKES):

https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=0

Save the downloaded file in `BASE_PATH`.

<!-- 
## Regulatory DNA Sequence Design

Our goal here is to optimize the activity of regulatory DNA sequences such that they drive gene expression in specific cell types, a critical task for cell and gene therapy. The detailed code and instructions are in `drakes_dna/`. 

## Protein Sequence Design: Optimizing Stability in Inverse Folding Model

Given a pretrained inverse folding model that generates sequences conditioned on the
backbone’s conformation (3D structure), our goal is to optimize the stability of these generated sequences.  The illustrative figure is as follows. The code and instructions are in `drakes_protein/`.

![img](main_fig2.png)


## Citation 

If you find this work useful in your research, please cite:

```
@article{wang2024finetuning,
  title={Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design},
  author={Chenyu Wang and Masatoshi Uehara and Yichun He and Amy Wang and Tommaso Biancalani and Avantika Lal and Tommi Jaakkola and Sergey Levine and Hanchen Wang and Aviv Regev},
  journal={arXiv preprint arXiv:2410.13643},
  year={2024}
}
```


|Method| Pred-Activity (median) ↑| ATAC-Acc ↑ (%)| 3-mer Corr ↑ |JASPAR Corr ↑ |Log-Lik (median) ↑|
|----------|----------|----------|---|---|--|
|Pretrained| 0.17(0.04)|1.5(0.2)| -0.061(0.034) |0.249(0.015) |-261(0.6)|
|DRAKES w/o KL| 6.44(0.04)| 82.5(2.8)| 0.307(0.001)| 0.557(0.015)| -281(0.6)|
|DRAKES| 5.61(0.07) |92.5(0.6) |0.887(0.002) |0.911(0.002) |-264(0.6)|
|PPO| 7.063| 92.5| 0.662| TODO| -253.40051| -->
