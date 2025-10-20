#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python glide.py --alpha 0.1 --ab_K 0.5 --beta 0.001 --range_min 4.0 --range_max 0.0 --seed 0 --use_reward_shaping True --name glide & 
CUDA_VISIBLE_DEVICES=1 python glide.py --alpha 0.1 --ab_K 2.0 --beta 0.001 --range_min 4.0 --range_max 0.0 --seed 0 --use_reward_shaping True --name glide & 
CUDA_VISIBLE_DEVICES=2 python glide.py --alpha 0.1 --ab_K 3.0 --beta 0.001 --range_min 4.0 --range_max 0.0 --seed 0 --use_reward_shaping True --name glide & 
CUDA_VISIBLE_DEVICES=3 python glide.py --alpha 0.1 --ab_K 0.5 --beta 0.001 --range_min 4.0 --range_max 0.0 --seed 1 --use_reward_shaping True --name glide & 
CUDA_VISIBLE_DEVICES=4 python glide.py --alpha 0.1 --ab_K 2.0 --beta 0.001 --range_min 4.0 --range_max 0.0 --seed 1 --use_reward_shaping True --name glide & 
CUDA_VISIBLE_DEVICES=5 python glide.py --alpha 0.1 --ab_K 3.0 --beta 0.001 --range_min 4.0 --range_max 0.0 --seed 1 --use_reward_shaping True --name glide & 
CUDA_VISIBLE_DEVICES=6 python glide.py --alpha 0.1 --ab_K 1.0 --beta 0.001 --range_min 4.0 --range_max 0.0 --seed 0 --use_reward_shaping True --name glide & 

