set -e

time python3 insert_gear_policy.py --out_dir results/results_rmdn_105_skip5_aug2_nodecay_subset60 --aug 4 --skipcount 5 --max_epoch 100 --subset 60 --gpu_id 0