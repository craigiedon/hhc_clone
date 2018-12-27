python3 goal_score_model.py --batch_size 100 \
			    --data_file_pattern 'trial{}_l_forearm.avi' \
			    --out_dir 'results/result_l_forearm' \
			    --max_epoch 200 \
			    --test_split 0.2

python3 goal_score_model.py --batch_size 100 \
			    --data_file_pattern 'trial{}_r_forearm.avi' \
			    --out_dir 'results/result_r_forearm' \
			    --max_epoch 200 \
			    --test_split 0.2

python3 goal_score_model.py --batch_size 100 \
			    --data_file_pattern 'trial{}_kinect2_qhd.avi' \
			    --out_dir 'results/result_kinect2' \
			    --max_epoch 200 \
			    --test_split 0.2
