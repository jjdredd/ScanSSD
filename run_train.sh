#! /bin/bash

python3 train.py 			--dataset GTDB \
	--dataset_root ./made_datasets_v1/Train/ \
	--batch_size 8 \
	--num_workers 3 \
	--exp_name IOU512_iter1 \
	--model_type 512 \
	--training_data data_file.txt \
	--cfg hboxes512 \
	--loss_fun ce \
	--kernel 1 5 \
	--padding 0 2 \
	--neg_mining True \
	--pos_thresh 0.75 \
