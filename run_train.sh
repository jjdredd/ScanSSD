#! /bin/bash

python3 train.py 			--dataset GTDB \
	--dataset_root ./made_datasets_v1/Test/ \
	--cuda False \
	--visdom False \
	--batch_size 16 \
	--num_workers 4 \
	--exp_name IOU512_iter1 \
	--model_type 512 \
	--training_data data_file.txt \
	--cfg hboxes512 \
	--loss_fun ce \
	--kernel 1 5 \
	--padding 0 2 \
	--neg_mining True \
	--pos_thresh 0.75 \
