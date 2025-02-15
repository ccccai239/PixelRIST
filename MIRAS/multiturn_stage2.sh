#! /bin/bash
cd /datas/MGM_/MIRAS
nohup deepspeed --master_port=28999 --include localhost:0
CUDA_VISIBLE_DEVICES=7 python3.12 
deepspeed --master_port=28999 --include localhost:1 
CUDA_VISIBLE_DEVICES=1 python3.12 train_stage2_v0.py \
  --model_name_or_path="/datas/MIRAS-7B_2" \
  --dataset_dir='/datas/multimodal_datasets/GRESDataset' \
  --log_base_dir='/datas/runs' \
  --vision_pretrained="/datas/sam_vit_h_4b8939.pth" \
  --dataset="multiturn||vqa" \
  --vqa_data="filtered_sharegpt4v" \
  --val_dataset="Multiturn|val" \
  --sample_rates="10,2" \
  --exp_name="mgmsa_stage2_5_noobj" \
  --refer_seg_data="refclef||refcoco||refcoco+||refcocog" \
  --batch_size=8 \
  --pretrain_mm_mlp_adapter="/datas/mm_7b_projector.bin" \
  --image_size_aux=768 \
  --lora_r=8 \
  --epochs=100 \
  --steps_per_epoch=250 \

 > output.log 2>&1 &
