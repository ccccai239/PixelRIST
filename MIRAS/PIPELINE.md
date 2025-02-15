### 生成模型文件
```bash
cd /datas/caidexian/MGM_/runs/mgmsa-7b_4_la1.5_v3/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### 合并lora权重
```bash
CUDA_VISIBLE_DEVICES=5 python /datas/caidexian/MGM_/merge_lora_weights_and_save_hf_model.py \
  --version="/datas/multimodal_LLMs/llava-v1.5-7b" \
  --weight="/datas/caidexian/MGM_/runs/mgmsa-7b_4_la1.5_woobj/pytorch_model.bin" \
  --save_path="MGMSA-7B_2" \
  --lora_r=8 \
```
CUDA_VISIBLE_DEVICES=2 python /datas/caidexian/MGM_/merge_lora_weights_and_save_hf_model.py \
  --version="/datas/huggingface/llava-v1.6-vicuna-7b-hf" \
  --weight="/datas/caidexian/MGM_/runs/mgmsa1.6-7b_6_woobj_vicuna/pytorch_model.bin" \
  --save_path="MGMSA1.6-7B" \
  --lora_r=8 \
### 推理评估
```bash
deepspeed --master_port=20999 --include localhost:1 
CUDA_VISIBLE_DEVICES=1 python3.12 train_ds_7b.py \
  --model_name_or_path='/datas/caidexian/MGMSA1.6-7B' \
  --vision_pretrained='/datas/caidexian/sam_vit_h_4b8939.pth' \
  --dataset_dir='/datas/multimodal_datasets/GRESDataset' \
  --dataset='reason_seg||caption' \
  --lora_r=8 \
  --pretrain_mm_mlp_adapter="/datas/caidexian/mm_7b_projector.bin" \
  --image_size_aux=768 \
  --val_dataset="refcocog|umd|val" \
  --eval_only \
```

### chat测试
```bash
CUDA_VISIBLE_DEVICES=3 python chat.py --version='/datas/caidexian/MGMSA-7B'
```

### debug
```bash
            "args":[
                "--model_name_or_path","/datas/huggingface/llava-v1.6-vicuna-7b",
                "--dataset_dir","/datas/multimodal_datasets/GRESDataset",
                "--log_base_dir","/datas/caidexian/MGM_/finetune/runs",
                "--vision_pretrained","/datas/caidexian/sam_vit_h_4b8939.pth" ,
                "--vqa_data","filtered_sharegpt4v",
                "--exp_name","mgmsa-ft-7b",
                "--refer_seg_data","refclef||refcoco||refcoco+||refcocog",
                "--batch_size","2",
                "--pretrain_mm_mlp_adapter","/datas/caidexian/mm_7b_projector.bin",
                "--image_size_aux","768",
                "--lora_r","8",
                "--epochs","150",
                "--steps_per_epoch","200",
            ]
```