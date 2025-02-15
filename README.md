### 生成模型文件
```bash
cd /datas/MGM_/runs/miras/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### 合并lora权重
```bash
CUDA_VISIBLE_DEVICES=0 python /datas/MGM_/merge_lora_weights_and_save_hf_model.py \
  --version="/datas/llava-v1.5-7b" \
  --weight="/datas/MGM_/runs/miras/pytorch_model.bin" \
  --save_path="MIRAS" \
  --lora_r=8 \
```

### 推理评估
```bash
deepspeed --master_port=20999 --include localhost:1 
CUDA_VISIBLE_DEVICES=1 python3.12 train_ds_7b.py \
  --model_name_or_path='/datas/MIRAS' \
  --vision_pretrained='/datas/sam_vit_h_4b8939.pth' \
  --dataset_dir='/datas/multimodal_datasets/Dataset' \ //数据集总路径
  --dataset='reason_seg||caption' \
  --lora_r=8 \
  --pretrain_mm_mlp_adapter="/datas/mm_7b_projector.bin" \
  --image_size_aux=768 \
  --val_dataset="refcocog|umd|val" \
  --eval_only \
```

### chat测试
```bash
CUDA_VISIBLE_DEVICES=3 python chat.py --version='/datas/MIRAS-7B'
```
