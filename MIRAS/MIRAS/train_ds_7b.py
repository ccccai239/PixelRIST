#!/usr/bin/env python3.10
import argparse
import os
import shutil
import sys
import time
from functools import partial
import copy
from nltk.translate.bleu_score import sentence_bleu 
#from rouge import Rouge
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

#from model.MGMSA_ import MgmsaForCausalLM
from MGMSA_ import MgmsaForCausalLM
#from model.mgm import conversation as conversation_lib
from mgm import conversation as conversation_lib
from MIRAS.utils.dataset import HybridDataset, ValDataset, collate_fn
from MIRAS.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

def parse_args(args):
    parser = argparse.ArgumentParser(description="MGM+SAM training script")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--model_name_or_path", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="/datas/huggingface/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--vision-tower-aux", default="/datas/CLIP-convnext_large_d_320-laion2B-s29B-b131K-ft-soup", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg||caption||vg||multisteps", type=str
    )#sem_seg||refer_seg||vqa||reason_seg||caption
    parser.add_argument("--sample_rates", default="9,6,3,2,2", type=str)
    parser.add_argument("--multihop_reason_seg_data", default="/datas/myfiles/multi_hop_reason_seg_v0.jsonl", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--vg_data", default="VG|train", type=str)
    parser.add_argument("--allava_data", default="ALLAVA/ALLaVA-Instruct-VFLAN-4V.json", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--caption_data", default="coco/annotations", type=str)
    parser.add_argument("--multiturn_data", default="/datas/myfiles/processed_multiturn_tot_01.json", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str) #refcocog|umd|val / refclef/refcoco/refcoco+|unc|val / ReasonSeg|val
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="MIRAS", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--image_size_aux", default=320, type=int)
    parser.add_argument("--image_grid", default=1, type=int)
    parser.add_argument("--image_global", default=False, type=bool)
    parser.add_argument("--image_processor", default=None, type=str)
    parser.add_argument("--image_size_raw", default=None, type=list)
    parser.add_argument("--image_aspect_ratio", default='square', type=str)
    parser.add_argument("--image_grid_pinpoints", default=None, type=str)
    parser.add_argument("--optimize_vision_tower",default=False, type=bool)
    parser.add_argument("--optimize_vision_tower_aux",default=False, type=bool)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    #parser.add_argument("--tune_mm_mlp_adapter", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--mm_projector_lr", default=None, type=float)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    parser.add_argument("--mm_use_im_patch_token", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_feature", default="patch", type=str)
    parser.add_argument("--freeze_backbone", default=False, type=bool)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    print(tokenizer("[SEG]",add_special_tokens=False))
    args.seg_token_idx = tokenizer("[SEG]",add_special_tokens=False).input_ids[1] #input_ids[0]
    print("seg_token_idx: ", args.seg_token_idx)
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "vision_tower_aux": args.vision_tower_aux,
        "use_mm_start_end": args.use_mm_start_end,
        "optimize_vision_tower" :args.optimize_vision_tower,
        "optimize_vision_tower_aux" :args.optimize_vision_tower_aux,
        "pretrain_mm_mlp_adapter": args.pretrain_mm_mlp_adapter,
        "image_size_aux": args.image_size_aux,
        "image_grid": args.image_grid,
        "image_global": args.image_global,
        "mm_use_im_start_end": args.use_mm_start_end,
        "mm_projector_lr": args.mm_projector_lr,
        "image_aspect_ratio": args.image_aspect_ratio,
        "mm_use_im_patch_token": args.mm_use_im_patch_token,
        "mm_vision_select_feature": args.mm_vision_select_feature,
        
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model =MgmsaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype, low_cpu_mem_usage=False, **model_args
    )

    if args.freeze_backbone:
        model.model.requires_grad_(False)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    
    vision_tower_aux = model.get_model().get_vision_tower_aux()
    vision_tower_aux.to(dtype=torch_dtype, device=args.local_rank)
    
    # if model_args.tune_mm_mlp_adapter:
    #     model.requires_grad_(False)
    #     for p in model.get_model().mm_projector.parameters():
    #         p.requires_grad = True

    args.image_processor = copy.deepcopy(vision_tower.image_processor)
    model.config.image_grid = args.image_grid
    model.config.image_global = args.image_global
    model.config.image_aspect_ratio = args.image_aspect_ratio
    model.config.image_grid_pinpoints = args.image_grid_pinpoints
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    model.config.mm_projector_lr = args.mm_projector_lr
    #args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model.get_model().config, tokenizer=tokenizer)

    args.image_size_raw = args.image_processor.crop_size.copy()
    model_args['image_size_aux'] = args.image_size_aux
    args.image_processor.crop_size['height'] = args.image_size_aux
    args.image_processor.crop_size['width'] = args.image_size_aux
    args.image_processor.size['shortest_edge'] = args.image_size_aux
    model.get_model().initialize_uni_modules(model.get_model().config)

    if not args.eval_only:
        model.get_model().initialize_mgmsa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in vision_tower_aux.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    
    if args.optimize_vision_tower:
        print('Optimize last 1/2 layers in vision tower')
        total_num = len(vision_tower.vision_tower.vision_model.encoder.layers)
        for _idx in range(total_num//2, total_num):
            vision_tower.vision_tower.vision_model.encoder.layers[_idx].requires_grad_(True)

    if args.optimize_vision_tower_aux:
        print('Optimize last layer of each block in vision tower aux')
        for _idx in range(len(vision_tower_aux.vision_stages)):
            vision_tower_aux.vision_stages[_idx].blocks[-1].requires_grad_(True)
    

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "vision_tower_aux",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    #make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.image_processor,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        allava_data=args.allava_data,
        cap_data=args.caption_data,
        reason_seg_data=args.reason_seg_data,
        vg_data=args.vg_data,
        multiturn_data=args.multiturn_data,
        explanatory=args.explanatory,
        args=args,
    )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.image_processor,
            args.val_dataset,
            args.image_size,
            args
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0
    best_score_true, cur_ciou_true = 0.06, 0.06

    if args.eval_only:
        giou, ciou,giou_true,ciou_true = validate(val_loader, model_engine, 0, writer, args,tokenizer)
        #mIoU, results_str = validate2(val_loader, model_engine, 0, writer, args)
        exit()
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou,giou_true,ciou_true = validate(val_loader, model_engine, epoch, writer, args,tokenizer)
            mIoU, results_str = validate2(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou
            #掩码为1的iou
            is_best_true = giou_true > best_score_true
            best_score_true = max(giou_true, best_score_true)
            cur_ciou_true = ciou_true if is_best_true else cur_ciou_true


        if args.no_eval or is_best or is_best_true:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(train_loader, model, epoch, scheduler, writer, train_iter, args):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    #layer_weights_=AverageMeter("LWeights",":4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
                input_dict["images_aux"] = input_dict["images_aux"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                input_dict["images_aux"] = input_dict["images_aux"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
                input_dict["images_aux"] = input_dict["images_aux"].float()
            
            input_dict["images_clip"] = input_dict["images_clip"].squeeze(0)
            #print(input_dict['image_paths'])
            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            #layer_weights=output_dict['layer_weights']

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            if mask_bce_loss == 0.0:
                mask_bce_losses.update(0.0, input_dict["images"].size(0))
            else:
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            if mask_dice_loss == 0.0:
                mask_dice_losses.update(0.0, input_dict["images"].size(0))
            else:
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            if mask_loss ==0.0:
                mask_losses.update(0.0, input_dict["images"].size(0))
            else:
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            #layer_weights_.update(layer_weights)
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                #layer_weights_.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            #layer_weights_.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args, tokenizer):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()


    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
            input_dict["images_aux"] = input_dict["images_aux"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            input_dict["images_aux"] = input_dict["images_aux"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
            input_dict["images_aux"] = input_dict["images_aux"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # mask decoder
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class.mean()
    ciou_true = iou_class[1]
    giou = acc_iou_meter.avg.mean()
    giou_true = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/giou_true", giou_true, epoch)
        writer.add_scalar("val/ciou_true", ciou_true, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        print("giou_true: {:.4f}, ciou_true: {:.4f}".format(giou_true, ciou_true))

    return giou, ciou, giou_true, ciou_true

def validate2(val_loader, model_engine, epoch, writer, args):
    # 初始化计量器，用于记录交集、并集和IoU的精度
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # 设定不同的IoU评估阈值
    eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]  # 不同的IoU阈值
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)  # 统计不同阈值下的正确预测次数
    seg_total = 0  # 总评估次数

    model_engine.eval()  # 设置模型为评估模式

    # 遍历验证数据集
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()  # 清理显存

        input_dict = dict_to_cuda(input_dict)  # 将输入数据移动到GPU
        # 根据不同精度处理图像
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
            #input_dict["images_aux"] = input_dict["images_aux"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            #input_dict["images_aux"] = input_dict["images_aux"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
            #input_dict["images_aux"] = input_dict["images_aux"].float()

        with torch.no_grad():
            # 模型前向计算，获取预测输出
            output_dict = model_engine(**input_dict)

        # 获取预测掩码和真实掩码
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()  # 二值化预测掩码
        assert len(pred_masks) == 1

        # 初始化交集、并集
        cum_I, cum_U = 0, 0  # 用于整体IoU计算
        mean_IoU = []  # 用于记录每个预测的IoU

        # 逐元素计算交集、并集和IoU
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection = intersection_i.cpu().numpy()
            union = union_i.cpu().numpy()
            
            # 确保 this_iou 是标量
            this_iou = (intersection / (union + 1e-5)).mean()  # 取 IoU 的平均值作为标量

            mean_IoU.append(this_iou)  # 记录每个 IoU

            # 更新累积交集和并集
            cum_I += intersection.sum()  # 确保累加的是标量
            cum_U += union.sum()

            # 根据不同IoU阈值更新精度
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)  # 比较时确保 this_iou 是标量
            seg_total += 1

        # 更新计量器数据
        intersection_meter.update(cum_I)
        union_meter.update(cum_U)
        acc_iou_meter.update(np.mean(mean_IoU), n=masks_list.shape[0])

    # 归约计量器数据
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    # 计算最终IoU和精度
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)  # 计算平均IoU
    # 打印精度
    print('Final results:')
    print(f'Mean IoU is {mIoU * 100:.2f}%')
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += f'    precision@{eval_seg_iou_list[n_eval_iou]} = {seg_correct[n_eval_iou] * 100. / seg_total:.2f}%\n'
    results_str += f'    overall IoU = {cum_I * 100. / cum_U:.2f}%\n'
    print(results_str)

    # 返回评估结果
    return mIoU, results_str


def validate(val_loader, model_engine, epoch, writer, args, tokenizer):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()


    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
            input_dict["images_aux"] = input_dict["images_aux"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            input_dict["images_aux"] = input_dict["images_aux"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
            input_dict["images_aux"] = input_dict["images_aux"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # mask decoder
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class.mean()
    ciou_true = iou_class[1]
    giou = acc_iou_meter.avg.mean()
    giou_true = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/giou_true", giou_true, epoch)
        writer.add_scalar("val/ciou_true", ciou_true, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        print("giou_true: {:.4f}, ciou_true: {:.4f}".format(giou_true, ciou_true))

    return giou, ciou, giou_true, ciou_true

if __name__ == "__main__":
    main(sys.argv[1:])

    
