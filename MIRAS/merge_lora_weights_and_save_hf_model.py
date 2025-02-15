
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from MGMSA.model.MGMSA import MgmsaForCausalLM
from MGMSA.utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
#from MGM_.MGMSA.utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="/root/autodl-tmp/MGM-7B"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="/datas/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument(
        "--vision-tower", default="/datas/huggingface/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--vision-tower-aux", default="/datas/CLIP-convnext_large_d_320-laion2B-s29B-b131K-ft-soup", type=str)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--image_size_aux", default=768, type=int)
    parser.add_argument("--image_grid", default=1, type=int)
    parser.add_argument("--image_global", default=False, type=bool)
    parser.add_argument("--image_processor", default=None, type=str)
    parser.add_argument("--image_size_raw", default=None, type=list)
    parser.add_argument("--image_aspect_ratio", default='square', type=str)
    parser.add_argument("--image_grid_pinpoints", default=None, type=str)
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./mgmsa_model", type=str, required=True)
    parser.add_argument("--mm_projector_lr", default=None, type=float)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    parser.add_argument("--mm_use_im_patch_token", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_feature", default="patch", type=str)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--optimize_vision_tower",default=False, type=bool)
    parser.add_argument("--optimize_vision_tower_aux",default=False, type=bool)
    #parser.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    #parser.add_argument("--mm_use_im_patch_token", action="store_true", default=True)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="/datas/caidexian/mm_7b_projector.bin", type=str)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens(["[SEG]"])
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[1] 
    #num_added_tokens = tokenizer.add_tokens(["[SEG]","[OBJ]"])
    #args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[1] 
    #args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[1]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
        #print(len(tokenizer))

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        #"obj_token_idx": args.obj_token_idx,
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
    model = MgmsaForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    #model.enable_input_require_grads()
    #model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    vision_tower_aux = model.get_model().get_vision_tower_aux()
    vision_tower_aux.to(dtype=torch_dtype, device=args.local_rank)
    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model.get_model().config, tokenizer=tokenizer)
    model.get_model().initialize_uni_modules(model.get_model().config)
    model.get_model().initialize_mgmsa_modules(model.get_model().config)

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
    #print(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=True,assign=True)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
        if "vision_tower_aux" not in k:
            state_dict[k] = v
        #if "vlm_uni_aux_projector" not in k:
        #    state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
