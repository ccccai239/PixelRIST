import argparse
import os
import sys
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from MGMSA.model.MGMSA import MgmsaForCausalLM
from MGMSA.model.mgm import conversation as conversation_lib
from MGMSA.model.mgm.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
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
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--image_size_aux", default=768, type=int)
    parser.add_argument("--image_grid", default=1, type=int)
    parser.add_argument("--image_global", default=False, type=bool)
    #parser.add_argument("--image_processor", default=None, type=str)
    parser.add_argument("--image_size_raw", default=None, type=list)
    parser.add_argument("--image_aspect_ratio", default='square', type=str)
    parser.add_argument("--image_grid_pinpoints", default=None, type=str)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    parser.add_argument("--mm_use_im_patch_token", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens(["[SEG]"])
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[1] 
    #args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[1]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    model = MgmsaForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    vision_tower_aux = model.get_model().get_vision_tower_aux()
    vision_tower_aux.to(dtype=torch_dtype)
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        vision_tower_aux = model.get_model().get_vision_tower_aux()
        model.model.vision_tower = None
        model.model.vision_tower_aux = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
        model.model.vision_tower_aux = vision_tower_aux.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    #args.image_processor = copy.deepcopy(vision_tower.image_processor)

    clip_image_processor=copy.deepcopy(vision_tower.image_processor)
    transform = ResizeLongestSide(args.image_size)
    model.config.image_grid = args.image_grid
    model.config.image_global = args.image_global
    model.config.image_aspect_ratio = args.image_aspect_ratio
    model.config.image_grid_pinpoints = args.image_grid_pinpoints
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    #model.config.mm_projector_lr = args.mm_projector_lr
    #args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
    args.image_size_raw = clip_image_processor.crop_size.copy()
    model.config.image_size_aux = args.image_size_aux
    clip_image_processor.crop_size['height'] = args.image_size_aux
    clip_image_processor.crop_size['width'] = args.image_size_aux
    clip_image_processor.size['shortest_edge'] = args.image_size_aux
    model.get_model().initialize_uni_modules(model.get_model().config)
    
    model.eval()

    # process multiturn
    text_output = ""
    for i in range(100):
        if i == 0:
            # init chat prompt
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            prompt = input("Please input your prompt: ")
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")

            # process image
            image_path = input("Please input the image path: ")
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))        
                continue

            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_=image_np[np.newaxis,:]
            #print(image_np.shape)
            original_size_list = [image_np.shape[:2]]
            image_clip = (
                clip_image_processor.preprocess(image_, return_tensors="pt")[
                    "pixel_values"
                ][0]
            )
            image_aux =image_clip.clone()
            raw_shape = [args.image_size_raw['height'] * args.image_grid,
                            args.image_size_raw['width'] * args.image_grid]
            if len(image_clip)==3:
                    image_clip = torch.nn.functional.interpolate(image_clip[None], 
                                                                size=raw_shape, 
                                                                mode='bilinear', 
                                                                align_corners=False)[0]
            else:
                    image_clip = torch.nn.functional.interpolate(image_clip, 
                                                                size=raw_shape, 
                                                                mode='bilinear', 
                                                                align_corners=False)
            image_clip = image_clip[None].to(dtype=model.dtype, device='cuda', non_blocking=True)
            image_aux = image_aux[None].to(dtype=model.dtype, device='cuda', non_blocking=True) if len(image_aux)>0 else None

            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
                image_aux = image_aux.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
                image_aux = image_aux.half()
            else:
                image_clip = image_clip.float()
                image_aux = image_aux.float()
            
            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()
        else:
            # process chat prompt
            conv.messages.pop()
            conv.append_message(conv.roles[1], text_output)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")

        prompt = conv.get_prompt()

        print(prompt)

        # process image
        # image_path = input("Please input the image path: ")
        # if not os.path.exists(image_path):
        #     print("File not found in {}".format(image_path))        
        #     continue

        # image_np = cv2.imread(image_path)
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # image_=image_np[np.newaxis,:]
        # #print(image_np.shape)
        # original_size_list = [image_np.shape[:2]]
        # image_clip = (
        #     clip_image_processor.preprocess(image_, return_tensors="pt")[
        #         "pixel_values"
        #     ][0]
        # )
        # image_aux =image_clip.clone()
        # raw_shape = [args.image_size_raw['height'] * args.image_grid,
        #                 args.image_size_raw['width'] * args.image_grid]
        # if len(image_clip)==3:
        #         image_clip = torch.nn.functional.interpolate(image_clip[None], 
        #                                                     size=raw_shape, 
        #                                                     mode='bilinear', 
        #                                                     align_corners=False)[0]
        # else:
        #         image_clip = torch.nn.functional.interpolate(image_clip, 
        #                                                     size=raw_shape, 
        #                                                     mode='bilinear', 
        #                                                     align_corners=False)
        # image_clip = image_clip[None].to(dtype=model.dtype, device='cuda', non_blocking=True)
        # image_aux = image_aux[None].to(dtype=model.dtype, device='cuda', non_blocking=True) if len(image_aux)>0 else None

        # if args.precision == "bf16":
        #     image_clip = image_clip.bfloat16()
        #     image_aux = image_aux.bfloat16()
        # elif args.precision == "fp16":
        #     image_clip = image_clip.half()
        #     image_aux = image_aux.half()
        # else:
        #     image_clip = image_clip.float()
        #     image_aux = image_aux.float()
        
        # image = transform.apply_image(image_np)
        # resize_list = [image.shape[:2]]

        # image = (
        #     preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        #     .unsqueeze(0)
        #     .cuda()
        # )
        # if args.precision == "bf16":
        #     image = image.bfloat16()
        # elif args.precision == "fp16":
        #     image = image.half()
        # else:
        #     image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        #input_ids = input_ids.unsqueeze(0).cuda()
        #print("input_ids: ", input_ids)
        output_ids, pred_masks = model.evaluate(
            image_clip,
            image_aux,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=128,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))

        prompt = input("Please input your prompt: ")

if __name__ == "__main__":
    main(sys.argv[1:])

        

