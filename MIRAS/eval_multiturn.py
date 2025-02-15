import argparse
import os
import sys
import copy
import cv2
import numpy as np
import json
import random
import torch
import torch.nn.functional as F
from skimage.feature import canny
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from MGMSA.model.MGMSA import MgmsaForCausalLM
from MGMSA.model.mgm import conversation as conversation_lib
from MGMSA.model.mgm.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source

def parse_args(args):
    parser = argparse.ArgumentParser(description="MGMSA chat")
    parser.add_argument("--version", default="/datas/caidexian/MGMSA-stage2-NOOBJ-7B")
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
    parser.add_argument("--vision-tower-aux", default="/datas/caidexian/CLIP-convnext_large_d_320-laion2B-s29B-b131K-ft-soup", type=str)
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

def get_mask_from_json_target(json_path, target_label):
    """
    从JSON文件中读取标注信息，并生成指定标签的掩码图像。
    
    参数：
    json_path: str - JSON文件的路径
    target_label: str - 指定的标签名称，如“The word 'ALOHA'”
    
    返回：
    mask: np.ndarray - 生成的掩码图像
    """
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    height, width = anno['imageHeight'], anno['imageWidth']

    # 初始化一个空白的mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 遍历所有标注信息，处理指定标签的部分
    for i in inform:
        label_id = i["label"]
        points = i["points"]

        # 只处理与指定标签名称匹配的标注
        if target_label.lower() in label_id.lower():
            label_value = 1  # 设置为1表示目标区域
            # 将多边形绘制到mask中
            cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
            cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask

def calculate_cIoU(pred_mask, gt_mask):
    """
    计算生成的mask和ground truth mask的cIoU指标
    
    参数：
    pred_mask: 预测的二进制mask，大小为[H, W]
    gt_mask: ground truth的二进制mask，大小为[H, W]
    
    返回：
    cIoU值
    """
    # 计算交集
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    # 计算并集
    union = np.logical_or(pred_mask, gt_mask).sum()

    # 如果并集为0，返回cIoU为1（假设此时预测和ground truth都是空白）
    if union == 0:
        return 1.0
    
    # 计算IoU
    iou = intersection / union
    return iou

def calculate_bIoU(pred_mask, gt_mask):
     # 使用 Canny 边缘检测提取边界
    #pred_boundary = canny(pred_mask.astype(np.uint8))
    gt_mask_2d = gt_mask.squeeze(0)
    #print(gt_mask_2d)
    pred_boundary = canny(pred_mask, low_threshold=0.1, high_threshold=0.3)
    gt_boundary = canny(gt_mask_2d, low_threshold=0.1, high_threshold=0.3)
    #print(gt_boundary)

    # 计算边界的交集和并集
    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()

    # 计算 B-IoU
    b_iou = intersection / (union + 1e-5)  # 加上小数防止除零
    return b_iou

def calculate_pixel_accuracy_in_roi(pred_mask, gt_mask):
    # 确保 gt_mask 是二维的，通过 squeeze 去除多余的维度
    gt_mask_2d = gt_mask.squeeze(0)  # 如果 gt_mask 是三维 [1, height, width]，转为 [height, width]
    
    # 假设 gt_mask 中值为 1 的区域是感兴趣区域（ROI）
    roi_mask = (gt_mask_2d == 1)
    
    # 计算 ROI 区域内的正确分类像素
    correct_pixels = (pred_mask[roi_mask] == gt_mask_2d[roi_mask]).sum()

    # 计算 ROI 区域内的总像素数
    total_pixels = roi_mask.sum()

    if total_pixels == 0:
        return 0.0  # 防止除以 0 的情况
    
    # 计算 Pixel Accuracy in ROI
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy

def process_multiturn(conv, args, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, clip_image_processor, model, tokenizer_image_token, IMAGE_TOKEN_INDEX, tokenizer, transform, preprocess,
                      img_path,user_queries):
    history = []
    #pred_masks_list = []
    pred_mask = None
    text_output = ""
    for i in range(10):
        if i == 0:
            # init chat prompt
            #conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            prompt = user_queries[0] #input("Please input your prompt: ")
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if args.use_mm_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")

            # process image
            image_path = img_path  #input("Please input the image path: ")
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))        
                continue

            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_ = image_np[np.newaxis, :]
            original_size_list = [image_np.shape[:2]]
            image_clip = clip_image_processor.preprocess(image_, return_tensors="pt")["pixel_values"][0]
            image_aux = image_clip.clone()
            raw_shape = [args.image_size_raw['height'] * args.image_grid, args.image_size_raw['width'] * args.image_grid]
            image_clip = torch.nn.functional.interpolate(image_clip[None], size=raw_shape, mode='bilinear', align_corners=False)[0]

            image_clip = image_clip[None].to(dtype=model.dtype, device='cuda', non_blocking=True)
            image_aux = image_aux[None].to(dtype=model.dtype, device='cuda', non_blocking=True) if len(image_aux) > 0 else None

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

            image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()

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
            # Get the next prompt from user_queries
            if i < len(user_queries):
                prompt = user_queries[i]
            else:
                break  # If user queries are exhausted, stop processing
            #prompt = input("Please input your prompt: ")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")

            # Check if prompt contains 'segment' and break loop if true
            if "segment" in prompt.lower():
                #print("Detected 'segment' in prompt. Ending conversation.")
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

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
                history.append(text_output)
                print("text_output: ", text_output)

                pred_mask = pred_masks[0].detach().cpu().numpy() > 0
                #pred_masks_list.append(pred_mask)
                break

        prompt = conv.get_prompt()
        print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

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
        history.append(text_output)
        print("text_output: ", text_output)

        pred_mask = pred_masks[0].detach().cpu().numpy() > 0
        #pred_masks_list.append(pred_mask)
    
    return history, pred_mask



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
    #加载测试集
    with open('/datas/caidexian/myfiles/processed_val_02.json','r') as f:
        multiturn_data = json.load(f)
    mask_data = '/datas/multimodal_datasets/GRESDataset/multiturn/Multiturn/val/masks'
    base_image_dir = '/datas/multimodal_datasets/GRESDataset/multiturn/Multiturn/val'
    conv = conversation_lib.conv_templates[args.conv_type].copy()

    all_text_outputs = []
    #all_pred_masks = []
    #all_cIoU_scores = []

    # cum_I, cum_U = 0, 0  # 用于累积IoU的交集与并集
    # mean_IoU = []  # 存储每个预测的IoU
    # seg_total = 0  # 总图片数
    # eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]  # 不同的IoU阈值
    # seg_correct = np.zeros(len(eval_seg_iou_list))  # 存储不同阈值下的精度统计
    ciou_list=[]
    b_iou_list = []
    pixel_acc_list = []

    i=0
    for item in multiturn_data:
        image_path = os.path.join(base_image_dir, item['img_id'])
        json_path = os.path.join(mask_data, item['img_id'].replace('.jpg', '.json'))
        focus = item['focus']
        #mask标签
        mask = get_mask_from_json_target(json_path, focus)
        gt_mask = [mask]
        gt_mask = np.stack(gt_mask, axis=0)
        gt_mask = torch.from_numpy(gt_mask)

        #对话标签
        source = item['dialogue']
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        gt_conversations = []
        user_queries = []  # 用来存储用户的询问内容
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
            
            # 如果该消息来自用户（即 "human"），将其添加到 user_queries 列表中
            if role == conv.roles[0]:  # conv.roles[0] 表示 human 角色
                user_queries.append(sentence["value"])
        gt_conversations.append(conv.get_prompt())

        history, pred_masks , image_np= process_multiturn(conv, args, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
                                    clip_image_processor, model, tokenizer_image_token, IMAGE_TOKEN_INDEX, tokenizer, transform, preprocess,
                                    image_path,user_queries)
        print(history)

        for k,pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue
            #print(pred_mask.shape,gt_mask.shape)
            #pred_mask = pred_mask.detach().cpu().numpy()[0]
            #pred_mask = pred_mask > 0
            gt_mask_np = gt_mask.detach().cpu().numpy()

            # 计算 B-IoU
            b_iou = calculate_bIoU(pred_mask, gt_mask_np)
            b_iou_list.append(b_iou)
            print(f"bIoU for {image_path}: {b_iou:.4f}")

            # 计算 Pixel Accuracy in ROI
            pixel_acc = calculate_pixel_accuracy_in_roi(pred_mask, gt_mask_np)
            pixel_acc_list.append(pixel_acc)
            print(f"Pixel Accuracy in ROI: {pixel_acc:.4f}")

            ciou = calculate_cIoU(pred_mask, gt_mask_np)
            ciou_list.append(ciou)
            print(f"cIoU for {image_path}: {ciou:.4f}")



            # save_path = "{}/{}_mask_{}.jpg".format(
            #     args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            # )
            # cv2.imwrite(save_path, pred_mask * 100)
            # print("{} has been saved.".format(save_path))

            # save_path = "{}/{}_masked_img_{}.jpg".format(
            #     args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            # )
            # save_img = image_np.copy()
            # save_img[pred_mask] = (
            #     image_np * 0.5
            #     + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            # )[pred_mask]
            # save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(save_path, save_img)
            # print("{} has been saved.".format(save_path))
        
        i+=1

        all_text_outputs.append({
            "img_id": item['img_id'],
            "focus": focus,
            "response": history
        })
        # 保存对话输出
        with open("all_text_outputs_stage2.json", "w") as f:
            json.dump(all_text_outputs, f,indent=4)
        #all_pred_masks.append(pred_mask)
        
    # 计算平均 B-IoU 和 Pixel Accuracy in ROI
    mean_b_iou = np.mean(b_iou_list) if len(b_iou_list) > 0 else 0
    mean_pixel_acc_roi = np.mean(pixel_acc_list) if len(pixel_acc_list) > 0 else 0
    mean_ciou = np.mean(ciou_list) if len(ciou_list) > 0 else 0

    print(f"Mean B-IoU: {mean_b_iou:.4f}")
    print(f"Mean Pixel Accuracy in ROI: {mean_pixel_acc_roi:.4f}")
    print(f"Mean cIoU: {mean_ciou:.4f}")



if __name__ == "__main__":
    main(sys.argv[1:])