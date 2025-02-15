import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

#from model.mgm import conversation as conversation_lib
#from model.segment_anything.utils.transforms import ResizeLongestSide
from mgm import conversation as conversation_lib
from segment_anything.utils.transforms import ResizeLongestSide
from .data_processing import get_mask_from_json_target
from .utils_v1 import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

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

class MultiTurnDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        exclude_val=False,
        mask_data="/datas/multimodal_datasets/GRESDataset/textcaps/masks",
        multiturn_data="/datas/caidexian/myfiles/processed_multiturn_tot_01.json",
        #explanatory=0.1,
        args=None,
    ): 
        self.exclude_val = exclude_val
        #self.multihop_reason_data = multihop_reason_data
        self.samples_per_epoch = samples_per_epoch
        #self.explanatory = explanatory
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        #self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor = vision_tower
        with open(multiturn_data, "r") as f:
            self.multiturn_data = json.load(f)
        self.mask_data = mask_data
        #self.reason_seg_data = (images, jsons)
        self.data_args = args
        
    def __len__(self):
        return self.samples_per_epoch
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):
        #images, jsons = self.reason_seg_data
        item = random.choice(self.multiturn_data)
        image_path = os.path.join(self.base_image_dir, "textcaps", "train", item['img_id'])
        json_path = os.path.join(self.base_image_dir, "textcaps", "masks", item['img_id'].replace('.jpg', '.json'))
        focus = item['focus']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_=image[np.newaxis,:]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image_, return_tensors="pt")[
            "pixel_values"
        ][0]
        #process image_aux
        if image_clip is not None and self.data_args.image_size_raw:
            image_aux = image_clip.clone()
            raw_shape = [self.data_args.image_size_raw['height'] * self.data_args.image_grid,
                         self.data_args.image_size_raw['width'] * self.data_args.image_grid]
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
        else:
            crop_size = self.data_args.image_processor.crop_size
            if hasattr(self.data_args, 'image_size_raw'):
                image_clip = torch.zeros(3, 
                                                 self.data_args.image_size_raw['height'] * self.data_args.image_grid, 
                                                 self.data_args.image_size_raw['width'] * self.data_args.image_grid)
                image_aux = torch.zeros(3, crop_size['height'], crop_size['width'])
            else:
                image_clip = torch.zeros(3, crop_size['height'], crop_size['width'])
        if image_clip is not None and self.data_args.image_grid >=2:
            raw_image=image_clip.reshape(
                3,self.data_args.image_grid, self.data_args.image_size_raw['height'], self.data_args.image_grid, self.data_args.image_size_raw['width']
            )
            raw_image=raw_image.permute(1,3,0,2,4)
            raw_image=raw_image.reshape(
                -1,3,self.data_args.image_size_raw['height'], self.data_args.image_size_raw['width']
            )
            if self.data_args.image_global:
                global_image = image_clip
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image, 
                                                        size=[self.data_args.image_size_raw['height'],
                                                              self.data_args.image_size_raw['width']], 
                                                        mode='bilinear', 
                                                        align_corners=False)
                # [image_crops, image_global]
                image_clip = torch.cat([raw_image, global_image], dim=0).contiguous()  

        mask = get_mask_from_json_target(json_path, focus)
        sampled_masks = [
            (mask == 1).astype(np.float32) 
        ]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        #对话
        conv = conversation_lib.default_conversation.copy()
        #item['dialogue'][0]['question'] = '<image>\n'+item['dialogue'][0]['question']
        source = item["dialogue"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        questions = conversations

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()) # (3, 1024, 1024)

        masks = np.stack(sampled_masks,axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            image_aux,
            conversations,
            masks,
            label,
            resize,
            questions,
            conversations,
            #sampled_sents,
        )