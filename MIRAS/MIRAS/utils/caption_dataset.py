import json
import os
import random
import numpy as np

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

#from model.mgm import conversation as conversation_lib
# from model.segment_anything.utils.transforms import ResizeLongestSide
from mgm import conversation as conversation_lib
from segment_anything.utils.transforms import ResizeLongestSide

from .utils_v1 import CAPTION_QUESTIONS,DEFAULT_IMAGE_TOKEN

class CocoCapDataset(torch.utils.data.Dataset):
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
        num_classes_per_sample: int = 3,
        exclude_val=False,
        cap_data="coco/annotations",
        args=None,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        #self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor = vision_tower
        self.cap_image_root = os.path.join(base_image_dir, "coco/train2017")
        DATA_DIR=os.path.join(base_image_dir, cap_data)
        mode = "train"
        with open(os.path.join(DATA_DIR, "captions_{}2017.json".format(mode))) as f:
            cap_data = json.load(f)
        self.cap_data = cap_data

        self.cap_query_list = CAPTION_QUESTIONS

        self.data_args = args
        print("caption dataset initialized:",len(self.cap_data["images"]))

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
        idx = random.randint(0, len(self.cap_data['images']) - 1)
        item = self.cap_data['images'][idx]
        item_image_id=item["id"]
        image_name= item['file_name']
        image_path = os.path.join(self.cap_image_root,image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_=image[np.newaxis,:]
        image_clip = self.clip_image_processor.preprocess(image_, return_tensors="pt")[
            "pixel_values"
        ][0]  # preprocess image for clip
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

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        #获取对应图像的多个captions
        captions = [item['caption'] for item in self.cap_data['annotations'] if item['image_id']==item_image_id]
        num_captions = min(len(captions), self.num_classes_per_sample)
        captions = random.sample(captions, num_captions)
        #构造成对话
        questions = []
        answers=[]
        for cap in captions:
            question = DEFAULT_IMAGE_TOKEN+"\n"+random.choice(self.cap_query_list)
            questions.append(question)
            answers.append(cap)

        conversations = []
        conv=conversation_lib.default_conversation.copy()

        i=0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0],questions[i])
            conv.append_message(conv.roles[1],answers[i])
            conversations.append(conv.get_prompt())
            i+=1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()) # (3, 1024, 1024)
        

        masks = torch.rand(0,*ori_size)
        label = torch.ones(ori_size) * self.ignore_label

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
            captions,
        )


