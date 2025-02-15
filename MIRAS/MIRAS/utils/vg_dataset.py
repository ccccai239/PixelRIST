import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#from model.mgm import conversation as conversation_lib
#from model.segment_anything.utils.transforms import ResizeLongestSide
from mgm import conversation as conversation_lib
from segment_anything.utils.transforms import ResizeLongestSide
from .utils_v1 import ANSWER_LIST_MULTI, SHORT_QUESTION_LIST

#定义函数从json中取出同张图像中的多个label
def getAttrFromjson(image_path,anns):
    #img_id = image_path.split("/")[-1].split(".")[0] # VG_1000001.jpg
    masks = []
    caps = []
    for ann in anns:
        if ann['image_path'] == image_path:
            masks.append(ann['mask'])
            caps.append(ann['caption'])
    return masks,caps


class VGDataset(torch.utils.data.Dataset):
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
        vg_data="VG|train",
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

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST_MULTI
        vg_data_ann = vg_data.split("|")[0]+"/train_new.json" # VG/train.json
        with open(os.path.join(base_image_dir, vg_data_ann)) as f:
            vg_data = json.load(f)
        self.vg_data = vg_data
        #self.vg_data = os.path.join(base_image_dir, vg_data_ann)
        self.VG_IMG_DIR=os.path.join(base_image_dir, "VG/images")
        self.VG_MASK_DIR = os.path.join(base_image_dir, "VG/masks")
        self.data_args = args
        

        print(f"Loading VG data from {vg_data_ann}")
    
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
        ds =random.randint(0,len(self.vg_data)-1)
        image_path = self.vg_data[ds]['image_path'] #os.path.join(self.VG_IMG_DIR,self.vg_data[ds]['image_path'])
        #image_bbox = self.vg_data['annotations'][ds]['bbox']

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

        masks,caps = getAttrFromjson(image_path,self.vg_data)
        #bbox,caps = self.vg_data[ds]['bbox'],self.vg_data[ds]['caption']
        if len(caps) >=self.num_classes_per_sample:
            sample_caps_idx = np.random.choice(list(range(len(caps))),size=self.num_classes_per_sample,replace=False) 
        else:
            sample_caps_idx = list(range(len(caps)))
        sample_caps = [caps[i] for i in sample_caps_idx]
        sample_masks = [cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                for i in range(len(sample_caps_idx))]
        
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        questions= []
        answers = []
        for text in sample_caps:
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(classes_name=text.lower()))

            answers_template = random.choice(self.answer_list)
            answers.append(answers_template.format(seg_result=f"{text.lower()}[SEG]"))
        
        conversations = []
        conv =conversation_lib.default_conversation.copy()

        i=0
        while i<len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0],questions[i])
            conv.append_message(conv.roles[1],answers[i])
            conversations.append(conv.get_prompt())
            i+=1
        
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
    

        masks = np.stack(sample_masks,axis=0)
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
            sample_caps,
        )





