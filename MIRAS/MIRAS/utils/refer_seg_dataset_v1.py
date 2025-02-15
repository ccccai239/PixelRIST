import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

# from model.mgm import conversation as conversation_lib
# from model.segment_anything.utils.transforms import ResizeLongestSide
from mgm import conversation as conversation_lib
from segment_anything.utils.transforms import ResizeLongestSide
from .grefer import G_REFER
from .refer import REFER
from .utils_v1 import ANSWER_LIST_MULTI, SHORT_QUESTION_LIST,ANSWER_LIST


class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        args=None,
        #image_size_raw=None,
        #image_grid=1,

    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.data_args =args
        #self.image_size_raw = args.image_size_raw
        #self.image_grid = args.image_grid
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        #self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.clip_image_processor = vision_tower
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list_obj = ANSWER_LIST_MULTI
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

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
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_=image[np.newaxis,:]

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image_, return_tensors="pt")[
            "pixel_values"
        ][0] # (C, H, W)= (3, 336, 336)
        

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

        image = self.transform.apply_image(image)  # preprocess image for sam --> (480, 360, 3)
        resize = image.shape[:2] #image(1024,768,3) -->resize(1024,768)

        questions = []
        answers = []
        #obj_dict = {f"obj{i+1}": val.lower() for i, val in enumerate(sampled_classes)}
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(classes_name=text.lower()))
            #random_num = random.choices([0,1],weights=[0.5,0.5],k=1)[0]
            #if random_num == 0:
            #    answers.append(random.choice(self.answer_list))
                #answers.append(answer_template.format(seg_result=f"{text.lower()}[SEG]"))
            #elif random_num == 1:
            answer_template = random.choice(self.answer_list_obj)
            answers.append(answer_template.format(seg_result=f"{text.lower()}[SEG]"))

        #for i in range(len(sampled_classes)):
        #    question_template = random.choice(self.short_question_list)
            # classes_name=",".join(obj_dict.values())
            # questions.append(question_template.format(classes_name=classes_name))
            # answers_template = random.choice(self.answer_list)
            # seg_results = ",".join([f"{k}[SEG]" for k in obj_dict.values()])
            # answers.append(answers_template.format(seg_result=seg_results))
            
        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m) #（480.360
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle) #（480,360,1）
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)

        # if ds == 'grefcoco' and flag:
        #     import shutil
        #     image_name = image_path.split("/")[-1]
        #     save_dir = os.path.join("/group/30042/xlai/LISA_refactor_final/debug", image_name.split(".")[0])
        #     os.makedirs(save_dir, exist_ok=True)
        #     shutil.copy(image_path, save_dir)
        #     for i in range(masks.shape[0]):
        #         cv2.imwrite(os.path.join(save_dir, "{}_{}_{}.jpg".format(image_name, i, sampled_classes[i])), masks[i].astype(np.int32) * 100)

        masks = torch.from_numpy(masks) #（3，480,360）
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            image_aux,
            conversations,
            masks,
            label, #（480,360）
            resize,
            questions,
            sampled_classes,
        )
