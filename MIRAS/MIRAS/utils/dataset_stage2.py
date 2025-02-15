import glob
import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

#from model.mgm import conversation as conversation_lib
from mgm import conversation as conversation_lib
from mgm.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from mgm.mm_utils import tokenizer_image_token
from segment_anything.utils.transforms import ResizeLongestSide

from MGMSA.utils.conversation import get_default_conv_template
from MGMSA.utils.data_processing import get_mask_from_json,get_mask_from_json_target
from MGMSA.utils.reason_seg_dataset import ReasonSegDataset
from MGMSA.utils.refer import REFER
from MGMSA.utils.refer_seg_dataset import ReferSegDataset
from MGMSA.utils.sem_seg_dataset import SemSegDataset
from MGMSA.utils.caption_dataset import CocoCapDataset
from MGMSA.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from MGMSA.utils.vqa_dataset import VQADataset
from MGMSA.utils.allava_dataset import ALLaVADataset
from MGMSA.utils.vg_dataset import VGDataset
from  MGMSA.utils.multiturn_dataset_v0 import MultiTurnDataset


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    images_aux_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        images_aux,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        images_aux_list.append(images_aux)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # ignore instruction

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if False:
                z = target.clone()
                z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
                if local_rank == 0:
                    print(
                        "conversation: ",
                        conversation,
                        "tokenizer.decode(z): ",
                        tokenizer.decode(z),
                    )
    elif conv_type == "llava_llama_2":
        sep = "[/INST] "
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # ignore instruction

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if False:
                z = target.clone()
                z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
                if local_rank == 0:
                    print(
                        "conversation: ",
                        conversation,
                        "tokenizer.decode(z): ",
                        tokenizer.decode(z),
                    )
    elif conv_type == "llama_3" or conv_type == "qwen2":
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            re_rounds=[conv.sep.join(rounds[:3])] #sys+user+gpt
            for conv_idx in range(3,len(rounds),2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2])) # user + gpt

            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) -1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids) -1
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                # include <|eot_id|> for all rounds
                round_len += 1
                instruction_len += 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # ignore instruction

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if False:
                z = target.clone()
                z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
                if local_rank == 0:
                    print(
                        "conversation: ",
                        conversation,
                        "tokenizer.decode(z): ",
                        tokenizer.decode(z),
                    )

        #if cur_len < tokenizer.model_max_length:
        #    assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "images_aux": torch.stack(images_aux_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="sem_seg||refer_seg||vqa||reason_seg||caption",
        sample_rate=[9, 6, 3, 1,2],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        allava_data="ALLAVA/ALLaVA-Instruct-VFLAN-4V.json",
        reason_seg_data="ReasonSeg|train",
        cap_data="coco/annotations",
        explanatory=0.1,
        vg_data="VG|train",
        multiturn_data="/datas/caidexian/myfiles/processed_multiturn_tot_01.json",
        args=None,

    ):
        self.args = args
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        self.args = args
        #self.image_size_raw = image_size_raw
        #self.image_grid = image_grid
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        args
                        
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        args
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        args
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        args
                    )
                )
            elif dataset == "caption":
                self.all_datasets.append(
                    CocoCapDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        cap_data,
                        args
                    )
                )
            elif dataset == "allava":
                self.all_datasets.append(
                    ALLaVADataset(
                        "/datas/multimodal_datasets",
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        allava_data,
                        args
                    )
                )
            elif dataset == "vg":
                self.all_datasets.append(
                    VGDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vg_data,
                        args
                    )
                )
            elif dataset == "multiturn":
                self.all_datasets.append(
                    MultiTurnDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        exclude_val,
                        "/datas/multimodal_datasets/GRESDataset/textcaps/masks",
                        multiturn_data,
                        args
                    )
                )
                

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        image_processor,
        val_dataset,
        image_size=1024,
        args=None,
        multiturn_data=None,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "multiturn", ds, split, "*.jpg")
            )
            self.images = images
            self.root_dir = os.path.join(self.base_image_dir, "multiturn", ds, split)
            with open(multiturn_data) as f:
                self.multiturn_data = json.load(f)
            self.data_type = "multiturn"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_root=os.path.join(self.base_image_dir,"refer_seg")
            refer_api = REFER(refer_root, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        refer_root,"images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        refer_root,
                        "images/mscoco/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.data_args = args
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        #self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor = image_processor

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.multiturn_data)

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
        #image_total = []
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            item = self.multiturn_data[idx]
            image_path = os.path.join(self.root_dir,item["img_id"])
            #image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = os.path.join(self.root_dir,"masks",item['img_id'].replace(".jpg", ".json"))
            mask_json = get_mask_from_json_target(json_path, item['focus'])
            sampled_sents = item['dialogue']
            
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        if self.data_type !="refer_seg":
            pending_user_message = None
            for turn in sampled_sents:
                conv.messages = []
                role = turn["from"]
                message = turn["value"].strip()

                if role == "human":
                    pending_user_message = message
                    #conv.append_message(conv.roles[0], message)
                    #pre_role = role
                elif role == "gpt":
                    if pending_user_message is not None:
                        # Append the user's message followed by the assistant's response
                        conv.append_message(conv.roles[0], pending_user_message)
                        conv.append_message(conv.roles[1], message)
                        pending_user_message = None  # Clear after pairing
                    else:
                        continue

                    conversations.append(conv.get_prompt())
        else:
            i = 0
            while i < len(sampled_sents):
                conv.messages = []
                text = sampled_sents[i].strip()
                if is_sentence:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n {} Please output segmentation mask.".format(text),
                    )
                    conv.append_message(conv.roles[1], "[SEG].")
                else:
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n What is {} in this image? Please output segmentation mask.".format(
                            text
                        ),
                    )
                    conv.append_message(conv.roles[1], "[SEG].")
                conversations.append(conv.get_prompt())
                i += 1
        

        image_=image[np.newaxis,:]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image_, return_tensors="pt")[
            "pixel_values"
        ][0]

        #image_total.append(image_clip)
        #if len(image_total) > 1:
        #    image_clip = torch.stack(image_total, dim=0)
        #else:
        #    image_clip = image_total[0]

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

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
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
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            image_aux,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
