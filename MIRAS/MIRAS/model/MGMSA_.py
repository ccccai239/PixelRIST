from typing import List, Dict, Any, Tuple
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from MGMSA.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from mgm.model.language_model.mgm_llama import (MGMLlamaForCausalLM,MGMLlamaModel)
#from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,LlavaLlamaModel)
from MGMSA.model.segment_anything.build_sam import build_sam_vit_h



def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=10000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def mask_after_first_occurrence(input_ids, seg_token_idx):
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i in range(input_ids.size(0)):
        idx = (input_ids[i] == seg_token_idx).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            mask[i, :idx[0]+1] = (input_ids[i, :idx[0]+1] == seg_token_idx)
    return mask


class MgmsaMetaModel:
    def __init__(self,
                config,
                **kwargs,) -> None:
        super(MgmsaMetaModel, self).__init__(config)
        self.config = config

        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_mgmsa_modules(self.config)

    def initialize_mgmsa_modules(self, config):
        #SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        
        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class MgmsaModel(MgmsaMetaModel,MGMLlamaModel):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(MgmsaModel, self).__init__(config, **kwargs)
        self.config.use_cache = False
        self.config.vision_tower=self.config.mm_vision_tower
        self.config.vision_tower_aux=self.config.mm_vision_tower_aux
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.config.optimize_vision_tower = True #
        self.config.optimize_vision_tower_aux = True #

class MgmsaForCausalLM(MGMLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            config.mm_vision_tower_aux = kwargs.get(
                "vision_tower_aux", "openai/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight",None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight",None)
        else:
            config.mm_vision_tower = config.vision_tower
            config.mm_vision_tower_aux = config.vision_tower_aux

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)
        self.ce_loss_weight = config.ce_loss_weight if hasattr(config, 'ce_loss_weight') else 1.0
        self.dice_loss_weight = config.dice_loss_weight if hasattr(config, 'dice_loss_weight') else 0.5
        self.bce_loss_weight = config.bce_loss_weight if hasattr(config, 'bce_loss_weight') else 2

        self.model= MgmsaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.layer_weights = nn.Parameter(torch.ones(config.num_hidden_layers+1))

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list= []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(pixel_values[i].unsqueeze(0))
                #image_embeddings = self.model.visual_model(pixel_values[i])
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, dim=0)
        return image_embeddings
    
    def forward(self,**kwargs):
        if "past_key_values" in kwargs :
            output=super().forward(**kwargs)
            return output
        return self.model_forward(**kwargs)
    
    def model_forward(
        self,
        images:torch.FloatTensor,  #(1,3,1024,1024)
        images_clip: torch.FloatTensor, #(5,3,336,336)
        images_aux: torch.FloatTensor, #（1，3，336，336）
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,     
    ):
        image_embeddings = self.get_visual_embs(images) #(bs,256,64,64)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) -1

        #print(type(self.seg_token_idx),self.seg_token_idx)
        seg_token_mask =input_ids[:,1:] == self.seg_token_idx #(3,65)
        #seg_token_mask = mask_after_first_occurrence(input_ids[:, 1:], self.seg_token_idx)

        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )  #(3,66)
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        ) #(3,321)
        if inference:
            n_batch=1
            length=input_ids.shape[0]
            assert images_clip.shape[0]==1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            images_aux_extend = images_aux.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            output_text=[]
            for i in range(n_batch):
                start_i,end_i = i * length,min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[:end_i-start_i],
                    images_aux=images_aux_extend[:end_i-start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                output_text.append(output_i.logits)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states = [tensor for tuple in output_hidden_states for tensor in tuple]
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []

            images_aux_list = []
            for i in range(len(offset) - 1):
                #images_clip_i = []
                start_i, end_i = offset[i], offset[i + 1]
                # images_clip_i = (
                #     images_clip[i]
                #     .unsqueeze(0)
                #     .expand(end_i - start_i, -1, -1, -1)
                #     .contiguous()
                # ) #（3，3，336，336），这一步是为了将一张图片复制成多张
                images_aux_i = (
                    images_aux[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                ) 
                images_clip_i= (
                        images_clip[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous())
                images_clip_list.append(images_clip_i)
                images_aux_list.append(images_aux_i)
            images_clip = torch.cat(images_clip_list, dim=0) #（bs*3，3，336，336）
            images_aux = torch.cat(images_aux_list, dim=0) 

            output = super().forward(
                images=images_clip,
                images_aux=images_aux,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) #(bs,2048,256)
        """ for i in range(len(output_hidden_states)):
            # 遍历所有层并应用相应的线性变换
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[i]))
        #hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        # 将所有层的隐含表示堆叠起来
        hidden_states = torch.stack(hidden_states, dim=-1) #(6,643,256,33)
        # 对每层的隐含表示进行加权和
        layer_weights = F.softmax(self.layer_weights, dim=0)  # 使用softmax使权重和为1
       
        weighted_sum_hidden_states = (hidden_states * layer_weights.view(1, 1, -1)).sum(dim=-1) """
        #seg_token_mask (3,321)
        # 创建一个与last_hidden_state形状匹配的掩码
        #seg_token_mask_expanded = seg_token_mask.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1])
        #indices = torch.arange(seg_token_mask.shape[0]).unsqueeze(-1).expand(-1, seg_token_mask.shape[1])
        #print(last_hidden_state.shape,seg_token_mask.shape)

        #pred_embeddings = last_hidden_state[seg_token_mask]
        #seg_token_mask = seg_token_mask.to(torch.int64)
        #pred_embeddings = torch.gather(last_hidden_state, 1, seg_token_mask.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1])) #(bs,321,256)
        # 找到 seg_token_mask 中值为1的元素的索引
        indices = torch.nonzero(seg_token_mask == 1)
        # 使用这些索引从 last_hidden_state 中选择嵌入向量
        pred_embeddings = last_hidden_state[indices[:, 0], indices[:, 1]]
        
        #seg_token_counts = seg_token_mask.int().sum(-1)  #[bs, ]
        #seg_token_offset = seg_token_counts.cumsum(-1)
        #seg_token_offset = torch.cat(
        #    [torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        #seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ =[]
        #gt_masks = []
        #assert len(masks_list) == len(seg_token_offset) - 1
        sub=0
        last_i,s_obj,sum_none=0,0,0
        for i in range(len(offset)-1):
            assert len(masks_list)==len(offset)-1
            #start_i,end_i = seg_token_offset[i],seg_token_offset[i+1]
            start_i, end_i = offset[i], offset[i + 1]
            # 去除第i个元素
            # if masks_list[i].shape[0] != 0 and end_i == start_i+1 :
            #     if i == 0: 
            #         sub=0
            #     else: 
            #         sub -=1

            if masks_list[i].shape[0]==0:
                if i == 0:
                    image_embeddings = image_embeddings[1:]
                    resize_list = resize_list[1:]
                    label_list = label_list[1:]
                    #last_i = 0
                #elif i> 0 or sub:
                else:
                    image_embeddings = torch.cat((image_embeddings[:i-sub], image_embeddings[i-sub + 1:]))
                    resize_list = resize_list[:i-sub]+resize_list[i-sub + 1:]
                    label_list = label_list[:i-sub]+label_list[i-sub + 1:]
                    #last_i -= 1  # Adjust for removed element in the next steps
            
                sub +=1
                sum_none += ((end_i-start_i) if end_i-start_i>1 else 1)
                continue
            if sub>=1:
                pred_embeddings_.append(pred_embeddings[start_i-sum_none:end_i-sum_none])
            else:
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            #if (end_i-start_i)//len(masks_list[i]) == 1:
            #    gt_masks.append(masks_list[i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        gt_masks = masks_list
        if len(gt_masks) != len(pred_embeddings):
            gt_masks=[x for x in gt_masks if len(x)!=0]
        
        #print(len(pred_embeddings),len(masks_list))
        for i in range(len(pred_embeddings)):
            if len(pred_embeddings[i])==1:
                (
                    sparse_embeddings,
                    dense_embeddings
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks,iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )#low_res_masks:(3,1,256,256)
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )[:, 0]

            elif len(pred_embeddings[i])>1:
                pred_embeddings[i]=pred_embeddings[i].unsqueeze(1)
                pred_masks_ = []
                for j in range(len(pred_embeddings[i])):
                    (
                        sparse_embeddings,
                        dense_embeddings
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i][j].unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i][j].dtype)
                    low_res_masks,iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=label_list[i].shape,
                    )
                    pred_masks_.append(pred_mask[:, 0])
                pred_mask = torch.stack(pred_masks_, dim=0)[:,0]
            else:
                continue
                #mask_tmp=torch.zeros_like(pred_mask)
                #masks_list[i] = mask_tmp            
            
            pred_masks.append(pred_mask)

        model_output =output
        

        if inference:
            return{
                    "logits": output_text,
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }
            
        output = model_output.logits


        ce_loss = model_output.loss
        ce_loss = ce_loss *self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            # if gt_mask.shape[0] < pred_mask.shape[0]:
            #     pred_mask = pred_mask[: gt_mask.shape[0]]
            # elif gt_mask.shape[0] > pred_mask.shape[0]:
            #     pad_size = gt_mask.shape[0] - pred_mask.shape[0]
            #     pred_mask = np.pad(pred_mask.cpu().detach().numpy(), ((0, pad_size), (0, 0), (0, 0)), 'constant', constant_values=0)
            #     pred_mask = torch.from_numpy(pred_mask).to(gt_mask.device)

            assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
            #import pdb
            #pdb.set_trace()
            #print(gt_mask.shape, pred_mask.shape)
            if gt_mask.ndim==4:
                gt_mask = gt_mask.squeeze(1)
                
            mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
            mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            #"layer_weights":self.layer_weights
        }
    def evaluate(
        self,
        images_clip,
        images_aux,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        
        with torch.no_grad():
            outputs = self.generate(
                input_ids,
                images=images_clip,
                images_aux=images_aux,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                temperature=0.1,
                do_sample=False,
                top_p=None,
                use_cache=True,
                )
            
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            # seg_token_mask = torch.cat(
            #     [
            #         torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
            #         seg_token_mask,
            #     ],
            #     dim=1,
            # )Can

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            #pred_embeddings = last_hidden_state[seg_token_mask]
            #seg_token_mask = seg_token_mask.to(torch.int64)
            #pred_embeddings = torch.gather(last_hidden_state, 1, seg_token_mask.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1]))
            # 找到 seg_token_mask 中值为1的元素的索引
            indices = torch.nonzero(seg_token_mask == 1)
            # 使用这些索引从 last_hidden_state 中选择嵌入向量
            pred_embeddings = last_hidden_state[indices[:, 0], indices[:, 1]]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])


        return output_ids, pred_masks


        