from typing import List, Dict, Any, Tuple
from typing import List
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from skimage.feature import canny
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

def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.8,
    gamma: float = 2,
    smooth: float = 1,
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
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
    focal_loss = focal_loss.sum() / (num_masks + 1e-8)
    return focal_loss



""" def boundary_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=10000.0,
    eps=1e-6
):
    
    Compute the boundary loss, which measures the similarity between
    the boundaries of the predicted mask and the ground truth mask.

    Args:
        inputs: A float tensor of arbitrary shape. The predicted mask.
        targets: A float tensor with the same shape as inputs. The ground truth mask.
        num_masks: The number of masks in the batch.
        scale: Optional scaling factor for normalization.
        eps: A small epsilon value to avoid division by zero.
    
    
    # Step 1: Sigmoid Activation and Binarization of Inputs
    inputs = (inputs.sigmoid() > 0.3).float()  # 使用 0.5 作为阈值进行二值化
    #targets = (targets > 0.5).float()  # 目标掩码同样进行二值化，假设目标是已经二值化的
    
    # Step 2: Convert Tensors to NumPy Arrays
    inputs_np = inputs.detach().cpu().numpy().astype(np.uint8)  # 转换为 NumPy 数组
    targets_np = targets.detach().cpu().numpy().astype(np.uint8)

    # Step 3: Use findContours for Boundary Detection
    def find_contours_boundary(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_image = np.zeros_like(mask)  # 创建一个空白图像来绘制轮廓
        cv2.drawContours(boundary_image, contours, -1, (255), 1)  # 绘制轮廓到空白图像上，边界值为255
        return boundary_image

    # 应用 findContours 获取输入和目标的边界图
    inputs_boundary_np = np.array([find_contours_boundary(mask) for mask in inputs_np])
    targets_boundary_np = np.array([find_contours_boundary(mask) for mask in targets_np])

    # Step 4: Convert Boundaries Back to PyTorch Tensors
    inputs_boundary = torch.tensor(inputs_boundary_np, device=inputs.device).float()
    targets_boundary = torch.tensor(targets_boundary_np, device=targets.device).float()

    # Step 5: Compute the L2 Loss between the boundaries
    loss = F.mse_loss(inputs_boundary / scale, targets_boundary / scale, reduction='none')
    loss = loss.view(loss.size(0), -1).sum(dim=-1)  # 展平所有像素并在最后一个维度上求和
    
    # Step 6: Normalize the loss across the number of masks
    loss = loss.sum() / (num_masks + eps)  # 对损失进行归一化处理
    
    return loss """

def sobel_conv(tensor: torch.Tensor) -> torch.Tensor:
    """
    使用 Sobel 卷积核来检测图像边缘。
    
    Args:
        tensor: 输入的二值图像张量 (batch, 1, height, width)
    
    Returns:
        边缘图像张量
    """
    # Sobel 卷积核
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 将 Sobel 卷积核移动到同一设备上
    sobel_kernel_x = sobel_kernel_x.to(tensor.device)
    sobel_kernel_y = sobel_kernel_y.to(tensor.device)

    # 对输入图像应用 Sobel 卷积
    grad_x = F.conv2d(tensor, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(tensor, sobel_kernel_y, padding=1)

    # 合成梯度
    edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # 返回边缘图像
    return edge_map

def boundary_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000.0,
    eps=1e-6
):
    """
    计算二值图像的边界损失，使用 Sobel 卷积核。
    
    Args:
        inputs: 预测掩码张量 (batch, height, width)
        targets: 真实掩码张量 (batch, height, width)
        num_masks: 批次中的掩码数量
        scale: 可选的归一化因子
        eps: 防止除零的微小数值
    """
    
    # Step 1: Apply Sigmoid and Binarize the Inputs
    inputs = (inputs.sigmoid() > 0.3).float().unsqueeze(1)  # 将预测掩码进行二值化，并添加通道维度
    targets = targets.float().unsqueeze(1)  # 目标掩码已经是二值化的，并添加通道维度

    # Step 2: Detect edges using Sobel convolution for both inputs and targets
    inputs_boundary = sobel_conv(inputs)
    targets_boundary = sobel_conv(targets)

    # Step 3: Compute the L2 loss between the boundaries
    loss = F.mse_loss(inputs_boundary / scale, targets_boundary / scale, reduction='none')
    loss = loss.view(loss.size(0), -1).sum(dim=-1)  # 展平所有像素并在最后一个维度上求和

    # Step 4: Normalize the loss across the number of masks
    loss = loss.sum() / (num_masks + eps)  # 对损失进行归一化处理
    
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
            self.focal_loss_weight = kwargs.pop("focal_loss_weight", None)
            self.boundary_loss_weight = kwargs.pop("boundary_loss_weight", None)
            #self.bce_loss_weight = kwargs.pop("bce_loss_weight",None)
        else:
            config.mm_vision_tower = config.vision_tower
            config.mm_vision_tower_aux = config.vision_tower_aux

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)
        self.ce_loss_weight = config.ce_loss_weight if hasattr(config, 'ce_loss_weight') else 1.0
        self.dice_loss_weight = config.dice_loss_weight if hasattr(config, 'dice_loss_weight') else 0.5
        #self.bce_loss_weight = config.bce_loss_weight if hasattr(config, 'bce_loss_weight') else 2
        self.focal_loss_weight = config.focal_loss_weight if hasattr(config, 'focal_loss_weight') else 1
        self.boundary_loss_weight = config.boundary_loss_weight if hasattr(config, 'boundary_loss_weight') else 0.5

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
        if "eval" in kwargs:
            with torch.no_grad():
                output = self.evaluate(**kwargs)
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
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        ) #(3,321)

        if inference:
            #self.model.eval()
            n_batch=1
            length=input_ids.shape[0]
            assert images_clip.shape[0]==1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            images_aux_extend = images_aux.expand(length, -1, -1, -1).contiguous()
            #tokenizer = kwargs.get("tokenizer")
            output_hidden_states = []
            output_text=[]
            inputs = []
            for i in range(n_batch):
                start_i,end_i = i * length,min((i + 1) * length, input_ids.shape[0])
                # output_ids, pred_masks = self.evaluate(
                #                 images_clip_extend[:end_i-start_i],
                #                 images_aux_extend[:end_i-start_i],
                #                 images[start_i:end_i],
                #                 input_ids[start_i:end_i],
                #                 resize_list[i],
                #                 label_list[i],
                #                 max_new_tokens=128,
                #                 tokenizer=tokenizer,
                #                 )
                # output_ids = output_ids[0][output_ids[0] != -200]

                # text_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                # text_output = text_output.replace("\n", "").replace("  ", " ")
                # print("text_output: ", text_output)
                
                # input_ids = input_ids[start_i:end_i]
                # out_put = self.generate(inputs = input_ids,
                #                         images = images_clip_extend[:end_i-start_i],
                #                         images_aux=images_aux_extend[:end_i-start_i])
                # print(out_put) 

                output_i = super().forward(
                    images=images_clip_extend[:end_i-start_i],
                    images_aux=images_aux_extend[:end_i-start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                output_text.append(output_i.logits)
                inputs.append(input_ids[start_i:end_i])
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
    
        # 找到 seg_token_mask 中值为1的元素的索引
        indices = torch.nonzero(seg_token_mask == 1)
        # 使用这些索引从 last_hidden_state 中选择嵌入向量
        pred_embeddings = last_hidden_state[indices[:, 0], indices[:, 1]]
        

        pred_embeddings_ =[]
        #gt_masks = []
        sub=0        #assert len(masks_list) == len(seg_token_offset) - 1

        last_i,s_obj,sum_none=0,0,0
        for i in range(len(offset)-1):
            assert len(masks_list)==len(offset)-1
            #start_i,end_i = seg_token_offset[i],seg_token_offset[i+1]
            start_i, end_i = offset[i], offset[i + 1]

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
            
            pred_masks.append(pred_mask)

        model_output =output
        

        if inference:
            return{
                    "logits": output_text,
                    "inputs": inputs,
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }
            
        output = model_output.logits


        ce_loss = model_output.loss
        ce_loss = ce_loss *self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        mask_focal_loss = 0
        mask_boundary_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
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
            
            mask_focal_loss += (
                    focal_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
            mask_boundary_loss += (
                    boundary_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
            
            num_masks += gt_mask.shape[0]

        #mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)

        mask_focal_loss = self.focal_loss_weight * mask_focal_loss / (num_masks + 1e-8)
        mask_boundary_loss = self.boundary_loss_weight * mask_boundary_loss / (num_masks + 1e-8)
        #mask_loss = mask_bce_loss + mask_dice_loss
        mask_loss = mask_dice_loss + mask_focal_loss + mask_boundary_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            #"mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,\
            "mask_focal_loss": mask_focal_loss,
            "mask_boundary_loss": mask_boundary_loss,
            "mask_loss": mask_loss,
            #"layer_weights":self.layer_weights
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        images_clip,
        images_aux,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=None,
        **kwargs,
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

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
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


        