from typing import List, Tuple, Type,int,bool
import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class BoxDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        box_head_depth: int = 3,
        box_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts bounding boxes given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict bounding boxes
          activation (nn.Module): the type of activation to use when
            upscaling embeddings
          box_head_depth (int): the depth of the MLP used to predict
            bounding box coordinates
          box_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict bounding box coordinates
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.box_token = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.box_prediction_head = MLP(
            transformer_dim, box_head_hidden_dim, 4, box_head_depth, sigmoid_output=True
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bounding boxes given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs

        Returns:
          torch.Tensor: batched predicted bounding boxes
          torch.Tensor: batched predictions of bounding box quality
        """
        boxes, iou_pred = self.predict_boxes(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Prepare output
        return boxes, iou_pred

    def predict_boxes(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts bounding boxes. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.box_token.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # image_embeddings: [1, C, H, W], tokens: [B, N, C]
        # dense_prompt_embeddings: [B, C, H, W]
        # Expand per-image data in batch direction to be per-box
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        box_token_out = hs[:, 1, :]

        # Upscale embeddings and predict boxes using the box token
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlps(box_token_out)
        b, c, h, w = upscaled_embedding.shape
        boxes = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, 4, h, w
        )

        # Generate box quality predictions
        iou_pred = self.box_prediction_head(iou_token_out)

        return boxes, iou_pred
    



class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
