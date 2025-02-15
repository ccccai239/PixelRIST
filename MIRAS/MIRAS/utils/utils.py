from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you identify the {class_name} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Are you able to delineate the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you pinpoint the {class_name} in this photo?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Is it possible for you to highlight the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you discern the {class_name} in the given picture?",

    DEFAULT_IMAGE_TOKEN + "\n" + "Can you provide me with asegment of the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please perform image segmentation to isolate the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Help me segment the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Would you be willing to segment the {class_name}?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you identify {class_name} in this picture? Please provide a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Could you point out {class_name} in this image and show it with a segmentation mask?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, where is {class_name}? I'd appreciate a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please highlight {class_name} in this image using a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In the picture provided, can you show where {class_name} is with a segmentation mask?",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Provide the segmentation mask for better understanding.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Could you include a segmentation mask in your response?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Identify the object with a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} A segmentation mask would be appreciated.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you provide a detailed segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Your response should include a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Could you show this with a segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please illustrate with a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Provide a segmentation mask for clarity.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Could you draw a segmentation mask for this?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} To clarify, please include a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please mark the areas with a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Include a segmentation mask to show the areas.",
]


EXPLANATORY_QUESTION_LIST = [
      "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
    "Provide the segmentation mask and describe your reasoning.",
    "Output the segmentation mask and justify your approach.",
    "Show the segmentation mask and explain your method.",
    "Include a segmentation mask and elaborate on your choice.",
    "Segmentation mask needed, with an explanation of why.",
    "Please give the segmentation mask and discuss the reasoning.",
    "Provide the segmentation mask along with an explanation of the process.",
    "Output the segmentation mask and clarify the reasoning.",
    "Show the segmentation mask and describe why.",
    "Provide a segmentation mask and detail your thought process.",
    "Include a segmentation mask and justify your decision.",
    "Explain your reasoning and provide the segmentation mask.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "The result is [SEG].",
    "Sure, the result is [SEG].",
    "The segmentation result is [SEG].",
    "The segmentation is [SEG].",
    "The result is [SEG].",
    "The segmentation mask is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "Sure, segmentation is [SEG].",
    "Sure, the segmentation is [SEG].",
    "Sure, the segmentation mask is [SEG].",
    "[SEG].",

]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
