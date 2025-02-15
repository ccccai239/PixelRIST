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
SEG_TOKEN="[SEG]"
#REJ_TOKEN="[REJ]"

#划分出object名称
OBJECT_NAME_START="[OBJ]"

CAPTION_QUESTIONS = [
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please provide a thorough description of the this image.',
    'Please describe in detail the contents of the image.',
    'Could you give a comprehensive explanation of what can be found within this picture?',
    'Could you give me an elaborate explanation of this picture?',
    'Could you provide me with a detailed analysis of this photo?',
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please describe in detail the contents of the image.',
    'Can you give a comprehensive explanation of this photo',
    'Please provide an elaborate explanation of this picture.',
    'Please provide an elaborate explanation of this picture',
    'Could you provide me with a detailed analysis of this photo',
    'Write a comprehensive caption for the image provided.',
    'Can you help me understand the image by providing a detailed caption?',
    'Please provide a vivid description of the image.',
    'Elaborate on the details of the image provided.',
    'Could you please interpret the image and write a detailed caption?',
    'Please depict the image in words.',
    'How would you describe the image to someone who cannot see it?',
    'Please enlighten me with a detailed description of the image.',
    'Can you transform the visual elements of the image into words?',
    'Please provide a detailed written representation of the image.',
    'Could you please transcribe the image into a descriptive paragraph?',
    'Please illustrate the image through your words.',
    'Please provide a detailed narrative of the image.',
    'Could you please express the image in a descriptive format?',
    'Please convert the visual information in the image into a detailed written explanation.',
]

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {classes_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {classes_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {classes_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {classes_name} in this image? Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you identify and segment the {classes_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Identify the {classes_name} in this image and provide a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please find and segment the {classes_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Show the segmentation mask for {classes_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What part of this image shows the {classes_name}? Please segment it.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Which area in this image corresponds to {classes_name}? Provide a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you mark the {classes_name} in this image with a segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please identify and segment the {classes_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What part of the image contains the {classes_name}? Provide a segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the {classes_name} area in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Locate and segment the {classes_name} in this image.",
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
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
    "Here is the segmentation: [SEG].",
    "The segmentation mask is [SEG].",
    "This is the segmentation: [SEG].",
    "Identified as [SEG].",
    "Segmented as [SEG].",
    "The result is [SEG].",
    "Certainly, [SEG].",
    "The segmentation is [SEG].",
    "Definitely, [SEG].",
    "The identified area is [SEG].",
]

ANSWER_LIST_MULTI = [
    "The segmentation result is " + OBJECT_NAME_START + "{seg_result}.",
    "The segmentation is " + OBJECT_NAME_START + "{seg_result}.",
    "Sure, " + OBJECT_NAME_START + "{seg_result}.",
    "Sure, the result is " + OBJECT_NAME_START + "{seg_result}.",
    "Sure, the segmentation result is " + OBJECT_NAME_START + "{seg_result}.",
    "Sure, the segmentation is " + OBJECT_NAME_START + "{seg_result}.",
    "The result is " + OBJECT_NAME_START + "{seg_result}.",
    OBJECT_NAME_START + "{seg_result}.",
    "Identified as " + OBJECT_NAME_START + "{seg_result}.",
    "Segmented as " + OBJECT_NAME_START + "{seg_result}.",
    "Certainly, the segmentation is " + OBJECT_NAME_START + "{seg_result}.",
    "Definitely, the segmentation result is " + OBJECT_NAME_START + "{seg_result}.",
    "The segmented result is " + OBJECT_NAME_START + "{seg_result}.",
    "Here is the segmentation: " + OBJECT_NAME_START + "{seg_result}.",
    "The area identified is " + OBJECT_NAME_START + "{seg_result}.",
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
    if output.dim() == 3 and output.size(0) == 1:
        output = output.squeeze(0)
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
