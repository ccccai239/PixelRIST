from openai import OpenAI
import os
import base64
import openai
import json
import tiktoken
from pydantic import BaseModel, validator
from typing import List, Optional

# 定义元素类
class Element(BaseModel):
    text: str
    number: Optional[str]
    caption: str

# 定义包含元素的类
class Elements(BaseModel):
    elements: List[Element]
# 定义推理路径类
class ReasoningPath(BaseModel):
    path_name: str  # 路径名称
    path_description: str  # 路径描述

# 定义推理节点类
class InferenceNode(BaseModel):
    node_name: str  # 节点名称
    overlap: Optional[str] = None  # 重叠路径信息，例如 "(Overlap of Path 1 and 3)"
    reasoning_paths: List[ReasoningPath]  # 包含的推理路径

# 定义推理树类
class Tree(BaseModel):
    root_node: str  # 根节点
    levels: List[str]  # 各个层级描述，例如 ["First Level: Level 1", "Second Level: Level 2"]
    inference_nodes: List[InferenceNode]  # 推理节点列表

# 定义单轮问答对
class QAPair(BaseModel):
    question: str  # 问题
    answer: str    # 回答

# 定义多轮对话类
class MultiTurnDialogue(BaseModel):
    dialogue: List[QAPair]  # 问答对的列表
    focus: str  # 对话的最终焦点对象

# 定义多轮对话的格式
class MultiTurnDialogues(BaseModel):
    dialogues: List[MultiTurnDialogue]  # 多轮对话的列表

client=OpenAI(
    api_key = "sk-NoC9tBZ8i1SuUgnk741499454bC54fE993Fc2aD69e68D434",
    base_url = "https://neudm.zeabur.app/v1"
)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
 
fig_path='Processed'
 # 假设只有一张图像
filename = '/data/caidexian/10037296734_0b7bcca795_z.jpg'  # 替换为你的图像文件名
image_path = os.path.join(fig_path, filename)
base64_image = encode_image(image_path)

messages=[{
   "role": "system",
   "content": """
You are a reasoning assistant that generates tree-structured Chain of Thought (COT) reasoning paths based on image details. Your task is to create multi-round dialogue formats that represent the reasoning process in a clear and structured manner, focusing on specific, identifiable objects within the image. the final reasoning step identifies a single, distinct object visible in the image.
--Your Task:--
Identify and list the elements visible in the image.
Generate a tree structure of inference paths based on these elements, highlighting overlapping nodes. Output only the tree structure showing all paths.
Create multi-round dialogues for each path to represent the logical progression. At the end of each path, focus on a single identifiable object that directly supports the reasoning conclusion. Output the dialogues and the final focus object for each path."""
}]

caption = "The side of a laptop is shown and the HDMI port is clearly labeled and visible."
#三轮对话的配置
dialogues = [
   {
      "prompt":"""What are the elements in the image?
      Please list as many elements as you can that are clearly visible in the diagram (include detailed information such as text, numbers, etc.)
      Note:At least 3 elements.
      --output JSON format:--
        {
            "elements": [
            {
            "text": "Text description of the element1",
            "number": (optional)"Number of the element",
            "caption": "Caption of the element"
            },
            {
            "text": "Text description of the element2",
            "number": (optional)"Number of the element",
            "caption": "Caption of the element"
            },
            ...
            ]
        }""",
    "response_format":Elements,
   },
   {
    "prompt":"""Task: Create 3 straightforward reasoning questions based on image details, with a score range of 2-3 points.
        --Scoring Criteria:--
        1 Point (Avoid these): Abstract or vague questions, or those that are either too simple (requiring no reasoning) or too complex (not intuitive). These questions typically focus on non-specific or overly broad elements, making them difficult to judge directly from the image.
        Examples:
        Q: How can you tell that the jet bridge is connected to the airplane? (Too simple, no reasoning required)
        Q: What aspects of the clock's design suggest that it is meant to be a decorative piece rather than just functional? (Not intuitive, overly complex)
        Q: Why does the lighting in the image appear soft? (Focus: blurry lighting effects, lacks specific objects)
        Q: Why does this image seem calm? (Focus: overall tone, lacking details)
        Q: What makes this image look vintage? (Focus: style and colors, lacking clear focal points)
        2 Points: Direct questions that can be answered through basic reasoning based on visible elements in the image. The complexity should be moderate, avoiding questions that are too simple or abstract.
        Examples:
        Q: How many photographs in the collage have handwriting that includes letters below them? (Requires simple recognition and counting; reasoning process is moderate)
        Q: How can you determine the stage of the wedding based on people’s positions and attire? (Focus: couple exchanging rings)
        3 Points: Clear questions that require reasoning using image details and common sense, focusing on specific objects. The questions should guide the respondent to observe distinct elements in the image and make reasonable inferences without involving overly complex or abstract reasoning.
        Examples:
        Q: How do bare trees enhance the winter atmosphere? (Focus: bare trees)
        --Guidelines:--
        Focus on Specific Visual Elements: Questions should directly relate to clear, visible objects or actions, avoiding abstract concepts like light, shadow, or mood. Avoid questions that are overly complex or not intuitive.
        Keep Questions Simple and Direct: Avoid complex phrasing; ensure questions are concise and easy to understand, allowing clear reasoning based on visible details.
        Examples:
        Not recommended: “Infer from the activities of the people and equipment to determine signs of the airplane preparing for takeoff.”
        Recommended: “What shows the airplane is preparing for takeoff?”
        Emphasize Clear Observational Details: Each question should highlight one specific object or action, avoiding vague or multi-object descriptions.
        --Requirements:--
        Design 3 reasoning questions based on the listed elements and details from the image and caption, with scores of 2-3 points.
        Each question must involve reasoning and use common sense to analyze visible elements.
        Display a reasoning tree structure: Present the reasoning tree structure showing the progression of reasoning for each question. Highlight any overlapping nodes or steps among the three questions to reflect common reasoning pathways.
        --Example Tree Structure:--
        {
        "root_node": "Observe books on the shelf",
        "levels": [
            "First Level: Read book covers and spine texts",
            "Second Level: Identify book themes"
        ],
        "inference_nodes": [
            {
            "node_name": "Identify chess-related books",
            "overlap": "Overlap of Path 1 and 3",
            "reasoning_paths": [
                {
                "path_name": "Path 1",
                "path_description": "Infer chess association through the “Grunfeld” keyword"
                },
                {
                "path_name": "Path 3",
                "path_description": "Infer theme from multiple chess-related titles"
                }
            ]
            },
            {
            "node_name": "Compare book colors",
            "overlap": "Path 2 Independent",
            "reasoning_paths": [
                {
                "path_name": "Path 2",
                "path_description": "Compare book colors to find the distinct one"
                }
            ]
            }
        ]
        }
        --Note:--
        Please generate the reasoning tree strictly following the below JSON format.
        --Output JSON Format:--
        {
        "root_node": "Root Node A",
        "levels": [
            "First Level: Level 1",
            "Second Level: Level 2"
        ],
        "inference_nodes": [
            {
            "node_name": "Inference Node 1",
            "overlap": (optional)"Overlap of Path 1 and 3",
            "reasoning_paths": [
                {
                "path_name": "Path 1",
                "path_description": "Reasoning Path 1 Description"
                },
                {
                "path_name": "Path 3",
                "path_description": "Reasoning Path 3 Description"
                }
            ]
            },
            {
            "node_name": "Inference Node 2",
            "overlap": (optional)"Path 2 Independent",
            "reasoning_paths": [
                {
                "path_name": "Path 2",
                "path_description": "Reasoning Path 2 Description"
                }
            ]
            }
        ]
        }
"""+f"""Based on caption:{caption}""",
    "response_format":Tree,
     },
     {
        "prompt":"""Task: Based on the reasoning paths generated in the previous step, create multi-turn dialogues for each reasoning path, demonstrating a clear and logical progression of reasoning in a quiz format.
--Guidelines:--
Each multi-turn dialogue should correspond to one reasoning path and must exhibit structured, logical reasoning without unnecessary repetition, unrelated discussions, or vague statements. Avoid generalizations.
Each dialogue round should consist of a guiding question and a specific, direct answer that maintains the flow of reasoning clearly tied to visible elements in the image.
In the final round of each dialogue, identify a distinct, singular object in the image that directly supports the inference. This object should be easily recognizable and should not combine multiple elements (e.g., "Blue light and neon on the roof" are considered two distinct items).
Reasoning should be grounded in clearly identifiable objects and specific details within the image, avoiding broad or abstract references such as light, shadow, tone, color, reflections, overall design, or other ambiguous features.
Focus each question on one specific, clearly defined object visible in the image, ensuring that each answer directly ties the reasoning to that object.
--Requirements:--
Each dialogue must consist of 4 to 8 rounds of questions and answers.
Each question should focus on a single, specific object visible in the image, with reasoning directly tied to the elements listed in the title and the image details.
Expand each reasoning path into a multi-round dialogue format according to the provided reasoning tree, demonstrating the logical progression of thought based on the identified paths.
--Example Based on the Reasoning Process:--
Question3: {
  Q1: "What indicates that some of these books are focused on teaching beginners?",
  A1: "To determine if any books are for beginners, I should look for titles that mention beginner-friendly terms.",
  Q2: "Are there any titles that suggest the book is aimed at beginners?",
  A2: "Yes, there is a book titled 'Chess for Beginners.'",
  Q3: "How does this title specifically indicate it is for beginners?",
  A3: "The phrase 'for Beginners' in the title directly indicates that it is intended to teach or guide beginners."
},
Focus: <Chess for Beginners>
--Output JSON Format:--
Generate the multi-turn dialogues in the specified JSON format for each reasoning path of the reasoning tree.:
{
  "Question1": {
    "Q1": "<First Question>",
    "A1": "<Answer to First Question>",
    "Q2": "<Second Question>",
    "A2": "<Answer to Second Question>",
    "Q3": "<Third Question>",
    "A3": "<Answer to Third Question>",
    ...
  },
  "Focus": "<Final Focus Object1>"
},
{
  "Question2": {
    "Q1": "<First Question>",
    "A1": "<Answer to First Question>",
    "Q2": "<Second Question>",
    "A2": "<Answer to Second Question>",
    "Q3": "<Third Question>",
    "A3": "<Answer to Third Question>",
    ...
  },
  "Focus": "<Final Focus Object2>"
},
{
  "Question3": {
    "Q1": "<First Question>",
    "A1": "<Answer to First Question>",
    "Q2": "<Second Question>",
    "A2": "<Answer to Second Question>",
    "Q3": "<Third Question>",
    "A3": "<Answer to Third Question>",
    ...
  },
  "Focus": "<Final Focus Object3>"
},
Note: Only output the multi-turn dialogue without additional examples or explanations.
""",
        "response_format":MultiTurnDialogues,
     },
]

# 迭代对话
responses = []
for i,dialogue in enumerate(dialogues):
    if i==0 :
        messages.append({"role": "user", "content":[{"type": "text", "text": dialogue["prompt"]},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{base64_image}"}}]})
    else:
        messages.append({"role": "user", "content": {"type": "text", "text": dialogue["prompt"]}})
    try:  
        completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=messages,
                response_format=dialogue["response_format"]
                )
        #解析响应内容
        response_content = completion.choices[0].message
        if response_content.parsed:
            response_content = response_content.parsed
        else:
            response_content = response_content.refusal
        #将响应内容添加到messages列表中
        messages.append({
            "role": "assistant",
            "content": str(response_content)
        })
        if i==0:
            #parsed_elements=Elements.parse_raw(response_content)
            #responses.append(json.dumps(parsed_elements.dict(), indent=2))
            responses.append(response_content)
        elif i==1:
            parsed_tree=Tree.parse_raw(response_content)
            responses.append(json.dumps(parsed_tree.dict(), indent=2))
        else:
            parsed_dialogues=MultiTurnDialogues.parse_raw(response_content)
            responses.append(json.dumps(parsed_dialogues.dict(), indent=2))
    except Exception as e:
        # Handle edge cases
        if type(e) == openai.LengthFinishReasonError:
            # Retry with a higher max tokens
            print("Too many tokens: ", e)
            pass
        else:
            # Handle other exceptions
            print(e)
            pass
       
for idx,res in enumerate(responses):
    print(res)
