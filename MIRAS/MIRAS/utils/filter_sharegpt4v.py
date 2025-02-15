import json

# 读取原始json文件
with open('/datas/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json', 'r') as f:
    data = json.load(f)

# 筛选出键为“image”的值以“coco/train2017/”和“sam/images/”为前缀的所有数据
filtered_data = [item for item in data if 'image' in item and (item['image'].startswith('coco/train2017/') or item['image'].startswith('sam/images/'))]
#filtered_data = [item for item in data if 'image' in item and item['image'].startswith('coco/train2017/') or ]

# 将筛选后的数据保存到新的json文件中
with open('filtered_sharegpt4v.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)
