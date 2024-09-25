import torch
import json
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import argparse
torch.set_num_threads(2)

question = '''
你是一名专业的数据标注员，请描述图像中的人物，包括以下几个方面：
    1. 性别和类型（例如，成年男性、成年女性、小女孩、老人等）。
    2. 衣着（包括衣服类型和颜色）。
请将上述信息整合成一段话，严格根据要求，不要输出任何无关内容和不确定的内容，不要输出似乎和好像等不确定性内容。如果图像中没有人物，直接输出：图像中没有人。
''' 
image1 = Image.open('./examples/1.jpg').convert('RGB')
answer1 = "穿着黑白相间的格子外套的深色短发女性"
image2 = Image.open('./examples/2.jpg').convert('RGB')
answer2 = "一名成年男性穿着黄色的工作服、黑色裤子和黑色鞋子"
image3 = Image.open('./examples/3.jpg').convert('RGB')
answer3 = "穿着白色上衣和黑色裤子的中年女性"
image4 = Image.open('./examples/4.jpg').convert('RGB')
answer4 = "一名男性厨师穿着白色厨师外套、黑色裤子和白色运动鞋，戴着白色帽子"
image5 = Image.open('./examples/5.jpg').convert('RGB')
answer5 = "穿着深色外套和黑色裤子的老年女性，身前背着红色的斜挎包"

# image_folder = "./dataset/last_test_query"
# output_jsonl = "./dataset/last_test_query.jsonl"
def filter_path(image_paths, output_jsonl):
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        data = set(item['image_path'] for item in data)
        return [path for path in data if path not in image_paths]
    else:
        return image_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model annotation')
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--output_jsonl', type=str)
    parser.add_argument('--gpu_id', type=str, default=0)
    args = parser.parse_args()
    print(args)
    device = f"cuda:{args.gpu_id}"

    model = AutoModel.from_pretrained('./MiniCPM-V-2_6', trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.to(device=device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6', trust_remote_code=True)

    images_paths = [os.path.join(args.image_folder, img_name) for img_name in os.listdir(args.image_folder) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_paths = filter_path(images_paths, args.output_jsonl)
    for image_path in tqdm(images_paths):
        try:
            image_test = Image.open(image_path).convert('RGB')
            msgs = [
                {'role': 'user', 'content': [image1, question]}, {'role': 'assistant', 'content': [answer1]},
                {'role': 'user', 'content': [image2, question]}, {'role': 'assistant', 'content': [answer2]},
                {'role': 'user', 'content': [image3, question]}, {'role': 'assistant', 'content': [answer3]},
                {'role': 'user', 'content': [image4, question]}, {'role': 'assistant', 'content': [answer4]},
                {'role': 'user', 'content': [image5, question]}, {'role': 'assistant', 'content': [answer5]},
                {'role': 'user', 'content': [image_test, question]}
            ]

            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False
            )
            # print(answer)

            metadata = {
                'image_path': image_path,
                'caption': answer
            }
            with open(args.output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        except BaseException as e:
            print(e)
