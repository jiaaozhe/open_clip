import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer 

model, preprocess = create_model_from_pretrained('ViT-H-14-378-quickgelu', pretrained='dfn5b')
tokenizer = get_tokenizer('ViT-H-14')

image = Image.open('./src/dataset/cars/1.jpg')
image = preprocess(image).unsqueeze(0)

labels_list = ["a car", "a black car", "a black SUV", "a white car"]
text = tokenizer(labels_list, context_length=model.context_length)

with torch.no_grad(), torch.amp.autocast('cuda'):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_similarity = image_features @ text_features.T

print("similarity: ", text_similarity)  # prints: [[1., 0., 0.]]