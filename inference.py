import torch
from PIL import Image
import open_clip

# frozen_laion5b_s13b_b90k
model, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained='pretrained_models/openclip-xlm-vit-h14.bin')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')

image = preprocess(Image.open("image.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]