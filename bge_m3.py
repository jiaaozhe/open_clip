from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["穿黑色衣服的人", "穿浅色衣服的人", "穿白色衣服的男人", "穿白色衣服的人"]

device = 'cuda:1'
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('./pretrained_models/bge-m3')
model = AutoModel.from_pretrained('./pretrained_models/bge-m3', device_map=device)
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input.to(device)
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
# print("Sentence embeddings:", sentence_embeddings)
similarity = sentence_embeddings @ sentence_embeddings.T
print(similarity.dtype)
print(similarity)
