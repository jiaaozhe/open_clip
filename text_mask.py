import torch
from PIL import Image
import open_clip
import numpy as np

tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')


def random_mask(tokens, mask_token_id, mask_prob=0.15):
    """
    使用numpy进行向量化操作，随机对文本中的token进行mask处理。
    tokens: 已经被tokenized的文本
    mask_token_id: 对应的mask标记
    mask_prob: mask的概率
    """
    # 生成与tokens相同长度的随机掩码数组
    mask = np.random.rand(len(tokens)) < mask_prob

    # 使用掩码替换对应位置的token为mask_token_id
    masked_tokens = np.where(mask, mask_token_id, tokens)

    return masked_tokens.tolist()  # 返回list形式

tokens = tokenizer(["随即掩码测试测试测试"])

# 定义mask的token ID（通常是模型定义的特殊标记，比如'[MASK]'或一个具体的ID）
mask_token_id = tokenizer.tokenizer.mask_token_id

# 随机mask token
masked_tokens = random_mask(np.array(tokens), mask_token_id)

# 解码查看masked后的文本
masked_text = tokenizer.tokenizer.decoder(masked_tokens)
print(masked_text)