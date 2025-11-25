from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch

model = SentenceTransformer('clip-ViT-B-32')


images = [
    # Dog image
    Image.open("1.jpg"),

    # Cat image
    Image.open("2.jpg"),

    # Beach image
    Image.open("3.jpg"),
]

# Map images to the vector space
img_embeddings = model.encode(images)

texts = [
    "A dog in the snow",
    "A cat",
    "a beach with palm trees",
]

text_embeddings = model.encode(texts)

# Compute cosine similarities:
cos_sim = util.cos_sim(text_embeddings, img_embeddings)

for text, scores in zip(texts, cos_sim):
    max_img_idx = torch.argmax(scores)
    print("Text:", text)
    print("Score:", scores[max_img_idx] )
    print("Index:", max_img_idx, "\n")