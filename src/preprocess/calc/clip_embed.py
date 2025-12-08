from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import open_clip

from ...features import VideoFeatures

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD  = (0.26862954, 0.26130258, 0.27577711)

cliptransform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
])

class CLIP_Dataset(Dataset):
    def __init__(self, decoder, keyframe_indices: list[int]):
        self.decoder = decoder
        self.keyframe_indices = keyframe_indices

    def __len__(self):
        return len(self.keyframe_indices)

    def __getitem__(self, idx):
        frame_idx = self.keyframe_indices[idx]
        frame = cliptransform(self.decoder[frame_idx])      
        return idx, frame

def calc_clip_embedding(decoder, features: VideoFeatures):
    device = "cuda"

    model_name = "ViT-B-32"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained="weights/open_clip_model.safetensors",
        device=device
    )
    model.eval()

    num_frames = len(decoder)
    keyframe_indices = list(range(0, num_frames, features.keyframe_interval))

    dataset = CLIP_Dataset(decoder, keyframe_indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    for kf_indices, frames in tqdm(dataloader):
        with torch.no_grad():
            embeds = model.encode_image(frames.to(device)).cpu().numpy()

        for i in range(len(kf_indices)):
            kf_idx = kf_indices[i].item()
            features.keyframes[kf_idx].clip_embed = embeds[i]