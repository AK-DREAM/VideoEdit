from torchvision import transforms
import math
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
from tqdm import tqdm
import open_clip

from ...features import VideoFeatures
from ...utils.decoder import PaddedVideoDecoder

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
    
class CLIP_IterableDataset(IterableDataset):
    def __init__(self, video_path: str, keyframe_indices: list[int]):
        self.video_path = video_path
        self.keyframe_indices = keyframe_indices

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        decoder = PaddedVideoDecoder(self.video_path, device='cpu')
        total_tasks = len(self.keyframe_indices)
        
        if worker_info is None:
            my_indices = list(range(0, total_tasks))
            my_kf_indices = self.keyframe_indices
        else:
            per_worker = int(math.ceil(total_tasks / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, total_tasks)
            
            if start >= total_tasks:
                my_indices = []
                my_kf_indices = []
            else:
                my_indices = list(range(start, end))
                my_kf_indices = self.keyframe_indices[start:end]

        for idx, kf_idx in zip(my_indices, my_kf_indices):
            frame = decoder[kf_idx]      
            frame = cliptransform(frame) 
            yield idx, frame

def calc_clip_embedding(video_path, num_frames, features: VideoFeatures, device = "cuda:0"):
    model_name = "ViT-B-32"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained="weights/open_clip_model.safetensors",
        device=device
    )
    model.eval()

    keyframe_indices = list(range(0, num_frames, features.keyframe_interval))

    dataset = CLIP_IterableDataset(video_path, keyframe_indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    for indices, frames in tqdm(dataloader):
        with torch.no_grad():
            embeds = model.encode_image(frames.to(device)).cpu().numpy()

        for i in range(len(indices)):
            idx = indices[i].item()
            features.keyframes[idx].clip_embed = embeds[i]