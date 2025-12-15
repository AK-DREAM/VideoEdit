from torchvision import transforms
import numpy as np
import math
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
from tqdm import tqdm
import importlib

from ...features import VideoFeatures
from ...utils.decoder import PaddedVideoDecoder

def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

u2nettransform = transforms.Compose([
    transforms.Resize(
        size=(320, 320),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True
    ),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class U2Net_Dataset(Dataset):
    def __init__(self, decoder, keyframe_indices: list[int]):
        self.decoder = decoder
        self.keyframe_indices = keyframe_indices

    def __len__(self):
        return len(self.keyframe_indices)

    def __getitem__(self, idx):
        frame_idx = self.keyframe_indices[idx]
        frame = u2nettransform(self.decoder[frame_idx])  
        return idx, frame

class U2Net_IterableDataset(IterableDataset):
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
            frame = u2nettransform(frame) 
            yield idx, frame

def calc_saliency(video_path, num_frames, features: VideoFeatures, device="cuda:0"):
    keyframe_indices = list(range(0, num_frames, features.keyframe_interval))
    total_tasks = len(keyframe_indices)
    batch_size = 32

    dataset = U2Net_IterableDataset(video_path, keyframe_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    U2NET = importlib.import_module("U-2-Net.model").U2NET
    net = U2NET(3, 1)
    net.load_state_dict(torch.load("U-2-Net/saved_models/u2net/u2net.pth"))
    net.to(device).eval()

    for indices, frames in tqdm(dataloader, total=math.ceil(total_tasks/batch_size)):
        with torch.no_grad():
            preds = net(frames.to(device))[0][:,0,:,:].cpu().detach().numpy()
            for i in range(len(indices)):
                idx = indices[i].item()
                sal = normPRED(preds[i]).reshape(64, 5, 64, 5).mean(axis=(1, 3))
                features.keyframes[idx].saliency = sal