from torchvision import transforms
import math
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
from tqdm import tqdm
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small

from ...features import VideoFeatures
from ...utils.decoder import PaddedVideoDecoder

rafttransform = transforms.Compose([
    transforms.Resize(
        size=(360, 640),
        interpolation=transforms.InterpolationMode.BILINEAR
    )
])

class RAFT_Dataset(Dataset):
    def __init__(self, decoder, keyframe_indices: list[int]):
        self.decoder = decoder
        self.keyframe_indices = keyframe_indices
        self.last_frame = rafttransform(self.decoder[self.keyframe_indices[0]])

    def __len__(self):
        return len(self.keyframe_indices) - 1

    def __getitem__(self, idx):
        frame_idx = self.keyframe_indices[idx+1]
        img1 = self.last_frame
        img2 = rafttransform(self.decoder[frame_idx])
        self.last_frame = img2

        return idx, img1, img2

class RAFT_IterableDataset(IterableDataset):
    def __init__(self, video_path: str, keyframe_indices: list[int]):
        self.video_path = video_path
        self.keyframe_indices = keyframe_indices

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        decoder = PaddedVideoDecoder(self.video_path, device='cpu')
        total_tasks = len(self.keyframe_indices) - 1
        
        if worker_info is None:
            my_indices = list(range(1, total_tasks+1))
            my_kf_indices = self.keyframe_indices[1:]
            last_frame = rafttransform(decoder[self.keyframe_indices[0]])
        else:
            per_worker = int(math.ceil(total_tasks / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, total_tasks)
            
            if start >= total_tasks:
                my_indices = []
                my_kf_indices = []
                last_frame = None
            else:
                my_indices = list(range(start+1, end+1))
                my_kf_indices = self.keyframe_indices[start+1:end+1]
                last_frame = rafttransform(decoder[self.keyframe_indices[start]])

        for idx, kf_idx in zip(my_indices, my_kf_indices):
            img1 = last_frame
            img2 = rafttransform(decoder[kf_idx]) 
            last_frame = img2
            yield idx, img1, img2

def calc_optical_flow(video_path, num_frames, features: VideoFeatures, device="cuda:0"):
    keyframe_indices = list(range(0, num_frames, features.keyframe_interval))

    dataset = RAFT_IterableDataset(video_path, keyframe_indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    weights = Raft_Large_Weights.DEFAULT
    raft_weight_transforms = weights.transforms()
    raft_model = raft_large(weights=weights, progress=False).to(device=device).eval()

    for indices, img1_batch, img2_batch in tqdm(dataloader):
        with torch.no_grad():
            img1_batch, img2_batch = raft_weight_transforms(img1_batch.to(device), img2_batch.to(device))
            flows = raft_model(img1_batch, img2_batch)[-1].cpu().detach().numpy()

            for i in range(len(indices)):
                idx = indices[i].item()
                features.keyframes[idx].optical_flow = flows[i].reshape(2, 45, 8, 80, 8).mean(axis=(2, 4))