from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small

from ...features import VideoFeatures

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


def calc_optical_flow(decoder, features: VideoFeatures):
    num_frames = len(decoder)
    keyframe_indices = list(range(0, num_frames, features.keyframe_interval))

    dataset = RAFT_Dataset(decoder, keyframe_indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    weights = Raft_Large_Weights.DEFAULT
    raft_weight_transforms = weights.transforms()
    raft_model = raft_large(weights=weights, progress=False).cuda().eval()

    for kf_indices, img1_batch, img2_batch in tqdm(dataloader):
        with torch.no_grad():
            img1_batch, img2_batch = raft_weight_transforms(img1_batch.cuda(), img2_batch.cuda())
            flows = raft_model(img1_batch, img2_batch)[-1].cpu().detach().numpy()

            for i in range(len(kf_indices)):
                kf_idx = kf_indices[i].item()
                features.keyframes[kf_idx+1].optical_flow = flows[i].reshape(2, 45, 8, 80, 8).mean(axis=(2, 4))