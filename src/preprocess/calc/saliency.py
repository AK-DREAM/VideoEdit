from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import importlib

from ...features import VideoFeatures

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
        # self.frames = decoder.get_frames_at(keyframe_indices).data

    def __len__(self):
        return len(self.keyframe_indices)

    def __getitem__(self, idx):
        frame_idx = self.keyframe_indices[idx]
        frame = self.decoder[frame_idx]         # HWC RGB 0–255
        frame = u2nettransform(frame)           # CHW
        return idx, frame

def calc_saliency(decoder, features: VideoFeatures):
    num_frames = len(decoder)
    keyframe_indices = list(range(0, num_frames, features.keyframe_interval))

    dataset = U2Net_Dataset(decoder, keyframe_indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    U2NET = importlib.import_module("U-2-Net.model").U2NET
    net = U2NET(3, 1)
    net.load_state_dict(torch.load("U-2-Net/saved_models/u2net/u2net.pth"))
    net.cuda()
    net.eval()

    for kf_indices, frames in tqdm(dataloader):
        with torch.no_grad():
            preds = net(frames.cuda())[0][:,0,:,:].cpu().detach().numpy()
            for i in range(len(kf_indices)):
                kf_idx = kf_indices[i].item()
                sal = normPRED(preds[i]).reshape(64, 5, 64, 5).mean(axis=(1, 3))
                features.keyframes[kf_idx].saliency = sal