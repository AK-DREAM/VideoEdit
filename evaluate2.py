import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
from torchcodec.decoders import VideoDecoder
import importlib
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from scenedetect import detect, open_video, AdaptiveDetector, ContentDetector
import open_clip
import importlib
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small
import ot, cv2

class VideoClip:
    # 包含 首位帧显著目标检测结果 平均语义 首位光流
    def __init__(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_saliency = None
        self.end_saliency = None
        self.embeddings = None
        self.start_flow = None
        self.end_flow = None

def decompose_clips(video_path):
    scene_list = detect(str(video_path), AdaptiveDetector(min_scene_len=5))
    print(f"成功切分 {len(scene_list)} 段 clip")
    clips = [VideoClip(scene[0].get_frames(), scene[1].get_frames() - 1) for scene in scene_list]
    return clips

# normalize the predicted SOD probability map
def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

u2nettransform = transforms.Compose([
    transforms.Resize(
        size=(320, 320),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    ),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class U2Net_Dataset(Dataset):
    def __init__(self, decoder: VideoDecoder, clips: list[VideoClip]):
        self.idx = []
        self.type = [] # 0: start 1: end
        self.decoder = decoder
        self.clips = clips
        for i in range(len(clips)):
            self.idx.append(i)
            self.type.append(0)
            self.idx.append(i)
            self.type.append(1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        clip_id = self.idx[idx]
        clip_type = self.type[idx]
        if clip_type == 0:
            clip_frame = self.decoder[self.clips[clip_id].start_frame]
        else:
            clip_frame = self.decoder[self.clips[clip_id].end_frame]
        clip_frame = u2nettransform(clip_frame)
        return clip_id, clip_type, clip_frame

def calc_saliency(decoder, clips):
    U2NET = importlib.import_module("U-2-Net.model").U2NET
    net = U2NET(3,1)
    net.load_state_dict(torch.load("U-2-Net/saved_models/u2net/u2net.pth"))
    net.cuda()

    data = U2Net_Dataset(decoder, clips)
    dataloader = DataLoader(data, batch_size=32, shuffle=False, num_workers=1)

    for idx, types, frames in tqdm(dataloader):
        with torch.no_grad():
            pred = net(frames.cuda())[0][:,0,:,:].cpu().detach().numpy()
            for i in range(len(idx)):
                clip_idx = idx[i].item()
                frame_type = types[i].item()
                if frame_type == 0:
                    clips[clip_idx].start_saliency = normPRED(pred[i]).reshape(64, 5, 64, 5).mean(axis=(1, 3))
                else:
                    clips[clip_idx].end_saliency = normPRED(pred[i]).reshape(64, 5, 64, 5).mean(axis=(1, 3))

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

num_clip_frames = 8

cliptransform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), # 调整大小
    transforms.CenterCrop(224), # 裁剪中心 (如果是长方形图片)
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD) # 归一化
])

class CLIP_Dataset(Dataset):
    def __init__(self, decoder: VideoDecoder, clips: list[VideoClip]):
        self.idx = []
        self.frame_idx = []
        self.decoder = decoder
        for i in range(len(clips)):
            frameidxs = np.linspace(clips[i].start_frame, clips[i].end_frame, num_clip_frames, dtype=int)
            for frameid in frameidxs:
                self.idx.append(i)
                self.frame_idx.append(frameid)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        clip_id = self.idx[idx]
        clip_frame = cliptransform(self.decoder[idx])
        return clip_id, clip_frame


def calc_clip_embedding(decoder, clips):
    model_name = 'ViT-B-32'

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, 
        pretrained="weights/open_clip_model.safetensors", 
        device="cuda"
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    data = CLIP_Dataset(decoder, clips)
    dataloader = DataLoader(data, batch_size=32, shuffle=False, num_workers=1)

    for idxs, frames in tqdm(dataloader):
        with torch.no_grad():
            res = model.encode_image(frames.cuda()).cpu().detach().numpy()
            for i in range(len(idxs)):
                clip_idx = idxs[i].item()
                if clips[clip_idx].embeddings is None:
                    clips[clip_idx].embeddings = res[i] / num_clip_frames
                else:
                    clips[clip_idx].embeddings += res[i] / num_clip_frames


resizetransforms = transforms.Resize(
        size=(320, 320),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )

class RAFT_Dataset(Dataset):
    def __init__(self, decoder: VideoDecoder, clips: list[VideoClip]):
        self.idx = []
        self.type = []
        self.decoder = decoder
        self.clips = clips
        for i in range(len(clips)):
            self.idx.append(i)
            self.type.append(0)
            self.idx.append(i)
            self.type.append(1)
    
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        clip_id = self.idx[idx]
        clip_type = self.type[idx]
        if clip_type == 0:
            start_idx = self.clips[clip_id].start_frame
            frame_indices = [start_idx + i for i in range(5)]
        else:
            end_idx = self.clips[clip_id].end_frame
            frame_indices = [end_idx - 4 + i for i in range(5)]
        frames = [resizetransforms(self.decoder[f_idx]) for f_idx in frame_indices]
        img1_frames = []
        img2_frames = []
        for i in range(4):
            img1_frames.append(frames[i])
            img2_frames.append(frames[i+1])
        img1_tensor = torch.stack(img1_frames, dim=0)
        img2_tensor = torch.stack(img2_frames, dim=0)
        return clip_id, clip_type, img1_tensor, img2_tensor

def calc_optical_flow(decoder, clips):
    weights = Raft_Small_Weights.DEFAULT
    rafttransforms = weights.transforms()

    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).cuda()
    model = model.eval()

    data = RAFT_Dataset(decoder, clips)
    dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=1)

    for idxs, types, img1_batch, img2_batch in tqdm(dataloader):
        with torch.no_grad():
            B, K, C, H, W = img1_batch.shape
            img1_flat = img1_batch.view(B * K, C, H, W)
            img2_flat = img2_batch.view(B * K, C, H, W)
            img1_batch, img2_batch = rafttransforms(img1_flat, img2_flat)
            flows = model(img1_batch.cuda(), img2_batch.cuda())[-1].reshape(B * K, 2, 64, 5, 64, 5).mean(axis=(3, 5))
            flows = flows.view(B, K, 2, flows.shape[-2], flows.shape[-1])
            flows = flows.permute(0, 2, 1, 3, 4).cpu().detach().numpy()
            for i in range(len(idxs)):
                clip_idx = idxs[i].item()
                clip_type = types[i].item()
                if clip_type == 0:
                    clips[clip_idx].start_flow = flows[i]
                elif clip_type == 1:
                    clips[clip_idx].end_flow = flows[i]

def cosine_similarity(vec1: np.array, vec2: np.array):
    """计算两个向量的余弦相似度"""
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 避免除以零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)

pot_size = 32
y, x = np.meshgrid(np.arange(pot_size), np.arange(pot_size))
coords = np.stack((x.flatten(), y.flatten()), axis=1)
M = ot.dist(coords, coords, metric='euclidean')

def get_pot_emd(map1, map2, reg=0.1):
    m1 = cv2.resize(map1, (pot_size, pot_size), interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(map2, (pot_size, pot_size), interpolation=cv2.INTER_AREA)
    
    a = m1.flatten() / (np.sum(m1) + 1e-10)
    b = m2.flatten() / (np.sum(m2) + 1e-10)
    
    dist = ot.sinkhorn2(a, b, M, reg=reg)
    
    # 归一化
    max_dist = np.sqrt(pot_size**2 + pot_size**2)
    return dist / max_dist

def emd_dist(H1, H2):
    cdf1 = np.cumsum(H1)
    cdf2 = np.cumsum(H2)
    dist = np.sum(np.abs(cdf1 - cdf2))
    return dist / len(H1)

def get_fast_emd(map1, map2):
    """
    极速投影 Wasserstein 距离 (利用 CDF 性质)
    时间复杂度: O(H + W) -> 几乎瞬间完成
    """
    m1 = map1 / (np.sum(map1) + 1e-10)
    m2 = map2 / (np.sum(map2) + 1e-10)

    x_pdf1 = np.sum(m1, axis=0)
    x_pdf2 = np.sum(m2, axis=0)
    
    y_pdf1 = np.sum(m1, axis=1)
    y_pdf2 = np.sum(m2, axis=1)

    dist_x = emd_dist(x_pdf1, x_pdf2)
    dist_y = emd_dist(y_pdf1, y_pdf2)

    norm_dist = np.sqrt((dist_x)**2 + (dist_y)**2)
    
    return norm_dist

def get_saliency_score(clip1: VideoClip, clip2: VideoClip, mode = 0, sigma = 0.3):
    dist = 0.0
    if mode == 0:
        dist = get_fast_emd(clip1.end_saliency, clip2.start_saliency)
    else:
        dist = get_pot_emd(clip1.end_saliency, clip2.start_saliency)
    
    return np.exp(-dist*dist / (2*sigma*sigma))

def get_embeddings_score(clip1: VideoClip, clip2: VideoClip):
    return cosine_similarity(clip1.embeddings, clip2.embeddings)

def _get_hoof_features(flow_field: np.ndarray, noise_threshold, static_ratio_threshold, bins: int = 36):
    """
    计算前景运动直方图 H 和运动像素占比 alpha。
    """
    u = flow_field[0]
    v = flow_field[1]
    # 1. 计算幅度和角度
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u) # 角度在 -pi 到 pi
    # 2. 生成运动掩码 (前景分离)
    motion_mask = mag > noise_threshold
    total_pixels = flow_field.shape[1] * flow_field.shape[2]
    motion_ratio = np.sum(motion_mask) / total_pixels

    if motion_ratio < static_ratio_threshold:
        return np.zeros(bins), motion_ratio, 0.0
    # 4. 只处理运动像素 (Foreground-Only)
    valid_angles = angle[motion_mask]
    valid_mags = mag[motion_mask]
    # 5. 计算 Bin 索引
    angle_deg = (np.degrees(valid_angles) % 180 + 180) % 180 # 转换为 0-360 度
    bin_idx = (angle_deg / (180 / bins)).astype(int) % bins
    # 6. 计算 HOOF (使用 Magnitude 加权)
    hist = np.bincount(bin_idx, weights=valid_mags, minlength=bins)
    # 7. 归一化 (L1 范数)
    hist_norm = hist / (np.sum(hist) + 1e-8)
    
    return hist_norm, motion_ratio, np.mean(valid_mags)

def circular_emd_dist(H1, H2):
    min_emd = float('inf')
    N = len(H1)
    for k in range(N):
        H1_shifted = np.roll(H1, k)
        H2_shifted = np.roll(H2, k)
        current_emd = emd_dist(H1_shifted, H2_shifted)
        if current_emd < min_emd:
            min_emd = current_emd
    return min_emd

def angular_sim(hoof1, hoof2, sigma):
    """角度感知高斯核"""
    n_bins = len(hoof1)
    kernel_matrix = np.zeros((n_bins, n_bins))
    
    for i in range(n_bins):
        for j in range(n_bins):
            angle_diff = min(
                abs(i - j),
                n_bins - abs(i - j)
            ) / n_bins
            kernel_matrix[i,j] = np.exp(-angle_diff**2 / (2 * sigma**2))

    value = hoof1 @ kernel_matrix @ hoof2.T
    norm1 = np.sqrt(hoof1 @ kernel_matrix @ hoof1.T)
    norm2 = np.sqrt(hoof2 @ kernel_matrix @ hoof2.T)
    
    return value / (norm1 * norm2)

def get_motion_score(clip1, clip2, noise_threshold=2.0, static_ratio_threshold=0.02, sigma=0.3):    
    # 1. 提取 HOOF 特征和运动占比
    H1, alpha1, mean_mag1 = _get_hoof_features(clip1.end_flow, noise_threshold, static_ratio_threshold)
    H2, alpha2, mean_mag2 = _get_hoof_features(clip2.start_flow, noise_threshold, static_ratio_threshold)
    
    # 2. 静态状态判断 (Decoupling Logic)
    is_static1 = (alpha1 < static_ratio_threshold)
    is_static2 = (alpha2 < static_ratio_threshold)
    
    if is_static1 and is_static2:
        return 1.0
    if is_static1 != is_static2:
        if is_static1:
            return np.exp(-mean_mag2/5)
        else:
            return np.exp(-mean_mag1/5)

    # dist = circular_emd_dist(H1, H2)
    # return np.exp(-dist*dist / (2*sigma*sigma))
    return angular_sim(H1, H2, sigma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="这是一个评估mp4视频的脚本")

    parser.add_argument("video_dir", type=str, help="视频目录")
    parser.add_argument("--file_name", type=str, default="video.mp4", help="视频文件名称")
    parser.add_argument("--log_name", type=str, default="clips.pkl", help="结果保存/读取路径")
    parser.add_argument("--recalc", action='store_true', help="重新计算所有特征")

    args = parser.parse_args()

    video_dir_path = Path(args.video_dir)
    video_path: Path = video_dir_path / args.file_name
    pkl_path: Path = video_dir_path / args.log_name

    assert(video_path.exists())

    decoder = VideoDecoder(video_path)

    print("成功读取视频")

    if pkl_path.exists() and not args.recalc:
        with open(pkl_path, 'rb') as f:
            clips = pickle.load(f)
            print(f"成功读取 {pkl_path}")
    else:
        print(f"无法读取计算结果 {pkl_path}")

        print("开始切分视频")
        clips = decompose_clips(video_path)

        print("开始显著性计算")
        calc_saliency(decoder, clips)

        print("开始 CLIP 计算")
        calc_clip_embedding(decoder, clips)

        print("开始光流计算")
        calc_optical_flow(decoder, clips)

        print(f"计算完成，保存到 {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(clips, f)

    similarities = [(i, get_saliency_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
    print("平均构图相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))

    similarities = [(i, get_embeddings_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
    print("平均语义相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))

    similarities = [(i, get_motion_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
    print("平均运动相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))