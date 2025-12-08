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
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from scenedetect import detect, open_video, AdaptiveDetector, ContentDetector
import open_clip
import importlib
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
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
            # TODO: 改一下，如果不到 8 帧就出错了
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
        size=(360, 640),
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
            clip_img1 = self.decoder[self.clips[clip_id].start_frame]
            clip_img2 = self.decoder[self.clips[clip_id].start_frame + 4]
        else:
            clip_img1 = self.decoder[self.clips[clip_id].end_frame - 4]
            clip_img2 = self.decoder[self.clips[clip_id].end_frame]
        clip_img1 = resizetransforms(clip_img1)
        clip_img2 = resizetransforms(clip_img2)
        return clip_id, clip_type, clip_img1, clip_img2

def calc_optical_flow(decoder, clips):
    weights = Raft_Large_Weights.DEFAULT
    rafttransforms = weights.transforms()

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).cuda()
    model = model.eval()

    data = RAFT_Dataset(decoder, clips)
    dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=1)

    for idxs, types, img1_batch, img2_batch in tqdm(dataloader):
        with torch.no_grad():
            img1_batch, img2_batch = rafttransforms(img1_batch, img2_batch)
            flows = model(img1_batch.cuda(), img2_batch.cuda())[-1].cpu().detach().numpy()
            for i in range(len(idxs)):
                clip_idx = idxs[i].item()
                clip_type = types[i].item()
                if clip_type == 0:
                    clips[clip_idx].start_flow = flows[i].reshape(2, 45, 8, 80, 8).mean(axis=(2, 4))
                elif clip_type == 1:
                    clips[clip_idx].end_flow = flows[i].reshape(2, 45, 8, 80, 8).mean(axis=(2, 4))

def cosine_similarity(vec1: np.array, vec2: np.array):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)

def emd_dist(H1, H2):
    cdf1 = np.cumsum(H1)
    cdf2 = np.cumsum(H2)
    dist = np.sum(np.abs(cdf1 - cdf2))
    return dist / len(H1)

def get_fast_emd(map1, map2):
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

def get_saliency_score(clip1: VideoClip, clip2: VideoClip):
    dist = get_fast_emd(clip1.end_saliency, clip2.start_saliency)
    
    return 1 - dist

def get_embeddings_score(clip1: VideoClip, clip2: VideoClip):
    return cosine_similarity(clip1.embeddings, clip2.embeddings)

def _extract_top_flows(flow_field, top_ratio):
    u = flow_field[0].reshape(-1)
    v = flow_field[1].reshape(-1)
    mag = np.sqrt(u**2 + v**2)
    k = max(1, int(len(mag) * top_ratio))
    idx = np.argpartition(mag, -k)[-k:]
    top_u = u[idx]
    top_v = v[idx]
    return (np.mean(top_u), np.mean(top_v))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_motion_score(clip1, clip2, top_ratio=0.05):    
    (u1, v1) = _extract_top_flows(clip1.end_flow, top_ratio)
    (u2, v2) = _extract_top_flows(clip2.start_flow, top_ratio)
    m1 = np.sqrt(u1**2+v1**2)
    m2 = np.sqrt(u2**2+v2**2)
    mag_sim = 1 - np.tanh(np.abs(m1 - m2) / 80.0)
    dir_sim = np.abs((u1 * u2 + v1 * v2) / (m1 * m2))
    # dir_conf = sigmoid((m1 - 8) / 2) * sigmoid((m2 - 8) / 2)
    dir_conf = (m1 / (m1 + 2)) * (m2 / (m2 + 2))
    return mag_sim * (1 - dir_conf) + dir_sim * dir_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="这是一个评估mp4视频的脚本")

    parser.add_argument("video_dir", type=str, help="视频目录")
    parser.add_argument("--file_name", type=str, default="video.mp4", help="视频文件名称")
    parser.add_argument("--log_name", type=str, default="clips.pkl", help="结果保存/读取路径")
    parser.add_argument("--recalc", action='store_true', help="重新计算所有特征")

    args = parser.parse_args()

    video_dir_path = Path(args.video_dir)
    video_path: Path = "data" / video_dir_path / args.file_name
    pkl_path: Path = "output" / video_dir_path / args.log_name

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
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(clips, f)

    similarities = [(i, get_saliency_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
    print("平均构图相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))

    similarities = [(i, get_embeddings_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
    print("平均语义相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))

    similarities = [(i, get_motion_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
    print("平均运动相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))