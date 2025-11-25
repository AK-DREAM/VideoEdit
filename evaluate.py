import argparse

parser = argparse.ArgumentParser(description="这是一个评估mp4视频的脚本")

parser.add_argument("video_dir", type=str, help="视频目录")
parser.add_argument("--file_name", type=str, default="video.mp4", help="视频文件名称")
parser.add_argument("--log_name", type=str, default="clips.pkl", help="结果保存/读取路径")

args = parser.parse_args()

from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
from torchcodec.decoders import VideoDecoder
import importlib


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



video_dir_path = Path(args.video_dir)
video_path: Path = video_dir_path / args.file_name
pkl_path: Path = video_dir_path / args.log_name

assert(video_path.exists())

decoder = VideoDecoder(video_path)

print("成功读取视频")

try:
    assert(pkl_path.exists())
    with open(pkl_path, 'rb') as f:
        clips = pickle.load(f)
        print(f"成功读取 {pkl_path}")
except:
    print(f"无法读取计算结果 {pkl_path}")
    
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torch.utils.data import Dataset, DataLoader
    from scenedetect import detect, AdaptiveDetector
    import open_clip

    scene_list = detect(str(video_path), AdaptiveDetector(min_scene_len = 5, adaptive_threshold = 10))
    print(f"成功切分 {len(scene_list)} 段 clip")
    clips = [VideoClip(scene[0].get_frames(), scene[1].get_frames() - 1) for scene in scene_list]

    print("开始显著性计算")
    import importlib
    U2NET = importlib.import_module("U-2-Net.model").U2NET
    net = U2NET(3,1)
    net.load_state_dict(torch.load("U-2-Net/saved_models/u2net/u2net.pth"))
    net.cuda()


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


    print("开始语义计算")
    for clip in clips:
        clip.embeddings = None
    num_frames = 8


    model_name = 'ViT-B-32'

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, 
        pretrained="weights/open_clip_model.safetensors", 
        device="cuda"
    )
    tokenizer = open_clip.get_tokenizer(model_name)


    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

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
                frameidxs = np.linspace(clips[i].start_frame, clips[i].end_frame, num_frames, dtype=int)
                for frameid in frameidxs:
                    self.idx.append(i)
                    self.frame_idx.append(frameid)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, idx):
            clip_id = self.idx[idx]
            clip_frame = cliptransform(self.decoder[idx])
            return clip_id, clip_frame

    data = CLIP_Dataset(decoder, clips)
    dataloader = DataLoader(data, batch_size=32, shuffle=False, num_workers=1)

    for idxs, frames in tqdm(dataloader):
        with torch.no_grad():
            res = model.encode_image(frames.cuda()).cpu().detach().numpy()
            for i in range(len(idxs)):
                clip_idx = idxs[i].item()
                if clips[clip_idx].embeddings is None:
                    clips[clip_idx].embeddings = res[i] / num_frames
                else:
                    clips[clip_idx].embeddings += res[i] / num_frames

    print("开始光流计算")
    from torchvision.models.optical_flow import Raft_Large_Weights

    weights = Raft_Large_Weights.DEFAULT
    rafttransforms = weights.transforms()

    from torchvision.models.optical_flow import raft_large

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).cuda()
    model = model.eval()

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
                clip_img1 = self.decoder[self.clips[clip_id].start_frame]
                clip_img2 = self.decoder[self.clips[clip_id].start_frame + 1]
            else:
                clip_img1 = self.decoder[self.clips[clip_id].end_frame - 1]
                clip_img2 = self.decoder[self.clips[clip_id].end_frame]
            clip_img1 = resizetransforms(clip_img1)
            clip_img2 = resizetransforms(clip_img2)
            return clip_id, clip_type, clip_img1, clip_img2


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
                    clips[clip_idx].start_flow = flows[i].reshape(2, 64, 5, 64, 5).mean(axis=(2, 4))
                elif clip_type == 1:
                    clips[clip_idx].end_flow = flows[i].reshape(2, 64, 5, 64, 5).mean(axis=(2, 4))
    print(f"计算完成，保存到 {pkl_path}")
    with open(pkl_path, "wb") as f:
        pickle.dump(clips, f)



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

    x_cdf1 = np.cumsum(x_pdf1)
    x_cdf2 = np.cumsum(x_pdf2)
    
    y_cdf1 = np.cumsum(y_pdf1)
    y_cdf2 = np.cumsum(y_pdf2)

    dist_x = np.sum(np.abs(x_cdf1 - x_cdf2))
    dist_y = np.sum(np.abs(y_cdf1 - y_cdf2))

    h, w = map1.shape
    norm_dist = np.sqrt((dist_x / w)**2 + (dist_y / h)**2)
    
    return norm_dist

def get_saliency_score(clip1: VideoClip, clip2: VideoClip):
    return 1 - get_fast_emd(clip1.end_saliency, clip2.start_saliency)

def get_embeddings_score(clip1: VideoClip, clip2: VideoClip):
    return cosine_similarity(clip1.embeddings, clip2.embeddings)

def get_flow_score(clip1: VideoClip, clip2: VideoClip, alpha = 0.7):
    def get_vec(f, s):
        w = s / (np.sum(s) + 1e-8)
        return np.sum(f.astype(np.float32) * w[np.newaxis, :, :], axis=(1, 2))
    v1 = get_vec(clip1.end_flow, clip1.end_saliency)
    v2 = get_vec(clip2.start_flow, clip2.start_saliency)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    dir_score = max(0.0, cosine)

    if norm1 < 2 or norm2 < 2:
        dir_score = 1.0

    speed_diff = abs(norm1 - norm2) / max(norm1, norm2)
    spd_score = 1.0 - speed_diff

    final_score = dir_score * spd_score
    
    return float(final_score)

similarities = [(i, get_saliency_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]

print("平均构图相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))

similarities = [(i, get_embeddings_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
print("平均语义相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))

similarities = [(i, get_flow_score(clips[i], clips[i + 1])) for i in range(len(clips) - 1)]
print("平均运动相似得分：{}".format(sum([x[1] for x in similarities]) / len(similarities)))