import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import concurrent.futures
import hashlib
import json
from tqdm import tqdm
import logging

logger = logging.getLogger('planner_summary')

from .editing_utils import load_video_footages
from .llm_interface import chat_with_llm, Message
from ..features import VideoFeatures
from ..utils.path import get_output_dir

summary_dir = get_output_dir() / "summary"
os.makedirs(summary_dir, exist_ok=True)


MAX_WORKERS = 8

def pil_to_base64(image, format="JPEG", quality=85):
    """
    将 PIL 图片对象转换为 base64 字符串
    :param image: PIL.Image 对象
    :param format: 保存格式（JPEG 可减小体积，PNG 质量更高）
    :param quality: JPEG 压缩质量 (1-100)
    """
    buffered = BytesIO()
    # 将图片保存到内存缓冲区
    image.save(buffered, format=format, quality=quality)
    # 获取字节数据并进行 base64 编码
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

class OpenCVCinematicIterator:
    def __init__(self, video_paths: dict[str, Path], video_features: dict[str, VideoFeatures], batch_size=16, slot_width=448, slot_height=252):
        """
        :param slot_width: 每个小格子的宽度 (默认 448)
        :param slot_height: 每个小格子的高度 (建议设为 16:9 比例，如 252)
        """
        self.video_paths = video_paths
        self.batch_size = batch_size
        self.per_batch = batch_size
        self.slot_size = (slot_width, slot_height)
        
        # 摊平所有 shot
        self.flat_shots = []
        for vid, feature in video_features.items():
            if vid in video_paths:
                for shot_idx, shot in enumerate(feature.shots):
                    self.flat_shots.append({
                        'vid': vid,
                        'path': str(video_paths[vid]),
                        'start': shot['start'],
                        'end': shot['end'],
                        'idx': shot_idx
                    })
        
    def _get_letterboxed_frame(self, frame, target_size):
        """保持比例缩放并填充黑边"""
        t_w, t_h = target_size
        h, w = frame.shape[:2]
        
        # 计算缩放比例
        scale = min(t_w / w, t_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建黑色画布并居中粘贴
        canvas = np.zeros((t_h, t_w, 3), dtype=np.uint8)
        x_offset = (t_w - new_w) // 2
        y_offset = (t_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    def __iter__(self):
        cap = None
        current_path = None
        
        # 按 batch 处理
        for i in range(0, len(self.flat_shots), self.per_batch):
            batch = self.flat_shots[i : i + self.per_batch]
            processed_frames = []
            
            for shot in batch:
                # 优化：避免重复打开同一个视频文件
                if shot['path'] != current_path:
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(shot['path'])
                    current_path = shot['path']
                
                # 取中间帧
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_pos = (shot['start'] + shot['end']) // 2
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    # 转换颜色空间 (OpenCV 是 BGR, VLM 通常需要 RGB)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frames.append(self._get_letterboxed_frame(frame, self.slot_size))
                else:
                    processed_frames.append(np.zeros((self.slot_size[1], self.slot_size[0], 3), dtype=np.uint8))

            yield processed_frames, batch

        if cap is not None:
            cap.release()

def _get_batch_summary_prompt(genre: str, batch_size: int):
    return f"""
### Task
Act as a Mashup Editor. Scan these {batch_size} shots from a {genre} library to identify high-value editing assets.

### Instructions
1. **Identify "Edit Hooks"**: Look for shots with strong motion (zooms, pans), high emotional intensity (eyes, screams), or striking aesthetics (neon, shadows).
2. **Action & Rhythm**: Focus on the kinetic energy. Is it fast-paced, slow-motion, or rhythmic?
3. **Ignore Metadata**: Strictly ignore UI overlays (#1-#16) and grid lines. Focus on the raw footage.
4. **Brevity**: Use punchy, descriptive tags that an editor can search for.

### Output Format
- **Visual Vibe**: [3 keywords for the aesthetic, e.g., Glitchy, Epic, Gritty]
- **Key Assets**: [Main subjects/actions, e.g., Car drifting, Sword clash, Rain-slicked face]
- **Mashup Potential**: [Describe the best 1-2 shots for a music video or trailer—focus on "The Money Shot"]
- **Rhythm**: [Describe the motion speed, e.g., Fast cuts, Slow-burn, Explosive]
"""

def _get_aggregation_prompt(full_corpus: str, total_shots: int, genre: str):
    return f"""
### Role
You are a Lead Mashup Strategist. Synthesize the provided logs from {total_shots} shots into a professional "Footage Analysis Report" for creative editing.

### Source Data
Below is the raw chronological narrative log of the footage from the {genre} project:
---
{full_corpus}
---

### Objective
Synthesize the data above into a strategic blueprint for a mashup video.
**Constraint**: Eliminate all technical indices and segment IDs. The final report must flow like a creative pitch.

### Output Structure

**Footage Analysis Report: [Project Title]**

**1. Visual Identity & Aesthetic**
A paragraph describing the "Cinematic Look" (color, lighting, texture) and how it dictates the mashup's vibe.

**2. Global Keywords**
A comma-separated list of 15 high-impact keywords for asset retrieval.

**3. Iconic Scene Library (Archetypes)**
Group the most memorable footage into 4-6 categories. For each:
- **[Set Title]**: (e.g., The Kinetic Chase, The Neon Void)
- **Edit-Value**: Why these shots are perfect for a rhythmic or emotional hook.
- **Standout Moments**: Describe 2-3 specific, high-impact highlights found in the logs.

**4. Visual Anchors & Motifs**
Identify recurring symbols (e.g., a specific prop, color shift) that can serve as recurring transitions or thematic anchors.

### Style Guide
- Professional, high-energy, and editor-focused.
- Use cinematic terminology (e.g., "match-cuts," "low-key lighting").
"""
def process_task(batch_idx, batch_data, genre):
    """处理一个 batch 的多张图并请求 LLM"""
    frames, metadata = batch_data
    try:
        msg = Message(role="user")
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            b64_str = pil_to_base64(frame_pil, quality=80)
            msg = msg.add_image_base64(b64_str)

        msg = msg.add_text(_get_batch_summary_prompt(genre=genre, batch_size=len(frames)))

        # 调用你的 API 接口
        res = chat_with_llm([msg.to_dict()])
        
        return {"id": batch_idx, "txt": res.strip()}
    except Exception as e:
        logger.error(f"Error processing batch {batch_idx}: {e}")
        return {"id": batch_idx, "error": str(e)}

def run_pipeline(video_paths, video_features, totshot, genre: str, max_workers: int=MAX_WORKERS):
    iterator = OpenCVCinematicIterator(video_paths, video_features)
    results = []

    # tqdm 进度条
    pbar = tqdm(total=(totshot + 15) // 16, desc="🎞️ Analyzing Footage")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {}

        for i, data in enumerate(iterator):
            future = executor.submit(process_task, i, data, genre)
            future_to_id[future] = i

            if len(future_to_id) >= max_workers * 2:
                done, _ = concurrent.futures.wait(
                    future_to_id.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f_done in done:
                    result = f_done.result()
                    results.append(result)
                    del future_to_id[f_done]
                    pbar.update(1)

        for f_done in concurrent.futures.as_completed(future_to_id):
            result = f_done.result()
            results.append(result)
            pbar.update(1)

    pbar.close()
    
    return results

def get_summary(csv_path: Path, data_root: Path, output_root: Path, summary_root: Path = summary_dir, genre: str = "Video Collection", use_cache: bool = True) -> str:
    """
    生成视频素材的 Planner Summary
    :param csv_path: 包含视频路径的 CSV 文件
    :param data_root: 视频文件的根目录
    :param output_root: 输出目录
    :param summary_root: Summary 缓存目录
    :param genre: 视频类型
    :param use_cache: 是否使用缓存
    """
    summary_root.mkdir(parents=True, exist_ok=True)

    # 计算 CSV 内容 + genre 的 Hash，用于缓存校验
    csv_bytes = csv_path.read_bytes()
    cache_hash = hashlib.md5(csv_bytes + genre.encode("utf-8")).hexdigest()
    cache_path = summary_root / f"{cache_hash}.json"

    if cache_path.exists() and use_cache:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.info(f"Using cached summary from {cache_path}")
            return cached["summary"]
        except json.JSONDecodeError:
            pass
    logger.info("Generating new summary...")
    # 重新生成 summary
    video_paths, video_features = load_video_footages(csv_path, data_root, output_root)
    totshot = sum(len(video_feature.shots) for video_feature in video_features.values())

    results = run_pipeline(video_paths, video_features, totshot, genre)

    # 按 id 排序并拼接成语料
    results = sorted(results, key=lambda x: x.get("id", 0))
    segments = []
    for item in results:
        if "txt" in item:
            segments.append(item['txt'])

    full_corpus = "\n".join(segments)
    aggregation_prompt = _get_aggregation_prompt(full_corpus, total_shots=totshot, genre=genre)
    summary = chat_with_llm([
        Message(role="user").add_text(aggregation_prompt).to_dict()
    ])

    cache_payload = {
        "csv_path": str(csv_path),
        "genre": genre,
        "cache_hash": cache_hash,
        "segments": results,
        "summary": summary
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary
    
