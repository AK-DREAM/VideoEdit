import allin1
import numpy as np
import librosa
import json

# 定义标签比例，直接写死在代码中
LABEL_PROPORTIONS = {
    'solo': 1.0,     # [最激烈] 独奏/高光时刻：100%保留，每一拍都可能是关键动作
    'chorus': 1.0,   # [最激烈] 副歌：100%保留，视觉密度最大化
    
    'intro': 0.5,    # [推进] 前奏：50%保留 (每2拍取1)，建立节奏感
    'verse': 0.5,    # [叙事] 主歌：50%保留，留出空间看清画面内容
    'bridge': 0.5,   # [衔接] 桥段：50%保留，维持流动性
    'inst': 0.5,     # [律动] 器乐：50%保留，跟随主要鼓点
    
    'start': 0.25,   # [铺垫] 起始：25%保留 (每4拍取1)，长镜头引入
    'break': 0.25,   # [呼吸] 停顿/变奏：25%保留，给观众喘息空间
    'outro': 0.25,   # [余韵] 尾奏：25%保留，情绪渐弱
    
    'end': 0.125     # [收尾] 结束：12.5%保留 (每8拍取1)，通常只需要最后的重音
}

def extract_beats(video_path):
    """
    提取节拍点并根据 segment.label 判断保留比例。

    参数：
        video_path (str): 视频文件路径。

    返回：
        list: 每段的节拍点。
    """
    # 使用 allin1 提取节拍点
    result = allin1.analyze(video_path)
    beats = result.beats

    # 使用 librosa 计算每个节拍点的强度
    y, sr = librosa.load(video_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strengths = [onset_env[int(beat * sr / 512)] for beat in beats]

    # 根据 result.segments 和 LABEL_PROPORTIONS 筛选节拍点
    segments = result.segments
    segmented_beats = []
    for segment in segments:
        label = segment.label
        proportion = LABEL_PROPORTIONS.get(label, 1.0)  # 默认保留全部节拍点
        print(f"Processing segment: start={segment.start}, end={segment.end}, label={segment.label} -> proportion={proportion}")
        segment_beats = [(beat, strength) for beat, strength in zip(beats, beat_strengths) if segment.start <= beat <= segment.end]
        segment_beats.sort(key=lambda x: x[1], reverse=True)  # 按强度降序排序
        top_beats = segment_beats[:int(len(segment_beats) * proportion)]  # 取前 proportion 的节拍点
        filtered_beats = [(round(beat - 0.1 * 60 / result.bpm, 2), label) for beat, _ in top_beats]
        segmented_beats.append(filtered_beats)
    segmented_beats.append([(round(len(y) / sr, 2), "")])  # 添加结束时间作为最后一个节拍点
    return segmented_beats

def generate_cut_points(video_path):
    """
    生成剪切点。

    参数：
        video_path (str): 视频文件路径。

    返回：
        list: 剪切点。
    """
    segmented_beats = extract_beats(video_path)
    cut_points = [beat for segment in segmented_beats for beat in segment]
    return sorted(cut_points)  # 按时间排序

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python planer.py <视频文件路径>")
        sys.exit(1)

    video_path = sys.argv[1]

    cut_points = generate_cut_points(video_path)
    print("剪切点:", cut_points)

    with open("temp.json", "w") as f:
        json.dump({'music': str(video_path), 'cut_points': cut_points}, f)