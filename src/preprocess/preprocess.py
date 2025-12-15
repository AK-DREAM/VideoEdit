import pickle
import math
from pathlib import Path

from ..features import VideoFeatures
from .shot_detect import detect_shots
from .calc import calc_saliency, calc_clip_embedding, calc_optical_flow
from ..utils.decoder import PaddedVideoDecoder

def preprocess_video(video_id: str, video_path: Path, output_path: Path, recalc: bool = False):
    if output_path.exists() and not recalc:
        print(f"\n=== Features already exist: {video_id}: {str(video_path)} ===")
        return

    if not video_path.exists():
        print(f"\n=== Video not found: {video_id}: {str(video_path)} ===")
        return

    print(f"\n=== Processing {video_id}: {str(video_path)} ===")
    decoder = PaddedVideoDecoder(video_path)
    num_frames = len(decoder)

    print(f"Decoder ready. Total frames = {num_frames}")

    features = VideoFeatures(video_id, num_frames)

    print(f"Calculating saliency...")
    calc_saliency(video_path, num_frames, features)
    print(f"Calculating CLIP embedding...")
    calc_clip_embedding(video_path, num_frames, features)
    print(f"Calculating optical flow...")
    calc_optical_flow(video_path, num_frames, features)
    
    print(f"Detecting shots...")
    detect_shots(video_path, features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {video_id}'s features to {str(output_path)}...")
    with open(output_path, "wb") as f:
        pickle.dump(features, f)