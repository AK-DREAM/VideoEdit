from pathlib import Path
from scenedetect import detect, AdaptiveDetector, ContentDetector
from ..features import VideoFeatures

def detect_shots(video_path, features: VideoFeatures):
    scene_list = detect(str(video_path), AdaptiveDetector())
    print(f"Detected {len(scene_list)} shots in {str(video_path)}")

    results = []
    for start, end in scene_list:
        results.append({
            "start": start.get_frames(),
            "end": end.get_frames() - 1
        })
        
    features.shots = results