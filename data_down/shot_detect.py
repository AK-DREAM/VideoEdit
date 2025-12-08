import csv
import json
import subprocess
from pathlib import Path
from scenedetect import detect, AdaptiveDetector

def get_video_codec(video_path):
    """使用 ffprobe 获取视频编码"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    codec = result.stdout.strip()
    return codec

def transcode_to_h264(input_path, output_path):
    """将视频转码成 H.264"""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-c:a", "copy", str(output_path)
    ]
    subprocess.run(cmd, check=True)

def detect_shots(video_path):
    """
    Run shot detection on one video using PySceneDetect.
    Returns: a list of (start_sec, end_sec)
    """
    scene_list = detect(str(video_path), AdaptiveDetector(min_scene_len=5))
    print(f"成功切分 {len(scene_list)} 段 clip")

    results = []
    for start, end in scene_list:
        results.append({
            "start": start.get_frames(),
            "end": end.get_frames()
        })
    return results


def process_csv(input_csv, output_json):
    input_csv = Path(input_csv)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r") as f_in, \
         output_json.open("w") as f_out:

        reader = csv.DictReader(f_in)

        for row in reader:
            video_id = row["video_id"]
            video_path = Path('data/videos/'+row["filepath"])

            print(f"[ShotDetect] Processing {video_id} ...")

            if not video_path.exists():
                print(f"[WARN] Skip {video_id}, file not found: {video_path}")
                continue

            codec = get_video_codec(video_path)
            if codec.lower() == "av1":
                print(f"[INFO] {video_id} is AV1, renaming original and transcoding ...")
                av1_path = video_path.with_name(video_path.stem + "_av1.mp4")
                video_path.rename(av1_path)
                transcode_to_h264(av1_path, video_path)

            shots = detect_shots('data/videos/'+row["filepath"])

            f_out.write(json.dumps({
                "video_id": video_id,
                "shots": shots
            }, ensure_ascii=False) + "\n")

    print(f"Finished! Output written to: {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="prep/csv/mashup.csv")
    parser.add_argument("--output_json", default="prep/json/mashup.json")
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_json)
