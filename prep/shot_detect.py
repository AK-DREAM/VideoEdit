import csv
import json
from pathlib import Path
from scenedetect import detect, AdaptiveDetector


def detect_shots(video_path):
    """
    Run shot detection on one video using PySceneDetect.
    Returns: a list of (start_sec, end_sec)
    """
    scene_list = detect(str(video_path), AdaptiveDetector(min_scene_len = 5, adaptive_threshold = 10))
    print(f"成功切分 {len(scene_list)} 段 clip")

    results = []
    for start, end in scene_list:
        results.append({
            "start": start.get_frames(),
            "end": end.get_frames()-1
        })
    return results


def process_csv(input_csv, output_jsonl):
    input_csv = Path(input_csv)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r") as f_in, \
         output_jsonl.open("w") as f_out:

        reader = csv.DictReader(f_in)

        for row in reader:
            video_id = row["video_id"]
            video_path = Path('../data/videos/'+row["filepath"])

            print(f"[ShotDetect] Processing {video_id} ...")

            if not video_path.exists():
                print(f"[WARN] Skip {video_id}, file not found: {video_path}")
                continue

            shots = detect_shots('../data/videos/'+row["filepath"])

            f_out.write(json.dumps({
                "video_id": video_id,
                "shots": shots
            }, ensure_ascii=False) + "\n")

    print(f"Finished! Output written to: {output_jsonl}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_jsonl", default="shot_boundaries.jsonl")
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_jsonl)
