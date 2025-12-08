import argparse
from pathlib import Path
import csv

from .preprocess import preprocess_video

def load_video_list(csv_path):
    videos = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id, filepath = row["video_id"], row["filepath"]
            videos.append((video_id, filepath))
    return videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV file with video_id, filepath")
    parser.add_argument("--data_root", type=str, default="./data/videos",
                        help="Root folder for raw video files")
    parser.add_argument("--output_root", type=str, default="./output/videos",
                        help="Root folder for output features")
    parser.add_argument("--recalc", action='store_true', help="Recalculate all features")
    args = parser.parse_args()

    video_list = load_video_list(args.csv)
    print(f"Loaded {len(video_list)} videos from CSV.")

    for video_id, filepath in video_list:
        FilePath = Path(filepath)

        video_path = args.data_root / FilePath
        output_path = args.output_root / FilePath.parent / "features.pkl"

        preprocess_video(video_id, video_path, output_path, args.recalc)

    print("\nAll videos processed.")