import argparse
from pathlib import Path
import csv

from .preprocess import preprocess_video
from .utils.path import load_video_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True,
                        help="CSV file with video_id, filepath")
    parser.add_argument("--data_root", type=Path, default=Path("./data/videos"),
                        help="Root folder for raw video files")
    parser.add_argument("--output_root", type=Path, default=Path("./output/videos"),
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