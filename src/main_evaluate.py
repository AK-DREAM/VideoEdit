import argparse
import pickle
from pathlib import Path

from .preprocess import preprocess_video
from .features import VideoFeatures
from .eval.evaluate import evaluate_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data/videos",
                        help="Root folder for raw video files")
    parser.add_argument("--output_root", type=str, default="./output/videos",
                        help="Root folder for output features")
    parser.add_argument("--recalc", action='store_true', help="Recalculate all features")

    args = parser.parse_args()

    FilePath = Path(args.filepath)
    video_path = args.data_root / FilePath
    output_path = args.output_root / FilePath.parent / "features.pkl"

    if not output_path.exists() or args.recalc:
        print(f"Cached features not found / recalc enabled. Re-calculating...")
        preprocess_video("temp_video", video_path, output_path, args.recalc)

    with open(output_path, 'rb') as f:
        features: VideoFeatures = pickle.load(f)
        print(f"Features loaded successfully.")
    
    saliency_scores, semantic_scores, motion_scores = evaluate_scores(features)

    print("Average saliency score: {}".format(sum(saliency_scores) / len(saliency_scores)))

    print("Average semantic score: {}".format(sum(semantic_scores) / len(semantic_scores)))

    print("Average motion score: {}".format(sum(motion_scores) / len(motion_scores)))