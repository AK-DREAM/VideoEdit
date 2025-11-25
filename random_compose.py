import json
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
import argparse

def load_shots(jsonl_file):
    shots_dict = {}
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            shots_dict[data["video_id"]] = data["shots"]
    return shots_dict

def sample_random_shots(shots_dict, video_base_path, n_shots):
    """
    从所有视频中随机选取 N 个 shot
    """
    all_shots = []
    for video_id, shots in shots_dict.items():
        video_path = Path("../data/videos/" + video_base_path + "{video_id}.mkv")
        for shot in shots:
            all_shots.append((video_path, shot["start"], shot["end"]))

    if n_shots > len(all_shots):
        n_shots = len(all_shots)

    return random.sample(all_shots, n_shots)

def compose_video(shots_sample, output_path):
    clips = []
    for video_path, start, end in shots_sample:
        clip = VideoFileClip(str(video_path)).subclip(start, end)
        clips.append(clip)
    
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile("../output/videos/" + str(output_path), codec="libx264", audio_codec="aac")
    final_clip.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots_jsonl", required=True, help="JSONL file containing shot boundaries")
    parser.add_argument("--video_base_path", default="2020", help="Folder containing videos")
    parser.add_argument("--n_shots", type=int, default=30, help="Number of shots to randomly select")
    parser.add_argument("--output", default="random_mix.mp4", help="Output video file")
    args = parser.parse_args()

    shots_dict = load_shots(args.shots_jsonl)
    sampled_shots = sample_random_shots(shots_dict, args.video_base_path, args.n_shots)
    compose_video(sampled_shots, args.output)

    print(f"Done! Output video saved to ../output/videos/{args.output}")
