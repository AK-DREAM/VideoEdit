from src.agent.editor import load_video_features, generate_segment_video
from src.agent.retriever import ScoreConfig
import open_clip
from IPython.display import Video, display

import logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s'))

file_handler = logging.FileHandler("tmp.log", mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'))

planner_logger = logging.getLogger('planner')
planner_logger.setLevel(logging.DEBUG)
planner_logger.addHandler(console_handler)
planner_logger.addHandler(file_handler)
planner_logger.propagate = False

editor_logger = logging.getLogger('editor')
editor_logger.setLevel(logging.DEBUG)
editor_logger.addHandler(console_handler)
editor_logger.addHandler(file_handler)
editor_logger.propagate = False

director_logger = logging.getLogger('director')
director_logger.setLevel(logging.DEBUG)
director_logger.addHandler(console_handler)
director_logger.addHandler(file_handler)
director_logger.propagate = False

retriever_logger = logging.getLogger('retriever')
retriever_logger.setLevel(logging.DEBUG)
retriever_logger.addHandler(console_handler)
retriever_logger.addHandler(file_handler)
retriever_logger.propagate = False

from pathlib import Path
music_path = Path("/home/keli/VideoEdit/data/musics/01 - Slow Down.mp3")
csv_path = Path("/home/keli/VideoEdit/data/csv/MissionImpossible.csv")
data_root = Path("/home/keli/VideoEdit/data/videos/")
output_root = Path("/home/keli/VideoEdit/output/videos/")

model_name = "ViT-B-32"
model, _, _ = open_clip.create_model_and_transforms(
    model_name,
    pretrained="weights/open_clip_model.safetensors",
    device="cuda"
)
model.eval()

load_video_features(csv_path, data_root, output_root)

prompt = "A movie scene of landscape with extreme long shot."

prompt_embed = model.encode_text(
    open_clip.tokenize(prompt).to("cuda")
).detach().cpu().numpy()[0]

config = ScoreConfig(
    prompt_embed=prompt_embed,
    prompt_weight=0.5,
    saliency_weight=0.125,
    motion_weight=0.125,
    semantic_weight=0.125,
    energy_value=20,
    energy_weight=0.125
)

result1 = generate_segment_video(prompt_embed, [2,4,6,8,10], config, 0)

result1[0].generate_video("output/result/tmp.mp4")