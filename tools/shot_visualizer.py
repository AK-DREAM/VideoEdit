import cv2
from scenedetect import detect, AdaptiveDetector, ContentDetector
import random
import argparse
import os


def detect_shots(video_path, threshold=30.0):
    """使用 PySceneDetect 返回 shot 列表 (start_frame, end_frame)。"""
    scene_list = detect(str(video_path), AdaptiveDetector())

    # 转换成 frame 范围
    shots = []
    for start, end in scene_list:
        shots.append((start.get_frames(), end.get_frames()))

    return shots


def add_colored_border(frame, color, border=15):
    """给帧加彩色边框。"""
    return cv2.copyMakeBorder(
        frame,
        border, border, border, border,
        cv2.BORDER_CONSTANT,
        value=color
    )


def visualize_shots(video_path, shots, output_path, border_size=15):
    """将每个 shot 的第一帧加彩色边框并写出视频。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w + border_size * 2, h + border_size * 2)
    )

    shot_starts = {s[0] for s in shots}
    current_color = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 如果这帧是 shot 的第一帧，则换一个随机颜色
        if frame_idx in shot_starts:
            current_color = [random.randint(0, 255) for _ in range(3)]

        # 如果已经有颜色，则加边框
        if current_color is not None:
            frame = add_colored_border(frame, current_color, border=border_size)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[OK] Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", default="shot_visualized.mp4", help="Output video file")
    parser.add_argument("--threshold", type=float, default=30.0, help="PySceneDetect threshold")
    parser.add_argument("--border", type=int, default=15, help="Border size in pixels")
    args = parser.parse_args()

    args.input = os.path.join("data/videos/", args.input)

    print("[1] Detecting shots...")
    shots = detect_shots(args.input, args.threshold)
    print(f"Detected {len(shots)} shots")

    print("[2] Rendering visualization...")
    visualize_shots(args.input, shots, args.output, args.border)


if __name__ == "__main__":
    main()
