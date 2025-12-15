import allin1
from .llm_interface import chat_with_llm
from ..utils.path import get_output_dir, get_log_path
import json
import os
from pathlib import Path
import argparse
import logging
from typing import List, cast
from dataclasses import dataclass

logger = logging.getLogger('planner')

@dataclass
class SegmentPlan:
    beat: List[float]
    prompt: str

@dataclass
class Plan:
    music_path: str
    segments: List[SegmentPlan]


def get_plan_with_llm(music_path: Path, user_prompt: str, retry_count: int=3, offset: float=-0.05) -> Plan:
    """
    使用 allin1 提取节拍点，并通过 LLM 生成剪辑点。

    参数：

    返回：
        list: 剪辑点。
    """
    # 设置缓存目录为上一层目录的 output/musics
    cache_dir = get_output_dir() / "musics"
    cache_file = cache_dir / f"{os.path.basename(music_path).split('.')[0]}.json"

    # 检查是否存在缓存文件
    if cache_file.exists():
        logger.info(f"使用缓存的分析结果 {cache_file}。")
        result = allin1.load_result(cache_file)
        if isinstance(result, list):
            logger.warning("分析返回了多个结果，使用第一个结果。")
            result = result[0]
    else:
        logger.info("未找到缓存，重新分析视频。")
        analyzed = allin1.analyze(music_path, out_dir=cache_dir)
        result = cast(allin1.typings.AnalysisResult, analyzed[0] if isinstance(analyzed, list) else analyzed)
        logger.info("分析结果已缓存到: %s", cache_file)
    
    # 初始化 OpenAI 客户端
    # 提取节拍点和段落数据
    beats = list(zip(result.beats, result.beat_positions))
    beat_count = sorted(result.beat_positions)[int(0.95 * len(result.beat_positions))]
    segments = [
        (segment, [(beat, position) for beat, position in beats if segment.start <= beat < segment.end])
        for segment in result.segments
    ]
    full_prompt = (
        "你是一个视频剪辑师规划师，需要为我规划视频剪辑方案。\n"
        "以下是音乐的段落标签及其对应的时间，请为每一段规划合理的策略\n"
        f"这是一首 {beat_count} 拍子的音乐\n"
        "段落信息如下：\n"
        + "\n".join([
            f"{seg.label}({seg.start} - {seg.end}): 共 {len(beat_info)} 个节拍点，平均每拍时长 {(seg.end - seg.start) / len(beat_info) if len(beat_info) > 0 else float('nan'):.2f}"
            for seg, beat_info in segments
        ]) + "\n"
        f"{user_prompt}\n"
        "注意画面切换速度应当快慢结合，如intro段四拍一切（只在每小节第一拍切换），verse段二拍一切（在每小节第一、三拍切换），最高潮的部分选择一小段一拍一切。\n"
        "请先分析整体应当如何剪辑（包含画面切换的速度，画面强度，可以选择的内容，剪辑的策略），分析尽可能简短，不需要包含更多的信息。\n"
    )
    logger.debug("整体剪辑提示词:\n%s", full_prompt)
    full_plan = chat_with_llm([
        {"role": "user", "content": full_prompt},
    ])
    logger.debug("整体剪辑策略\n%s", full_plan)
    segments_plan = []
    for segment, beat_info in segments:
        if len(beat_info) == 0:
            continue
        segment_prompt1 = (
            f"请你回顾本段 {segment.label}({segment.start} - {segment.end})的策略，"
            "包含画面切换的速度，画面强度，可以选择的内容，剪辑的策略。\n"
            "分析尽可能简短，不需要包含更多的信息。"
        )
        segment_prompt2 = (
            "接下来给你一段的具体的节拍点信息，节拍点信息是一个 (节拍时间, 节拍位置) 的二元组，如 (1.1, 2) 表示这个节拍在 1.1 s，是本小节的第二拍。请结合刚才的策略，请对这一段选择合理的节拍点"
            f"{segment.label}({segment.start} - {segment.end}):{beat_info}\n"
            "直接输出 JSON 块，包含\"prompt\"和\"cut_point\"，前者是一个一段提示词，方便后面的剪辑，后者是剪辑点时间列表，例如: \n```json\n{\"prompt\": \"高潮段，快速切换，内容可以是...，策略是...\", \"cut_point\": [0.21, 0.34, 1.21]}\n```\n"
        )

        logger.info(f"处理段落: {segment.label}({segment.start} - {segment.end})")
        logger.debug("提示词1:\n%s", segment_prompt1)
        logger.debug("提示词2:\n%s", segment_prompt2)
        retry_countdown = retry_count
        while True:
            try:
                llm_response1 = chat_with_llm([
                    {"role": "user", "content": full_prompt},
                    {"role": "assistant", "content": full_plan},
                    {"role": "user", "content": segment_prompt1},
                ])
                logger.debug("段落策略\n%s", llm_response1)
                llm_response2 = chat_with_llm([
                    {"role": "user", "content": full_prompt},
                    {"role": "assistant", "content": full_plan},
                    {"role": "user", "content": segment_prompt1},
                    {"role": "assistant", "content": llm_response1},
                    {"role": "user", "content": segment_prompt2},
                ])
                llm_response = llm_response2
                logger.debug("剪辑点响应\n%s", llm_response)
                # 解析 LLM 返回的 JSON 格式内容
                start_index = llm_response.find("```json") + len("```json")
                end_index = llm_response.rfind("```")
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    llm_response = llm_response[start_index:end_index].strip()
                else:
                    raise ValueError("未找到 JSON 格式的列表内容")

                data = json.loads(llm_response)
                cut_points = data.get("cut_point", [])
                if abs(cut_points[0] - segment.start) < 0.01:
                    del cut_points[0]
                if len(cut_points) == 0 or abs(cut_points[-1] - segment.end) > 0.01:
                    cut_points.append(segment.end)

                segment_plan = SegmentPlan(
                    beat=[round(cp + offset, 2) for cp in cut_points],
                    prompt=data.get("prompt", "")
                )
                segments_plan.append(segment_plan)
                break
            except Exception as e:
                logger.error("处理段落时出错: %s", e)
                retry_countdown -= 1
                if retry_countdown <= 0:
                    raise
                logger.info(f"重试中，剩余次数: {retry_countdown}")
    cut_plan = Plan(music_path=str(music_path), segments=segments_plan)
    logger.info("规划结果: %s", cut_plan)
    return cut_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner Script")
    parser.add_argument("music_path", type=Path, help="音乐文件路径")
    parser.add_argument("output_path", type=Path, help="输出文件路径")
    parser.add_argument("user_prompt", type=str, help="剪辑提示词")
    args = parser.parse_args()

    music_path = args.music_path
    output_path = args.output_path
    user_prompt = args.user_prompt

    # 配置日志 - 只记录本程序的日志
    log_file = get_log_path()
    
    # 文件 handler: 记录 DEBUG 及以上级别
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    
    # 控制台 handler: 只输出 INFO 及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 只为本程序的 logger 添加 handler
    app_logger = logging.getLogger('planner')
    app_logger.setLevel(logging.DEBUG)
    app_logger.addHandler(file_handler)
    app_logger.addHandler(console_handler)
    
    # 禁用 propagate 避免日志传递到 root logger
    app_logger.propagate = False
    
    # 抑制第三方库日志
    logging.basicConfig(level=logging.WARNING)
    
    # 保持原有逻辑
    plan = get_plan_with_llm(music_path, user_prompt)

    # 将 Plan 对象转换为可序列化的字典
    plan_dict = {
        'music': plan.music_path,
        'segments': [
            {
                'beat': segment.beat,
                'prompt': segment.prompt
            }
            for segment in plan.segments
        ]
    }

    with open(output_path, "w") as f:
        json.dump(plan_dict, f, indent=2)