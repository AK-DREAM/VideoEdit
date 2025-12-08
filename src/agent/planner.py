import allin1
from .llm_interface import chat_with_llm
from ..utils.path import get_output_dir, get_log_path
import json
import os
import argparse
import logging

logger = logging.getLogger('planner')


def get_plan_with_llm(video_path, retry_count=3, offset=-0.05):
    """
    使用 allin1 提取节拍点，并通过 LLM 生成剪辑点。

    参数：
        video_path (str): 视频文件路径。
        api_key (str): OpenAI API 密钥。

    返回：
        list: 剪辑点。
    """
    # 设置缓存目录为上一层目录的 output/musics
    cache_dir = get_output_dir() / "musics"
    cache_file = cache_dir / f"{os.path.basename(video_path).split('.')[0]}.json"

    # 检查是否存在缓存文件
    if cache_file.exists():
        logger.info(f"使用缓存的分析结果 {cache_file}。")
        result = allin1.load_result(cache_file)
    else:
        logger.info("未找到缓存，重新分析视频。")
        result = allin1.analyze(video_path, out_dir=cache_dir)
        logger.info("分析结果已缓存到: %s", cache_file)
    
    # 初始化 OpenAI 客户端
    # 提取节拍点和段落数据
    beats = list(zip(result.beats, result.beat_positions))
    beat_count = sorted(result.beat_positions)[int(0.95 * len(result.beat_positions))]
    segments = [
        (segment, [(beat, position) for beat, position in beats if segment.start <= beat < segment.end])
        for segment in result.segments
    ]
    prompt = (
        "请让切换点对齐节拍点，如二拍、四拍等\n"
        "不要频繁变换剪辑点之间的距离\n"
        "请根据段落来判断需要保留的节拍点以及画面的速度\n"
        "激烈的段落可以切的更快，请在最高潮部分加入一小段快速切换、高速度的炫技段，但是请不要有大段（如连续超过5s）的快速剪辑（即每拍切换）\n"
        "画面切换的速度应当与段落的激烈程度匹配，不要重复使用一样的策略\n"
        "请不要有过多太长的片段无剪辑点\n"
    )
    full_prompt = (
        "你是一个专业的视频剪辑师，擅长根据音乐节拍点规划视频剪辑点。\n"
        "以下是视频的段落标签及其对应的节拍点及位置，请根据这些数据生成适合剪辑的视频剪辑点，"
        "并规划每一段画面的速度。\n"
        f"音乐的bpm是: {result.bpm:.2f}，每拍的时长是 {60 / result.bpm:.2f} 秒\n"
        f"这是一首 {beat_count} 拍子的音乐\n"
        "段落信息如下：\n"
        + "\n".join([
            f"{seg.label}({seg.start} - {seg.end}): 共 {len(beat_info)} 个节拍点"
            for seg, beat_info in segments
        ]) + "\n"
        f"{prompt}\n"
        "请先分析整体应当如何剪辑，只需要对每一段分析保留剪辑点和画面速度的策略，不需要分析更多的剪辑信息，不需要给出具体的切换点\n"
    )
    logger.debug("整体剪辑提示词:\n%s", full_prompt)
    full_plan = chat_with_llm([
        {"role": "user", "content": full_prompt},
    ])
    logger.debug("整体剪辑策略\n%s", full_plan)
    cut_plan = []
    for segment, beat_info in segments:
        if len(beat_info) == 0:
            continue
        segment_prompt1 = (
            f"{prompt}\n"
            f"请你回顾本段 {segment.label}({segment.start} - {segment.end})的策略，"
            "总结这一段应当如何筛选节拍点，不需要分析更多的剪辑信息，不需要给出具体的切换点"
        )
        segment_prompt2 = (
            "接下来给你一段的具体的节拍点信息，节拍点信息是一个 (节拍时间, 节拍位置) 的二元组。请结合刚才的策略，请对这一段选择合理的节拍点"
            f"{segment.label}({segment.start} - {segment.end}):{beat_info}\n"
            "直接输出 JSON 块，包含\"strength\"和\"cut_point\"，前者是一个1~10的整数，后者是剪辑点时间列表，例如: \n```json\n{\"strength\": 7, \"cut_point\": [0.21, 0.34, 1.21]}\n```\n"
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

                cut_points = json.loads(llm_response)
                cut_plan.append({
                    "label": segment.label,
                    "strength": cut_points.get("strength"),
                    "cut_point": [cp + offset for cp in cut_points.get("cut_point", [])]
                })
                break
            except Exception as e:
                logger.error("处理段落时出错: %s", e)
                retry_countdown -= 1
                if retry_countdown <= 0:
                    raise
                logger.info(f"重试中，剩余次数: {retry_countdown}")
    return cut_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner Script")
    parser.add_argument("video_path", type=str, help="视频文件路径")
    parser.add_argument("output_path", type=str, help="输出文件路径")
    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output_path

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
    cut_points = get_plan_with_llm(video_path)
    print("剪辑点:", cut_points)

    with open(output_path, "w") as f:
        json.dump({'music': video_path, 'plan': cut_points}, f)