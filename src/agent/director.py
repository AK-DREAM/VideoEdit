from .editor import load_video_features, generate_segment_video, EditResult
from .retriever import ScoreConfig
from .planner import Plan
from .llm_interface import chat_with_llm

import logging
import json
import open_clip

logger = logging.getLogger('director')

model_name = "ViT-B-32"
model, _, _ = open_clip.create_model_and_transforms(
    model_name,
    pretrained="weights/open_clip_model.safetensors",
    device="cuda"
)
model.eval()
logger.info(f"已加载 OpenCLIP 模型 {model_name}。")

def generate_video(user_prompt: str, plan: Plan, retry_count: int=3) -> EditResult:
    result = EditResult(
        video_candidates=[],
        total_frames=0,
        total_score=0.0
    )
    for segment in plan.segments:
        logger.info(f"处理段落，提示词: {segment.prompt}")
        prompt = (
            "你是一个视频剪辑师，需要更具用户的需求分析如何剪辑视频中的段落。\n"
            f"用户的需求是: {user_prompt}\n"
            f"当前段落的提示词是: {segment.prompt}\n"
            "你需要决定本段视频应该如何剪辑，具体应当包括：\n"
            "- prompt: CLIP 提示词，用于在检索视频片段，应当使用英文\n"
            "- prompt_weight: CLIP 提示词权重，应当是一个比较高的值，比如 0.5\n"
            "- semantic_weight: 语义相似度权重\n"
            "- saliency_weight: 显著性权重\n"
            "- motion_weight: 运动权重\n"
            "- energy_weight: 激烈程度权重\n"
            "- energy_value: 期望的激烈程度，范围是0~1000\n"
            "请结合提示词分析权重，比如高速的段落要求更高的运动权重。权重的和应当为 1。\n"
            "请以 JSON 格式返回上述内容，例如：\n"
            "```json\n"
            "{\n"
            "    \"prompt\": \"a fast-paced action scene\",\n"
            "    \"prompt_weight\": 0.5,\n"
            "    \"semantic_weight\": 0.2,\n"
            "    \"saliency_weight\": 0.2,\n"
            "    \"motion_weight\": 0.1,\n"
            "    \"energy_weight\": 0.1,\n"
            "    \"energy_value\": 500,\n"
            "}\n"
            "```\n"
        )
        logger.debug(f"提示词: \n{prompt}")
        retry_countdown = retry_count
        while retry_countdown > 0:
            try:
                llm_response = chat_with_llm([{
                    "role": "user",
                    "content": prompt
                }])
                logger.debug(f"LLM 回复: \n{llm_response}")
                start_index = llm_response.find("```json") + len("```json")
                end_index = llm_response.rfind("```")
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    llm_response = llm_response[start_index:end_index].strip()
                else:
                    raise ValueError("未找到 JSON 格式的列表内容")
                data = json.loads(llm_response)
                prompt_embed = model.encode_text(
                    open_clip.tokenize([data.get("prompt")]).to("cuda")
                ).detach().cpu().numpy()[0]
                config = ScoreConfig(
                    prompt_embed=prompt_embed,
                    prompt_weight=float(data.get("prompt_weight")),
                    semantic_weight=float(data.get("semantic_weight")),
                    saliency_weight=float(data.get("saliency_weight")),
                    motion_weight=float(data.get("motion_weight")),
                    energy_weight=float(data.get("energy_weight")),
                    energy_value=float(data.get("energy_value")),
                )
                new_segment = generate_segment_video(
                    prompt_embed,
                    segment.beat,
                    config,
                    result.total_frames,
                )[0]
                result.extend(new_segment)
                break
            except Exception as e:
                logger.error("处理段落时出错: %s", e)
                retry_countdown -= 1
                if retry_countdown <= 0:
                    raise
                logger.info(f"重试中，剩余次数: {retry_countdown}")
    return result

    