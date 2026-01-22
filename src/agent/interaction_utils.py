from dataclasses import dataclass
from typing import List, Dict
import open_clip

from .editing_utils import ScoreConfig

@dataclass
class SectionInfo:
    """音乐段落信息"""
    label: str
    energy_level: str
    visual_tags: str
    rationale: str

@dataclass
class SegmentGuidance:
    """段落指导信息"""
    retrieval_query: str
    weight_profile: str
    visual_energy: int
    pacing_control: List[int]

    
# ==================== 预定义权重模板 ====================

class ConfigPresets:
    # 运动连贯性优先
    ACTION_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=0.2,
        semantic_weight=0.1,
        saliency_weight=0.1,
        motion_weight=0.5,  
        energy_weight=0.1,
        energy_value=0,
    )

    # 语义优先
    SEMANTIC_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=0.5,
        semantic_weight=0.2,
        saliency_weight=0.1,
        motion_weight=0.1,  
        energy_weight=0.1,
        energy_value=0,
    )

    # 显著性优先
    SALIENCY_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=0.2,
        semantic_weight=0.1,
        saliency_weight=0.5,
        motion_weight=0.1,  
        energy_weight=0.1,
        energy_value=0,
    )

    # 视觉综合优先
    VISUAL_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=0.2,
        semantic_weight=0.1,
        saliency_weight=0.3,
        motion_weight=0.3,  
        energy_weight=0.1,
        energy_value=0,
    )

    # 均衡
    BALANCED_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=0.2,
        semantic_weight=0.2,
        saliency_weight=0.2,
        motion_weight=0.2,  
        energy_weight=0.2,
        energy_value=0,
    )

def get_weight_profile_by_name(profile_name: str) -> ScoreConfig:
    """根据权重配置名称获取评分配置"""
    if profile_name == "Motion_Continuity_Priority":
        return ConfigPresets.ACTION_PRIORITY
    elif profile_name == "Semantic_Priority":
        return ConfigPresets.SEMANTIC_PRIORITY
    elif profile_name == "Composition_Similarity_Priority":
        return ConfigPresets.SALIENCY_PRIORITY
    elif profile_name == "Visual_Complexity_Priority":
        return ConfigPresets.VISUAL_PRIORITY
    elif profile_name == "Default_Priority":
        return ConfigPresets.BALANCED_PRIORITY
    else:
        return ConfigPresets.BALANCED_PRIORITY


model_name = "ViT-B-32"
model, _, _ = open_clip.create_model_and_transforms(
    model_name,
    pretrained="/home/keli/VideoEdit/weights/open_clip_model.safetensors",
    device="cuda"
)
model.eval()

def parse_segment_guidance(segment_guidance: Dict):
    """解析原始段落指导信息为标准格式。"""
    retrieval_query = segment_guidance["retrieval_query"]
    prompt_embed = model.encode_text(
        open_clip.tokenize([retrieval_query]).to("cuda")
    ).detach().cpu().numpy()[0]

    pacing_control = segment_guidance["pacing_control"]
    
    config = get_weight_profile_by_name(segment_guidance["weight_profile"])
    config.prompt_embed = prompt_embed
    config.energy_value = segment_guidance.get("visual_energy")

    return prompt_embed, pacing_control, config