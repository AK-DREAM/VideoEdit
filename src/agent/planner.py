import allin1
from .llm_interface import chat_with_llm
from ..utils.path import get_output_dir, get_log_path
import json
import os
import re
from pathlib import Path
import argparse
import logging
from typing import List, cast
from dataclasses import dataclass

logger = logging.getLogger('planner')

def _get_full_prompt(
    user_prompt: str, 
    footage_summary: str, 
    music_info: str
):
    full_prompt = f"""
# Role
You are a professional Movie Montage Planner. Your goal is to plan the visual narrative of a rhythm-sync movie montage based on the user's intent, the characteristics of the footage library, and the structure of the background music.

# Task
Analyze the provided [User Prompt], [Footage Summary], and [Music Information].

You must output two parts:
1. **Global Narrative Flow**: A high-level planning of the video's story arc and emotional progression.
2. **Detailed Section Plan**: A JSON list containing specific visual tags for each music section. These tags will be used for semantic retrieval of specific video materials later. You must output exactly one corresponding JSON item for every single music section listed in the music info.

# Guidelines
1. **Narrative Consistency**: The flow must align with the theme described in the [User Prompt].
2. **Energy Alignment**: Match the visual intensity to the music structure (e.g., Low energy for Intro, High energy for Chorus).
3. **Multi-Dimensional Tags**:
   - **Subject/Action**: (e.g., chase, dance, fight)
   - **Atmosphere/Vibe**: (e.g., dark, intense, joyful)
   - **Shot Type**: (e.g., close-up, wide shot)
   - **Energy Level**: (e.g., fast-paced, calm, dynamic)
4. **Tag Diversity & Contrast**: Avoid repeating the same tags. There **MUST** be a significant and visible difference between visual tags of different sections. 
5. **Vibe Tags**: Strictly avoid overly specific objects or scenes that might limit the retrieval space. Focus on general moods, styles, and cinematographic techniques.
6. **Tag Counts**: There is no need for too many tags. Usually 4-6 tags per section are sufficient.

# Input Data
**User Prompt:**
{user_prompt}

**Footage Summary:**
{footage_summary}

**Music Information:**
{music_info}

# Output Format
Please strictly follow this format:

## Global Narrative Flow
[Describe the overall story arc here. How does the video start? How does it build up? What is the climax? How does it end?]

## Detailed Section Plan
```json
[
  {{
    "section_name": "Intro",
    "energy_level": "Low/Medium/High",
    "visual_tags": ["tag1", "tag2", ...],
    "rationale": "Brief explanation..."
  }},
  {{
    "section_name": "Chorus",
    "energy_level": "...",
    "visual_tags": ["...", "..."],
    "rationale": "..."
  }},
  ...
]
```
    """
    return full_prompt

def get_plan_with_llm(
    user_prompt: str, 
    footage_summary: str,
    music_info: str,
    retry_count: int=3
):
    """
    使用 LLM 生成 json 格式的视频段落规划。
    """
    full_prompt = _get_full_prompt(
        user_prompt=user_prompt,
        footage_summary=footage_summary,
        music_info=music_info
    )
    logger.debug("完整提示词:\n%s", full_prompt)

    for attempt in range(retry_count):
        try:
            # 1. Call the LLM
            response = chat_with_llm([
                {"role": "user", "content": full_prompt},
            ])
            
            # 2. Extract the JSON block using Regular Expression
            # This looks for content inside ```json ... ``` or just the first [ or {
            json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: if no brackets found, the whole response might be raw JSON
                json_str = response.strip()

            # 3. Parse JSON
            plan_data = json.loads(json_str)
            
            # 4. Optional: Extract the Narrative Flow text separately if needed
            narrative_flow = ""
            if "## Global Narrative Flow" in response:
                narrative_flow = response.split("## Global Narrative Flow")[-1].split("##")[0].strip()

            return narrative_flow, plan_data, response

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Attempt {attempt + 1} failed to parse JSON: {e}")
            if attempt == retry_count - 1:
                print("Max retries reached. Returning None.")
                return None, None, None
            continue
