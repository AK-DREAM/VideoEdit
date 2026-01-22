from .llm_interface import chat_with_llm

import logging
import json, re

logger = logging.getLogger('director')

def _get_retrieval_query_prompt(
    user_prompt: str, 
    footage_summary: str, 
    music_info: str,
    section_info,
    beats_remaining,
    prev_queries
):
    retrieval_query_prompt = f"""
# Role
You are the **Director Agent** of a intelligent movie montage editing system (Phase 1: Visual Ideation). 
Your task is to brainstorm the visual content for the **current segment** (approx. 5-10s) that matches the vibe of the music section. 
There are multiple segments in a music section. You are only responsible for the current segment.

# Input Data
**Global Context:**
- User Prompt: {user_prompt}
- Footage Summary: {footage_summary}
- Music Info: {music_info.split('\n', 1)[0]}

# Current Section Info:
- Section Name: {section_info.get("section_name", "N/A")} 
- Energy Level: {section_info.get("energy_level", "N/A")}
- Vibe Keywords: {section_info.get("visual_tags", [])} 
- Beats Remaining: {beats_remaining} 

# Previous Segments (DO NOT REPEAT)
{'\n'.join([f"- {q}" for q in prev_queries])}

# Task
Generate a **Retrieval Query** that vaguely describes the suitable visual content for this segment.

# Guidelines

## 1. Matching the Vibe and Energy
- **Vibe Alignment**: The query should reflect the vibe provided for this music section.
- **Energy Alignment**:  Match the visual intensity to the `energy_level` of the music section.
- Use the Footage Summary and Vibe Keywords as *inspiration* for the visual content, but do NOT limit yourself to these specific scenes if they lead to repetition with previous segments. 

## 2. CLIP-Friendly Query Format
The task is retrieval-oriented, not generation-oriented. The query should ensure a wide range of retrievable footage.
- **Structure**: Use simple, declarative sentences: "Subject + Action" or "Subject + Description".
- **Style**: Be concise and visually descriptive. Avoid narrative or abstract words. Do not stuff the query with too many adjectives from `vibe_keywords`.
- **Simple Principle**: Focus on core visual elements. The scene should be common to ensure a wide range of retrievable footage. Limit the word count to 3-7 words.
- **BAD (Too Specific)**: "A man rushing through a rain-slicked alley at night, wearing a black tactical vest, with motion blur and neon lights reflecting off wet surfaces."
- **GOOD (USE THIS)**: "A man running fast."

## 3. Diversity and Anti-Repetition (IMPORTANT, Strict Check)
A montage requires visual variety. Consecutive segments do NOT need to be narratively continuous.
- Review the **Previous 4 Queries** above. 
- You **MUST NOT** reuse the same subject or action type.
- Try to improvise for fresh ideas while still aligning with the overall vibe.

# Output Format
Strictly JSON:
```json
{{
  "thought_process": "1. Analyze Vibe... 2. Check History (Previous was X, so I will do Y for diversity)... 3. Simplify for CLIP...",
  "retrieval_query": "Your final CLIP-friendly query string here"
}}
```
    """
    return retrieval_query_prompt

def _get_weight_profile_prompt(
    retrieval_query: str,
    section_info,
):
    weight_profile_prompt = f"""
You are the **Director Agent** of a intelligent movie montage editing system (Phase 2: Optimization Strategy). 
Your task is to select the technical **Constraint Weight Profile** that best optimizes the search for the specific visual query generated in the previous step.
You are also required to estimate the expected **Visual Energy Level** (0-100) of the video segment based on the query's content and vibe.

# Input Data
**Target Visual Query (from Step 1):**
"{retrieval_query}"

# Current Section Info:
- Section Name: {section_info.get("section_name", "N/A")} 
- Energy Level: {section_info.get("energy_level", "N/A")}
- Vibe Keywords: {section_info.get("visual_tags", [])} 

# Predefined Weight Profiles (Choose One)
1. **Semantic_Priority**: For specific scene/narrative focus.
2. **Motion_Continuity_Priority**: For action (chases, racing) requiring smooth flow.
3. **Composition_Similarity_Priority**: For still shots of characters/objects (e.g., match-cuts of close-up shots).
4. **Visual_Complexity_Priority**: For complex action (fights) balancing motion & framing.
5. **Default_Priority**: Balanced mode.

# Task
Analyze the **Target Visual Query**. Which profile will retrieve the highest quality video segment for this specific description?
Also estimate the expected **Visual Energy Level** (0-100) that best matches the intensity and dynamism of the video segment.

# Output Format
Strictly JSON:
```json
{{
  "thought_process": "The query describes a 'car chase', which relies heavily on motion smoothness...",
  "weight_profile": "Name_of_Profile"
  "visual_energy": An integer from 0 (lowest) to 100 (highest) indicating the expected visual energy level of the video segment. 
}}
```
    """
    return weight_profile_prompt

def _get_pacing_control_prompt(
    retrieval_query: str,
    footage_summary: str, 
    section_info,
    beats_remaining: int,
):
    pacing_control_prompt = f"""
# Role
You are the **Director Agent** (Phase 3: Rhythmic Execution).
Your task is to calculate the precise cutting rhythm (`pacing_control`) for the current video segment, based on the visual query generated in the previous step.
Specifically, you need to output a list of integers representing the duration (in beats) of each shot within this segment. (e.g. `[2, 2, 2, 2]` means 4 shots of 2 beats each)

# Input Data
**Target Visual Query (from Step 1):**
"{retrieval_query}"

**Global Context:**
- Footage Summary: {footage_summary}

# Current Section Info:
- Section Name: {section_info.get("section_name", "N/A")} 
- Energy Level: {section_info.get("energy_level", "N/A")}
- Vibe Keywords: {section_info.get("visual_tags", [])} 
- Beats Remaining: {beats_remaining} 

# Guidelines

## 1. Segment Length
- There are multiple segments in a music section. You are only responsible for the current segment.
- The total segment duration must fit within the remaining beats of the music section.
- That is, sum of all integers in the output list must be no more than `beats_remaining`.
- However, the segment duration should not be too long. Aim for a total duration of **4~16 beats**.

## 2. Musical Alignment (Tempo & Meter)
- **High Energy**: Use short durations per shot (1 or 2 beats). (e.g. `[1, 1, 1, 1, 2, 2]`)
- **Low Energy**: Use long durations per shot (4, 8, or 16 beats). (e.g. `[8]` or `[4, 4]`)
- **Triple Meter (3/4 time)**: If `Time Signature` is 3/4, avoid 2-beat or 4-beat cuts. Prioritize **3-beat** or **6-beat** cuts to match the waltz feel.

## 3. Adaptive Shot Density (Most Complex and Important)
Analyze the **Target Visual Query**. 
Is it a "Common Action" (easy to find many clips) or a "Specific Moment" (hard to find)?
Is it fast-paced or slow-paced?

| Content Type | Visual Richness | Strategy | Recommended Density |
| :--- | :--- | :--- | :--- |
| **Common Action**<br>(Running, Fighting, Car Chase, Dancing, Close-up Shot) | **High**<br>(Abundant footage) | **High Density**: Pack many short shots to create coherent video segment. | Use **Many Shots**.<br>e.g., 8 shots × 2 beats |
| **Specific Narrative / Emotion**<br>(Specific reaction, Looking at object, Crying, Explosion) | **Low**<br>(Scarce footage) | **Low Density**: Use fewer shots to avoid repetition or low-quality matches. Let the audience see the detail. | Use **1 or 2 Shots**.<br>e.g., 1 shot × 8 beats |
| **Atmospheric / Establishing**<br>(Landscapes, Wide City shots) | **Medium** | **Medium Density**: Let the visual breathe. | Use **1 or 2 Shots**.<br>e.g., 2 shots × 4 beats |

# Task
Based on the Target Visual Query and Current Section Info, decide the Pacing Strategy and output the list.

# Output Format
Strictly JSON:
```json
{{
  "thought_process": "1. Analyze Content: Query is "Indoor fighting scene", which is High Density Action and common. 2. Strategy: I will split it into 2-beat cuts... 3. Length check: Total 12 beats fits within remaining 30 beats...",
  "pacing_control": [2, 2, 2, 2, 2, 2]
}}
    """
    return pacing_control_prompt

def get_segment_guidance(
    user_prompt: str, 
    footage_summary: str, 
    music_info: str,
    section_info,
    beats_remaining: int,
    prev_queries,
    retry_count: int=3,
):
    retrieval_query_prompt = _get_retrieval_query_prompt(
        user_prompt=user_prompt,
        footage_summary=footage_summary,
        music_info=music_info,
        section_info=section_info,
        beats_remaining=beats_remaining,
        prev_queries=prev_queries,
    )
    for attempt in range(retry_count):
        try:
            response1 = chat_with_llm([
                {"role": "user", "content": retrieval_query_prompt}
            ])
            json_match = re.search(r'\{.*\}', response1, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response1.strip()

            retrieval_query_json = json.loads(json_str)
            break
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Attempt {attempt + 1} failed to parse JSON: {e}")
            if attempt == retry_count - 1:
                print("Max retries reached. Returning None.")
                return None
            continue

    retrieval_query = retrieval_query_json.get("retrieval_query", "")

    weight_profile_prompt = _get_weight_profile_prompt(
        retrieval_query=retrieval_query,
        section_info=section_info,
    )
    for attempt in range(retry_count):
        try:
            response2 = chat_with_llm([
                {"role": "user", "content": weight_profile_prompt}
            ])
            json_match = re.search(r'\{.*\}', response2, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response2.strip()

            weight_profile_json = json.loads(json_str)
            break
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Attempt {attempt + 1} failed to parse JSON: {e}")
            if attempt == retry_count - 1:
                print("Max retries reached. Returning None.")
                return None
            continue

    weight_profile = weight_profile_json.get("weight_profile", "")
    visual_energy = weight_profile_json.get("visual_energy", 0)

    pacing_control_prompt = _get_pacing_control_prompt(
        retrieval_query=retrieval_query,
        footage_summary=footage_summary,
        section_info=section_info,
        beats_remaining=beats_remaining,
    )
    for attempt in range(retry_count):
        try:
            response3 = chat_with_llm([
                {"role": "user", "content": pacing_control_prompt}
            ])
            json_match = re.search(r'\{.*\}', response3, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response3.strip()

            pacing_control_json = json.loads(json_str)
            break
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Attempt {attempt + 1} failed to parse JSON: {e}")
            if attempt == retry_count - 1:
                print("Max retries reached. Returning None.")
                return None
            continue

    pacing_control = pacing_control_json.get("pacing_control", [])

    guidance_profile = {
        "retrieval_query": retrieval_query,
        "weight_profile": weight_profile,\
        "visual_energy": visual_energy,
        "pacing_control": pacing_control,
    }

    return guidance_profile, (response1, response2, response3)
