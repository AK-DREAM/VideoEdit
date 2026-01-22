import os
from pathlib import Path
import allin1
from ..utils.path import get_output_dir

def analyze_music(music_path: Path) -> allin1.typings.AnalysisResult:
    cache_dir = get_output_dir() / "musics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{os.path.basename(music_path).split('.')[0]}.json"
    """
    分析音乐并将结果写入缓存文件，返回缓存文件路径。
    """
    if cache_file.exists():
        result = allin1.load_result(cache_file)
        if isinstance(result, list):
            # 分析返回了多个结果，使用第一个结果
            result = result[0]
    else:
        analyzed = allin1.analyze(music_path, out_dir=cache_dir)
        result = cast(allin1.typings.AnalysisResult, analyzed[0] if isinstance(analyzed, list) else analyzed)
    
    return result

def get_music_info(music_prof: allin1.typings.AnalysisResult) -> str:
    beats = list(zip(music_prof.beats, music_prof.beat_positions))
    # Note: Identifying the time signature/meter (e.g., 4/4) 
    beat_count = sorted(music_prof.beat_positions)[int(0.95 * len(music_prof.beat_positions))]
    sections = [
        (section, [(beat, position) for beat, position in beats if section.start <= beat < section.end])
        for section in music_prof.segments
    ]
    
    sections_info = []
    for section, beat_info in sections:
        sections_info.append(
            f"- {section.label} ({section.start:.2f}s - {section.end:.2f}s), "
            f"{len(beat_info)} beats total"
        )
    
    music_info = (
        f"This music has a total duration of {music_prof.segments[-1].end}s, a BPM of {music_prof.bpm:.0f}, "
        f"and is in {beat_count}/4 time. It contains {len(music_prof.segments)} sections:\n"
        + "\n".join(sections_info)
    )
    return music_info