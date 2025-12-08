DINNER123

```
project_root/
│
├── data/
│   ├── raw/                       # 原始视频
│   ├── processed/                 # pad 后视频（可选）
│   ├── lists/
│   │   └── videos.csv
│   ├── benchmark/                 # 基准评测需要的标注/样本
│   │   ├── queries/               # prompt、场景描述等
│   │   ├── ground_truth.jsonl     # 评测集标注
│   │   └── rules.yaml             # 评测的 scoring 规则
│   └── agent/                     # agent 生成需要的文本材料等
│
├── output/
│   ├── features/                  # 每个视频的 VideoFeatures
│   ├── agent_results/             # 自动生成的 mashups/视频片段预案
│   ├── eval_results/              # 自动评测结果
│   ├── logs/                      # 日志
│   └── debug/
│
├── configs/
│   ├── preprocess.yaml            # 预处理配置
│   ├── eval.yaml                  # 评测配置（metrics weights 等）
│   ├── agent.yaml                 # agent 生成策略配置
│   └── paths.yaml
│
├── src/
│   ├── main_preprocess.py         # 全流程预处理入口
│   ├── main_evaluate.py           # 自动评测入口
│   ├── main_agent.py              # 自动 Agent 生成入口
│   │
│   ├── features/                  # VideoFeatures 的定义与加载
│   │   ├── video_features.py
│   │   ├── shots.py
│   │   ├── keyframes.py
│   │   └── __init__.py
│   │
│   ├── preprocess/
│   │   ├── saliency.py
│   │   ├── clip_embed.py
│   │   ├── optical_flow.py
│   │   ├── pad.py
│   │   └── __init__.py
│   │
│   ├── eval/                      # 评测模块
│   │   ├── metrics/
│   │   │   ├── saliency_score.py
│   │   │   ├── motion_score.py
│   │   │   ├── embedding_score.py
│   │   │   └── __init__.py
│   │   ├── evaluator.py           # 加载 features + metrics → 计算总分
│   │   ├── report.py              # 输出评测报告
│   │   └── __init__.py
│   │
│   ├── agent/                     # Agent 生成模块
│   │   ├── planner.py             # 规划 Agent 生成策略（分镜、镜头）
│   │   ├── retriever.py           # 用 features 做检索（CLIP/flow/saliency）
│   │   ├── scorer.py              # Agent 内部 scoring 函数
│   │   ├── llm_interface.py       # 和 LLM 交互的封装（OpenAI / vLLM）
│   │   ├── pipeline.py            # Agent 全流程控制
│   │   └── __init__.py
│   │
│   ├── utils/                     # 通用工具
│   │   ├── io.py
│   │   ├── video.py
│   │   ├── transforms.py
│   │   ├── path.py                # 统一管理路径
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── scripts/
│   ├── run_preprocess.sh
│   ├── run_evaluate.sh
│   ├── run_agent.sh
│   └── run_benchmark.sh
│
├── notebooks/
│   ├── debug_preprocess.ipynb
│   ├── debug_eval.ipynb
│   ├── debug_agent.ipynb
│   └── analysis.ipynb
│
├── environment.yml
├── requirements.txt
└── README.md
```