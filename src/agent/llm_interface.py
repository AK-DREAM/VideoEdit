from ..utils.path import get_config_path
from pathlib import Path
from openai import OpenAI

config_path = get_config_path("agent.yaml")

def load_agent_config():
    """加载 agent 配置文件。

    返回：
        dict: 配置文件内容。
    """
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

config = load_agent_config()

client = OpenAI(
    api_key=config.get("openai_api_key"),
    base_url=config.get("openai_base_url")
)
model_name = config.get("llm_model_name")

def chat_with_llm(messages, verbose=False):
    """与 LLM 进行对话。

    参数：
        messages (list): 消息列表。
        verbose (bool): 是否实时打印详细信息。
    返回：
        str: LLM 的回复内容。
    """
    if verbose:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True  # 开启流式传输
        )

        reply_content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                reply_content += delta.content
                print(delta.content, end='', flush=True)
        
        print()  # 换行

        return reply_content
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
