from typing import List, Dict, Union, cast
from ..utils.path import get_config
from openai import OpenAI
from openai.types.chat import ChatCompletion

config = get_config("agent.yaml")

client = OpenAI(
    api_key=config.get("openai_api_key"),
    base_url=config.get("openai_base_url")
)

model_name: str = config.get("llm_model_name", "")

def chat_with_llm(messages, verbose: bool = False) -> str:
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
            stream=True,
        )

        reply_content = ""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    reply_content += delta.content
                    print(delta.content, end='', flush=True)
        
        print()  # 换行

        return reply_content
    else:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
        )
        result = chat_response.choices[0].message.content
        return result if result else ""
