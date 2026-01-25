from ..utils.path import get_config
from openai import OpenAI

config = get_config("agent.yaml")

client = OpenAI(
    api_key=config.get("openai_api_key"),
    base_url=config.get("openai_base_url")
)

model_name: str = config.get("llm_model_name", "")

class Message:
    """表示与 LLM 交互的消息。

    属性：
        role (str): 消息角色（如 "user"、"assistant"、"system"）。
        content (str): 消息内容。
    """

    def __init__(self, role: str):
        self.role = role
        self.content = []
    def add_text(self, text: str):
        """添加文本内容到消息。

        参数：
            text (str): 文本内容。
        """
        self.content.append({
            "type": "text",
            "text": text
        })
        return self
    def add_image(self, image_path: str):
        """添加图像内容到消息。

        参数：
            image_path (str): 图像文件路径。
        """
        self.content.append({
            "type": "image_url",
            "image_url": {"url": image_path},}
        )
        return self
    def add_image_base64(self, image_base64: str):
        """添加 Base64 编码的图像内容到消息。

        参数：
            image_base64 (str): 图像的 Base64 编码字符串。
        """
        self.content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},}
        )
        return self
    def add_video(self, video_path: str, fps: int=2):
        """添加视频内容到消息。

        参数：
            video_path (str): 视频文件路径。
        """
        self.content.append({
            "type": "video_url",
            "video_url": {"url": video_path},
            "fps": fps
        })
        return self
    
    def to_dict(self) -> dict:
        """将消息转换为字典格式。

        返回：
            dict: 包含角色和内容的字典。
        """
        return {"role": self.role, "content": self.content}

def chat_with_llm(messages) -> str:
    """与 LLM 进行对话。

    参数：
        messages (list): 消息列表。
        verbose (bool): 是否实时打印详细信息。
    返回：
        str: LLM 的回复内容。
    """
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    result = chat_response.choices[0].message
    # if hasattr(result, "reasoning_content"):
    #     print(f"LLM 推理内容: {result.reasoning_content}")
    return result.content.strip() if result.content else ""
