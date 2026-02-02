import os
import json
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from typing import Sequence
from utils.path_tool import get_abs_path

history_path = get_abs_path("history_storage")
def get_history(session_id: str):
    os.makedirs(history_path, exist_ok=True)
    return FileChatMessageHistory(session_id, history_path)

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_path: str):
        self.session_id = session_id
        self.storage_path = history_path  # 修复拼写错误
        # 确保存储目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        # 文件路径应该指向一个JSON文件，而不是目录
        self.file_path = os.path.join(self.storage_path, f"{session_id}.json")

    def add_messages(self,messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)
        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                d = json.load(f)
                return messages_from_dict(d)
        except FileNotFoundError:
            return []

    def clear(self):
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)