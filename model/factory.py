from abc import ABC,abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf
import os
import dotenv
dotenv.load_dotenv()

class base_model_factory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class chat_model_factory(base_model_factory):
    def generator(self) -> Optional[BaseChatModel]:
        return ChatOpenAI(model = rag_conf["chat_model"],api_key = os.getenv("deepseek_api_key"),base_url = os.getenv("deepseek_base_url"),temperature = 0)

class embedding_model_factory(base_model_factory):
    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model = rag_conf["embedding_model"],dashscope_api_key = os.getenv("dashscope_api_key"))

chat_model = chat_model_factory().generator()
embed_model = embedding_model_factory().generator()
