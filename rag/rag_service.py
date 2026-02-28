from rag.vector_store import vector_store_service
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from utils.logger_handler import logger
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
import redis.asyncio as redis  # 👈 注意：这里用了异步 Redis
import hashlib
import json
import asyncio
import os
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
def prompt_print(prompt):
    # 调试打印
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class rag_service:
    def __init__(self):
        # 1. 初始化同步组件
        self.vector_store = vector_store_service()
        # 注意：load_documents 暂时保持同步（如果是应用启动时运行一次没问题）
        # self.vector_store.load_documents([]) 
        
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        
        # 2. 关键：不要在 init 里做异步操作，设为 None
        self.retriever = None 
        self.redis_client = None

    async def initialize(self):
        """
        新增一个显式的初始化方法，专门处理异步连接
        """
        if not self.retriever:
            logger.info("正在初始化混合检索器...")
            self.retriever = await self.get_final_retriever()
        
        if not self.redis_client:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("✅ Redis (Async) 连接成功")
            except Exception as e:
                logger.warning(f"❌ Redis 连接失败: {e}")
                self.redis_client = None

    def _init_chain(self):
        chain = self.prompt_template | prompt_print | self.model | StrOutputParser()
        return chain

    def _get_cache_key(self, query: str) -> str:
        md5_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"rag:cache:{md5_hash}"

    async def rag_sumarize(self, query: str):
        try:
            # 1. 确保初始化完成 (Lazy Init)
            if not self.retriever or not self.redis_client:
                await self.initialize()

            # 2. 查缓存 (Async)
            cache_key = self._get_cache_key(query)
            if self.redis_client:
                try:
                    cached = await self.redis_client.get(cache_key)
                    if cached:
                        logger.info(f"⚡️ 命中 Redis 缓存, 缓存: {cached}")
                        return cached
                except Exception as e:
                    logger.warning(f"Redis 读取异常: {e}")

            logger.info("缓存未命中，执行 RAG 检索...")
            
            # 3. 检索 (Async invoke)
            documents = await self.retriever.ainvoke(query)
            
            context = ""
            for i, doc in enumerate(documents):
                context += f"文档{i+1}：{doc.page_content}\n"
            
            # 4. 生成 (Async invoke)
            prompt_input = {"input": query, "context": context}
            response = await self.chain.ainvoke(prompt_input)
            
            # 5. 写入缓存 (Async)
            if self.redis_client:
                try:
                    await self.redis_client.setex(cache_key, 3600, response)
                except Exception as e:
                    logger.warning(f"Redis 写入异常: {e}")
                
            return response

        except Exception as e:
            logger.error(f"RAG 执行过程出错: {e}")
            return "抱歉，系统暂时无法检索信息。"
            
        finally:
            # 🧹【关键修改】扫尾工作：清理所有绑定在当前 Event Loop 上的资源
            if self.redis_client:
                await self.redis_client.aclose() # 关闭连接
                self.redis_client = None         # 重置为 None
            
            # Retriever 内部可能包含 Async Client (如 OpenAI HTTP Client)，也需要重置
            # 这里的 vector_store 是同步加载的，不需要重置，只需重置检索器包装器
            self.retriever = None 
            logger.info("🔄 资源清理完成，准备下一次调用")
    
    async def get_final_retriever(self):
        # 这里调用 vector_store 里的异步方法
        hybrid_retriever = await self.vector_store.get_hybrid_retriever()
        
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
        # 注意：langchain_classic 可能需要改为 langchain.retrievers
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid_retriever
        )
        return retriever