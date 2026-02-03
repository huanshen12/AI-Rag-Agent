from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.file_handler import get_file_md5_hex, txt_loader
from utils.path_tool import get_abs_path
import os
from utils.file_handler import txt_loader,pdf_loader,listdir_with_allowed_type
from utils.logger_handler import logger
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever


class vector_store_service:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name = chroma_conf["collection_name"],
            embedding_function = embed_model,
            persist_directory = chroma_conf["persist_directory"],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = chroma_conf["chunk_size"],
            chunk_overlap = chroma_conf["chunk_overlap"],
            separators = chroma_conf["separator"],
            length_function = len,
        )
    
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs = {"k": chroma_conf["k"]})
    
    # def get_hybrid_retriever(self):
    #     # 1. 获取向量检索器
    #     vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

    #     # 2. 核心修复：从 Chroma 缓存中提取所有已存在的文档
    #     # 在 1.x 中，Chroma 对象提供了一个简洁的 .get() 方法
    #     try:
    #         # 提取当前 collection 中的所有原始文档和元数据
    #         stored_data = self.vector_store.get() 
    #         all_documents = [
    #             Document(page_content=text, metadata=meta) 
    #             for text, meta in zip(stored_data['documents'], stored_data['metadatas'])
    #         ]
    #         logger.info(f"成功从持久化存储中同步了 {len(all_documents)} 个片段到 BM25")
    #     except Exception as e:
    #         logger.error(f"从 Chroma 提取文档失败: {e}")
    #         all_documents = []

    #     # 3. 字符匹配检索器 (BM25)
    #     if all_documents:
    #         # 这里的 BM25 会根据 query 的原始字符进行 TF-IDF 匹配
    #         bm25_retriever = BM25Retriever.from_documents(all_documents)
    #         bm25_retriever.k = 10
    #     else:
    #         logger.warning("BM25 索引构建失败，回退至纯向量模式")
    #         return vector_retriever

    # # 4. 组合两者 (Ensemble)
    # # 向量检索用 Embedding(query)；BM25 直接用 query 字符串
    #     ensemble_retriever = EnsembleRetriever(
    #         retrievers=[bm25_retriever, vector_retriever],
    #         weights=[0.3, 0.7]
    #     )
    #     return ensemble_retriever

    def get_hybrid_retriever(self):
    # 向量检索器（同步）
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        # 异步获取文档（I/O密集型）
        import asyncio
        all_documents = asyncio.run(self._async_get_documents())
        
        if all_documents:
            # 这里的 BM25 会根据 query 的原始字符进行 TF-IDF 匹配
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 10
        else:
            logger.warning("BM25 索引构建失败，回退至纯向量模式")
            return vector_retriever

    # 4. 组合两者 (Ensemble)
    # 向量检索用 Embedding(query)；BM25 直接用 query 字符串
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )
        return ensemble_retriever

    async def _async_get_documents(self):
        """异步获取文档，避免阻塞主线程"""
        try:
            # 这里可以使用异步方式获取文档
            # 例如，使用异步文件读取或异步数据库查询
            stored_data = self.vector_store.get() 
            all_documents = [
                Document(page_content=text, metadata=meta) 
                for text, meta in zip(stored_data['documents'], stored_data['metadatas'])
            ]
            logger.info(f"成功从持久化存储中同步了 {len(all_documents)} 个片段到 BM25")
            return all_documents
        except Exception as e:
            logger.error(f"从 Chroma 提取文档失败: {e}")
            return []
    def load_documents(self,documents:list[Document]):
        def check_md5_hex(md5_hex:str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
               open(get_abs_path(chroma_conf["md5_hex_store"]),"w").close()
               return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]),"r") as f:
                md5_hexs = f.readlines()
                for line in md5_hexs:
                    line = line.strip()
                    if line == md5_hex:
                        return True
                        
                return False
        def save_md5_hex(md5_hex:str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"a") as f:
                f.write(md5_hex+"\n")
        
        def get_file_documents(file_path:str):
            if file_path.endswith("txt"):
                return txt_loader(file_path)
            if file_path.endswith("pdf"):
                return pdf_loader(file_path)
            return None

        allowed_files_path = listdir_with_allowed_type(
            chroma_conf["data_path"],
            tuple(chroma_conf["allow_knowledge_file_type"]),
            )
        
        for file_path in allowed_files_path:
            md5_hex = get_file_md5_hex(file_path)
            if check_md5_hex(md5_hex):
                logger.info(f"文件 {file_path} 已加载")
                continue
            try:
                documents = get_file_documents(file_path)
                if not documents:
                    logger.error(f"文件 {file_path} 没有有效信息")

                    continue
                split_document = self.spliter.split_documents(documents)
                if not split_document:
                    logger.error(f"文件 {file_path} 分块后没有有效信息")
                    continue
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"文件 {file_path} 加载成功")
            except Exception as e:
                logger.error(f"文件 {file_path} 加载失败: {str(e)}",exc_info=True)
                continue


                
       