from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, txt_loader, pdf_loader
from utils.path_tool import get_abs_path
import os
from utils.logger_handler import logger
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import asyncio
from concurrent.futures import ThreadPoolExecutor

class vector_store_service:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separator"],
            length_function=len,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    async def get_hybrid_retriever(self):
        """
        异步构建混合检索器
        """
        # 1. 向量检索器
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # 2. 真正异步获取文档 (放入线程池，避免阻塞)
        logger.info("开始从 Chroma 加载全量文档构建 BM25...")
        loop = asyncio.get_running_loop()
        # 将同步的 get() 放入线程池执行
        all_documents = await loop.run_in_executor(None, self._sync_get_all_documents)

        if all_documents:
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 10
            logger.info(f"BM25 索引构建完成，文档数: {len(all_documents)}")
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7]
            )
            return ensemble_retriever
        else:
            logger.warning("未找到文档，回退至纯向量模式")
            return vector_retriever

    def _sync_get_all_documents(self):
        """
        同步的阻塞操作，专门给线程池调用
        """
        try:
            stored_data = self.vector_store.get()
            return [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(stored_data['documents'], stored_data['metadatas'])
            ]
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


                
       