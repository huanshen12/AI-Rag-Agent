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

if __name__ == "__main__":
    vector_store_service = vector_store_service()
    vector_store_service.load_documents([])
    retriever = vector_store_service.get_retriever()
    results = retriever.invoke("迷路")
    for result in results:
        print(result.page_content)
        print("="*20)
    

                
       