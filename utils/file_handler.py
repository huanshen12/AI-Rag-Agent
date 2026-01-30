from logger_handler import logger
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

def get_file_md5_hex(file_path:str):   #获取md5值
    import hashlib
    if not os.path.exists(file_path):
        logger.error(f"文件{file_path}不存在")

    if not os.path.isfile(file_path):
        logger.error(f"{file_path}不是文件")

    chunk_size = 4096                     #分片防止文件过大，爆内存
    try:
        with open(file_path, "rb") as f:
            md5_hash = hashlib.md5()
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
    except Exception as e:
        logger.error(f"获取文件{file_path}的md5值时出错：{str(e)}")

    return md5_hash.hexdigest()

def listdir_with_allowed_type(file_path:str,allowed_types:tuple[str]):       #获取允许的文件格式
    file = []
    if not os.path.isdir(file_path):
        logger.error(f"{file_path}不是文件夹")
        return allowed_types
    for f in os.listdir(file_path):
        if f.endswith(allowed_types):
            file.append(os.path.join(file_path,f))
    return tuple(file)

        
def pdf_loader(file_path:str,passwd = None) -> list[Document]:
    return PyPDFLoader(file_path,passwd).load()

     

def txt_loader(file_path:str) -> list[Document]:
    return TextLoader(file_path).load()
