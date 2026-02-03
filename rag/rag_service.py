from rag.vector_store import vector_store_service
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from utils.logger_handler import logger
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.flashrank_rerank import FlashrankRerank

def prompt_print(prompt:str):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt

class rag_service():
    def __init__(self):
        self.vector_store = vector_store_service()
        logger.info("正在加载文档...")
        self.vector_store.load_documents([])
        self.retriever = self.get_final_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | prompt_print | self.model | StrOutputParser()
        return chain

    # async def retriever_documents(self,query:str):
    #     documents = await self.retriever.ainvoke(query)
    #     return documents
    async def rag_sumarize(self,query:str):
        documents = await self.retriever.ainvoke(query)   
        context = ""
        counter = 0
        for doc in documents:
            logger.info(f"参考资料:{doc.page_content}")
            print(doc.page_content)
            counter += 1
            context += f"文档{counter}：{doc.page_content},元数据：{doc.metadata}\n"
        
        prompt = {
            "input":query,
            "context":context
        }
        response = await self.chain.ainvoke(prompt)
        return response
    
    async def get_final_retriever(self):
        hybrid_retriever = await self.vector_store.get_hybrid_retriever()
        
        compressor = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2",
            top_n=3
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid_retriever
        )
        return retriever

