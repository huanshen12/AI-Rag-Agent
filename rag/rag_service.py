from rag.vector_store import vector_store_service
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser

def prompt_print(prompt:str):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt

class rag_service():
    def __init__(self):
        self.vector_store = vector_store_service()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | prompt_print | self.model | StrOutputParser()
        return chain

    def retriever_documents(self,query:str):
        documents = self.retriever.invoke(query)
        return documents
    def rag_sumarize(self,query:str):
        documents = self.retriever_documents(query)
        context = ""
        counter = 0
        for doc in documents:
            counter += 1
            context += f"文档{counter}：{doc.page_content},元数据：{doc.metadata}\n"
        
        prompt = {
            "input":query,
            "context":context
        }
        response = self.chain.invoke(prompt)
        return response

if __name__ == "__main__":
    rag = rag_service()
    response = rag.rag_sumarize("小户型适合那种扫地机器人？")
    print(response)
