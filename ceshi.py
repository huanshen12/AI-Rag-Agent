from rag.vector_store import vector_store_service

vector = vector_store_service()


res=vector.get_hybrid_retriever().invoke("机器人开机无响应")
print(res)