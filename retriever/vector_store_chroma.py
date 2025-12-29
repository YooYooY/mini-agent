import chromadb
import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

emb = OpenAIEmbeddings(model="text-embedding-3-small")


class ChromaVectorIndex:
    def __init__(self, collection_name: str = "askmydocs_orders"):
        # 持久化到本地 ./chroma_db
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, docs: list[dict]) -> None:
        """
        docs 格式示例：
        [
          {
            "id": "orders-api-001",
            "title": "订单查询接口",
            "text": "整篇文档文本 ..."
          },
          ...
        ]
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for d in docs:
            chunks = splitter.split_text(d["text"])
            for i, chunk in enumerate(chunks):
                cid = f"{d['id']}::chunk_{i}"
                ids.append(cid)
                texts.append(chunk)
                metadatas.append(
                    {
                        "doc_id": d["id"],
                        "title": d["title"],
                        "chunk": chunk,
                    }
                )

        embeddings = emb.embed_documents(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query: str, k: int = 3) -> list[dict]:
        """返回真实 chunk 形式的 hits，供 retriever_node 使用"""
        q_emb = emb.embed_query(query)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
        )

        hits: list[dict] = []
        # 结果结构：ids / documents / metadatas / distances
        ids_list = results.get("ids", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            meta = metas_list[i]
            hits.append(
                {
                    "doc_id": meta["doc_id"],
                    "title": meta["title"],
                    "chunk": meta["chunk"],
                    "score": float(distances_list[i]),
                }
            )

        return hits


# 全局 index 实例
vector_index = ChromaVectorIndex()
