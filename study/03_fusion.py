from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.callbacks.manager import CallbackManager

load_dotenv()

# Load
docs = TextLoader("data/docs.txt").load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = splitter.split_documents(docs)

# Vector DB
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=docs, embedding=embedding, persist_directory="./chroma_db"
)
vector_retriver = vectorstore.as_retriever(k=3)
print("vectorstore count: ", vectorstore._collection.count())

# BM25
bm25 = BM25Retriever.from_documents(splits)

# MultiQuery
llm = ChatOpenAI(temperature=0)
multis = MultiQueryRetriever.from_llm(retriever=vector_retriver, llm=llm)


# Fusion
fusion = EnsembleRetriever(
  retrievers=[vector_retriver, bm25],
  weights=[0.5, 0.5]
)

# # test
q = "Tell me about my neighbors"

# vector_docs = vector_retriver.invoke(q)
# print("vector_docs: ", vector_docs[0].page_content)

# bm25_docs = bm25.invoke(q)
# print("bm25_docs: ", bm25_docs[0].page_content)

fusion_docs = fusion.invoke(q)

multi_docs = multis.invoke(q)
# print("multi_docs: ", multi_docs[0].page_content)

all_docs = fusion_docs + multi_docs
unique_docs = {d.page_content: d for d in all_docs}.values()

for doc in unique_docs:
    print("----")
    print("result: ", doc.page_content)
