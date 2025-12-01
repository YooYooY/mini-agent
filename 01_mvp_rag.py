from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

DOC = """
LangChain is a framework for building LLM applications.

RAG stands for Retrieval Enhancement Generation: Search first, then answer.
"""

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_text(DOC)

emb = OpenAIEmbeddings(model="text-embedding-3-small")


vectordb = Chroma.from_texts(
    texts = docs,
    embedding=emb,
    persist_directory="./chroma_db"
)

retriever = vectordb.as_retriever()

llm = ChatOpenAI(temperature=0)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if __name__ == "__main__":
    query = "What is LangChain?"
    print("Query:", query)
    answer = qa.invoke({"query": query})
    print("Answer:", answer["result"])
    