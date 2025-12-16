from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def build_retriever():
    loader = TextLoader("data/langchain_intro.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 2})


retriever = build_retriever()
