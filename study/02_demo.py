from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

load_dotenv()

loader = TextLoader("./data/docs.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
print("docs", docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("vectorstore stats =>", vectorstore._collection.count())

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


llm = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "You MUST answer strictly based on the given context. "
            "If the context does not contain the answer, say: "
            "'The provided document does not include this information.'",
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer strictly based on context.",
        ),
    ]
)


def rag_ask(query):
    relevant_docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in relevant_docs])

    messages = prompt.format_messages(context=context, question=query)
    print("messages=>", messages)

    res = llm.invoke(messages)

    return res.content


print(rag_ask("what does this document talk about?"))
