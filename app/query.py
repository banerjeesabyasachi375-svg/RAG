from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()


# Embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vector store
vector_store = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings_model
)


# Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20
    }
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.2,
    max_output_tokens=512
)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using only the provided context."
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


# Chat history
chat_history = []


# RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": lambda _: chat_history
    }
    | prompt
    | llm
    | StrOutputParser()
)


def query_rag(question: str):

    answer = rag_chain.invoke(question)

    # Store conversation
    chat_history.append(
        HumanMessage(content=question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    return answer