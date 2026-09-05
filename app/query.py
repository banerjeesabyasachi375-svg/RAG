from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI  # Updated import

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_openai import ChatOpenAI  # Updated import

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()


# ==========================================
# Embeddings
# ==========================================

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ==========================================
# Vector Store
# ==========================================

vector_store = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings_model
)


# ==========================================
# Retriever
# ==========================================

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


# ==========================================
# LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",  # Or "gpt-4o"
    temperature=0.2,
    max_tokens=512
)


# ==========================================
# Prompt
# ==========================================

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


# ==========================================
# Chat History
# ==========================================

chat_history = []


# ==========================================
# RAG Chain
# ==========================================

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