from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# Embeddings & Vector Store
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
vector_store = Chroma(
    persist_directory="data/chroma", embedding_function=embeddings_model
)
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# LLM & Prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=512)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only the provided context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)

# Base Chain Construction
base_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# In-Memory Session Store Dictionary
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a history instance for a unique user session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Wraps the Base Chain for Automatic History Management
rag_chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


def query_rag(question: str, session_id: str) -> str:
    """Executes the query idempotently per session_id."""
    return rag_chain_with_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )