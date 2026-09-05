from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load PDF
loader = PyPDFLoader("data/documents/Sabyasachi_Banerjee_Resume.pdf")
documents = loader.load()

print("Pages:", len(documents))
print(documents[0].page_content[:500])


# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 3. Create vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="vector_store",
    collection_name="pdf_test"
)


# 4. Add PDF pages
vector_store.add_documents(documents)


# 5. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# 6. Search
results = retriever.invoke("ACHIEVEMENTS of sabyasachi banerjee")

for doc in results:
    print("-----")
    print(doc.page_content[:1000])
    print(doc.metadata)