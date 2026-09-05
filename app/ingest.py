import os

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)


def get_loader(file_path: str):
    extension = os.path.splitext(file_path)[1].lower()

    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
    }

    loader_cls = loaders.get(extension)

    if not loader_cls:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return loader_cls(file_path)


def ingest_document(file_path: str):

    loader = get_loader(file_path)

    documents = loader.load()

    chunks = text_splitter.split_documents(documents)

    vector_store = Chroma(
        persist_directory="data/chroma",
        embedding_function=embeddings_model
    )

    vector_store.add_documents(chunks)

    return {
        "documents": len(documents),
        "chunks": len(chunks)
    }