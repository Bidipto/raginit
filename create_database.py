# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import requests

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

url = os.environ.get("OPENAI_API_BASE_URL")

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    open = OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",  # LM Studio endpoint (updated parameter name)
        api_key="dummy-key",  # LM Studio doesn't need real API key (updated parameter name)
        model="text-embedding-qwen3-embedding-4b",  # Your model name
        # Alternative model names to try if the above doesn't work:
        # model="qwen3-embedding-4b"
        # model="embedding" 
    )

    embeddings = LMStudioEmbeddings()

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

class LMStudioEmbeddings:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "text-embedding-qwen3-embedding-4b"):
        self.base_url = base_url
        self.model = model
        import requests
        self.requests = requests
    
    def embed_documents(self, texts):
        """Embed multiple documents"""
        # Ensure texts are strings
        clean_texts = [str(text) for text in texts]
        
        response = self.requests.post(
            f"{self.base_url}/embeddings",
            json={
                "input": clean_texts,
                "model": self.model
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            raise Exception(f"Embedding API failed: {response.text}")
        
        result = response.json()
        return [item["embedding"] for item in result["data"]]
    
    def embed_query(self, text):
        """Embed a single query"""
        return self.embed_documents([str(text)])[0]


if __name__ == "__main__":
    main()