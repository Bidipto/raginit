import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question in-depth based on the above context: {question}
"""

class LMStudioEmbeddings:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "text-embedding-qwen3-embedding-4b"):
        self.base_url = base_url
        self.model = model
    
    def embed_documents(self, texts):
        """Embed multiple documents"""
        clean_texts = [str(text).strip() for text in texts]  # Clean and strip whitespace
        
        response = requests.post(
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
        embeddings = [item["embedding"] for item in result["data"]]
        
        # Debug: Print embedding info
        if embeddings:
            print(texts)
            print(f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")
            print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")
        
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query"""
        result = self.embed_documents([str(text).strip()])
        return result[0]
    
    # Add these methods to be more compatible with LangChain's expected interface
    def __call__(self, input):
        """Make the class callable"""
        if isinstance(input, str):
            return self.embed_query(input)
        elif isinstance(input, list):
            return self.embed_documents(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    @property 
    def dimension(self):
        """Property to get embedding dimension"""
        # Test with a small text to get dimension
        test_embedding = self.embed_query("test")
        return len(test_embedding)


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    # embedding_function = OpenAIEmbeddings()
    embedding_function = LMStudioEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=13)
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # model = ChatOpenAI(
    #     base_url="http://localhost:1234/v1",  # LM Studio endpoint
    #     api_key="dummy-key",
    #     model="qwen/qwen3-4b-thinking-2507"
    # )
    # response_text = model.predict(prompt)

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)

    response = requests.post(
                    "http://localhost:1234/v1/chat/completions",
                    json={
                        "model": "qwen/qwen3-4b-thinking-2507",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 40000,
                        "temperature": 0.7
                    },
                    headers={"Content-Type": "application/json"}
                )
                
    if response.status_code == 200:
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        print("\n=== Model Response (Direct API) ===")
        print(f"RESPONSE: {response_text}\n")
    else:
        print(f"Direct API call failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    main()