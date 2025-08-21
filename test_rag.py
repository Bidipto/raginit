import requests
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context, keep the answer clean: {question}
"""
# Simple LMStudioEmbeddings class
class LMStudioEmbeddings:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "text-embedding-nomic-embed-text-v1.5"):
        self.base_url = base_url
        self.model = model
    
    def embed_documents(self, texts):
        """Embed multiple documents"""
        clean_texts = [str(text).strip() for text in texts]
        
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
        
        print(f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query"""
        result = self.embed_documents([str(text).strip()])
        return result[0]

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    return 0.0

def main():
    print("=== Simple LM Studio Embedding Test ===\n")
    
    # Initialize embeddings
    embeddings = LMStudioEmbeddings()
    
    # Test documents
    documents = [
        "Chagla is gay",
        "Anubhav is a software engineer.",
        "Chagla and Anubhav are friends.",
        "Anubhav loves programming.",
        "Let's enjoy",
        "Chagla is a great person.",
        "Anubhav and Chagla often collaborate on projects.",
        "Chagla enjoys hiking and outdoor activities.",
        "Anubhav is passionate about AI and machine learning.",
        "Chagla and Anubhav met at a tech conference.",
        "Chagla is known for his humor.",
        "Anubhav has a pet cat named Whiskers.",
        "Chagla loves looking at Anubhav's nipples.",
        "Chagla once saw anubhav while he was taking a shower and he liked it.",
    ]
    
    # Test query
    query = input("Enter your query: ")
    
    print("Test Documents:")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc}")
    print(f"\nQuery: {query}\n")
    
    try:
        # Get embeddings for documents
        print("Getting document embeddings...")
        doc_embeddings = embeddings.embed_documents(documents)
        
        # Get embedding for query
        print("Getting query embedding...")
        query_embedding = embeddings.embed_query(query)
        
        print(f"Query embedding dimension: {len(query_embedding)}")
        print(f"Query embedding sample: {query_embedding[:5]}")
        
        # Calculate similarities
        print("\n=== Similarity Scores ===")
        similarities = []
        
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = calculate_similarity(query_embedding, doc_emb)
            similarities.append((similarity, i, documents[i]))
            print(f"Doc {i+1}: {similarity:.6f} | {documents[i][:50]}...")
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)

        context_text = "\n\n---\n\n".join([doc for _, _, doc in similarities[:5]])
        # print(f"\nContext for prompt:\n{context_text}\n")
        
        print("\n=== Top Matches ===")
        for sim, idx, doc in similarities[:5]:
            print(f"Score: {sim:.6f} | Doc {idx+1}: {doc}")

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        print(f"\nPROMPT:\n{prompt}\n")

        response = requests.post(
                    "http://localhost:1234/v1/chat/completions",
                    json={
                        "model": "qwen/qwen3-4b-thinking-2507",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 500,
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
        
        
        # # Test with exact matches
        # print("\n=== Testing Exact Matches ===")
        # exact_tests = [
        #     "Chagla", 
        #     "Anubhav",
        #     "programming"
        # ]
        
        # for test_word in exact_tests:
        #     test_embedding = embeddings.embed_query(test_word)
        #     print(f"\nTesting word: '{test_word}'")
            
        #     for i, doc_emb in enumerate(doc_embeddings):
        #         similarity = calculate_similarity(test_embedding, doc_emb)
        #         if similarity > 0.5:  # Only show good matches
        #             print(f"  Strong match with Doc {i+1}: {similarity:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()