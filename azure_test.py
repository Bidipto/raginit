import numpy as np
from openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context, keep the answer clean: {question}
"""


class AzureOpenAIEmbeddingsWrapper:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    def embed_documents(self, texts):
        """Embed multiple documents"""
        clean_texts = [str(text).strip() for text in texts]

        response = self.client.embeddings.create(
            input=clean_texts, model=self.embedding_deployment
        )

        embeddings = [item.embedding for item in response.data]
        print(
            f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}"
        )
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
    print("=== Azure OpenAI Embedding Test ===\n")

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # Initialize embeddings wrapper
    embeddings = AzureOpenAIEmbeddingsWrapper()

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
        print(f"{i + 1}. {doc}")
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
            print(f"Doc {i + 1}: {similarity:.6f} | {documents[i][:50]}...")

        # Sort by similarity (highest first)
        similarities.sort(reverse=True)

        context_text = "\n\n---\n\n".join([doc for _, _, doc in similarities[:5]])

        print("\n=== Top Matches ===")
        for sim, idx, doc in similarities[:5]:
            print(f"Score: {sim:.6f} | Doc {idx + 1}: {doc}")

        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        print(f"\nPROMPT:\n{prompt}\n")

        # Get response using Azure OpenAI Chat model
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5000,
            temperature=0.7,
        )

        print("\n=== Azure OpenAI Model Response ===")
        print(f"RESPONSE: {response.choices[0].message.content}\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
