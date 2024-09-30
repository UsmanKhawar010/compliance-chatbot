import argparse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchParams, ScoredPoint
import openai
import os
from dotenv import load_dotenv

# Load environment variables.
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Set Qdrant connection URL and collection name
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "db_collection002"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI to accept a query from the user.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the Qdrant client.
    qdrant_client = QdrantClient(url=QDRANT_URL)

    # Prepare the embeddings function.
    embedding_function = OpenAIEmbeddings()

    # Generate embeddings for the query text.
    query_embedding = embedding_function.embed_query(query_text)

    # Perform a similarity search in Qdrant.
    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,  # Search using the generated query embedding.
        limit=3,  # Return the top 3 results.
        search_params=SearchParams(hnsw_ef=128, exact=False)  # Search parameters.
    )

    # Check if results were found.
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    # Construct the context by joining the contents of the top results.
    context_text = "\n\n---\n\n".join([result.payload['page_content'] for result in results])

    # Prepare the prompt with the retrieved context and the user's query.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Generate a response from OpenAI based on the prompt.
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # Extract sources from the results metadata (if any).
    sources = [result.payload.get("metadata", {}).get("source", None) for result in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
