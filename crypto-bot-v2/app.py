import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchParams, ScoredPoint
import openai
import os
from dotenv import load_dotenv
# added test
# added test1
# Load environment variables.
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Set Qdrant connection URL and collection name
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "db_collection002"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """


PROMPT_TEMPLATE =   """


Based on the following context, provide the most accurate and insightful answer to the question. Where the context may be unclear or lacking, use your knowledge to infer the best possible answer.

Context:
{context}

---

Question: {question}

Give the most informed, thoughtful, and contextually relevant answer possible.



"""


# # Streamlit application
# def main():
#     st.title("Your Questions")

#     # Add a preset message as a sleek heading
#     st.markdown("<h3 style='text-align: center; color: grey;'>Hi, I am a Crypto Bot. What would you like me to answer?</h3>", unsafe_allow_html=True)


#     # Input field for query text
#     query_text = st.text_input("Enter your question:", value="")

#     # Button to trigger the search and response generation
#     if st.button("Submit"):
#         if not query_text:
#             st.error("Please enter a question.")
#             return

#         # Prepare the Qdrant client.
#         qdrant_client = QdrantClient(url=QDRANT_URL)

#         # Prepare the embeddings function.
#         embedding_function = OpenAIEmbeddings()

#         # Generate embeddings for the query text.
#         query_embedding = embedding_function.embed_query(query_text)

#         # Perform a similarity search in Qdrant.
#         results = qdrant_client.search(
#             collection_name=QDRANT_COLLECTION,
#             query_vector=query_embedding,  # Search using the generated query embedding.
#             limit=3,  # Return the top 3 results.
#             search_params=SearchParams(hnsw_ef=128, exact=False)  # Search parameters.
#         )

#         # Check if results were found.
#         if len(results) == 0:
#             st.error("Unable to find matching results.")
#             return

#         # Construct the context by joining the contents of the top results.
#         context_text = "\n\n---\n\n".join([result.payload['page_content'] for result in results])

#         # Prepare the prompt with the retrieved context and the user's query.
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query_text)

#         # Display the prompt (optional for debugging).
#         st.write(f"Prompt:\n{prompt}")

#         # Generate a response from OpenAI based on the prompt.
#         model = ChatOpenAI()
#         response_text = model.predict(prompt)

#         # Extract sources from the results metadata (if any).
#         sources = [result.payload.get("metadata", {}).get("source", None) for result in results]
#         formatted_response = f"Response: {response_text}\nSources: {sources}"

#         # Display the response and sources on the page.
#         st.write(formatted_response)

import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
import openai
import os
from dotenv import load_dotenv

# Load environment variables.
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']
#print(os.environ.get('OPENAI_API_KEY'))

# Set Qdrant connection URL and collection name
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "db_collection002"


PROMPT_TEMPLATE = """


Based on the following context, provide a comprehensive and well-organized response to the question. Ensure your answer includes all relevant headings and subheadings exactly as they appear in the context. Organize your response with clear headings and bullet points for each key section.

1. Include All Contextual Headings: Ensure every heading and subheading from the context is represented in your response.
2. Accuracy and Completeness: Address all relevant details (well explained) under each heading as outlined in the context.
3. Clarity and Structure: Use bullet points to present information clearly under each heading.
4. Insightfulness: Provide a thorough answer that accurately reflects the information from the context.
5. Completeness: Try to utilize all of the information present in the context and explain it well.
6. Details: Try to give a detailed background for each point so that user gets all the surrounding information for the particular query.

Context:
{context}

---

Question: {question}

Give the most informed, thoughtful, and contextually relevant answer possible.
"""

def main():
    # Sleek, classy message as a heading with improved styling
    st.markdown("""
    <style>
    .sleek-header {
        font-size: 28px;
        color: #FFFFFF; /* Pure white color */
        font-family: 'Arial', sans-serif;
        text-align: center;
        font-weight: 700; /* Bold font weight */
        margin-bottom: 30px;
    }
    .stTextInput label {
        font-size: 18px; /* Adjust label font size */
        color: #CCCCCC; /* Subtle color for input label */
    }
    .stTextInput input {
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #444444; /* Slightly rounded input */
        background-color: #222222; /* Dark background for input box */
        color: #FFFFFF; /* White text inside input */
    }
    .response-box {
        font-size: 16px;
        color: #E0E0E0; /* Light grey color */
        background-color: #2C2C2C; /* Darker background for the response */
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        line-height: 1.6;
    }
    .section-header {
        font-size: 22px;
        color: #FFD700; /* Gold color for section headers */
        font-weight: bold;
        margin-top: 20px;
    }
    .source-box {
        font-size: 14px;
        color: #AAAAAA; /* Light grey color for sources */
        margin-top: 10px;
    }
    </style>
    <h3 class='sleek-header'>Hi, I am a Crypto Bot. What would you like me to answer?</h3>
    """, unsafe_allow_html=True)

    # Input field for query text
    query_text = st.text_input("Enter your question:", value="")

    # Button to trigger the search and response generation
    if st.button("Submit"):
        if not query_text:
            st.error("Please enter a valid question.")
            return

        # Prepare the Qdrant client
        qdrant_client = QdrantClient(url=QDRANT_URL)

        # Prepare the embeddings function
        embedding_function = OpenAIEmbeddings()

        # Generate embeddings for the query text
        query_embedding = embedding_function.embed_query(query_text)

        # Perform a similarity search in Qdrant
        results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,  # Search using the generated query embedding
            limit=20, # Return the top 10 results
            search_params=SearchParams(hnsw_ef=456, exact=False)  # Search parameters #128
        )

        # Check if results were found
        if len(results) == 0:
            st.error("Unable to find matching results.")
            return

        # Display the raw results from Qdrant search
        # st.markdown("<h4 class='section-header'>Raw Search Results:</h4>", unsafe_allow_html=True)
        # for result in results:
        #     st.write(result.payload['page_content'])

        # Construct the context by joining the contents of the top results
        context_text = "\n\n---\n\n".join([result.payload['page_content'] for result in results])

        # Prepare the prompt with the retrieved context and the user's query
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Generate a response from OpenAI based on the prompt
        model = ChatOpenAI()
        response_text = model.predict(prompt)

        # Extract sources from the results metadata (if any)
        sources = [result.payload.get("metadata", {}).get("source", None) for result in results]
        formatted_sources = f"Sources: {', '.join([source for source in sources if source])}" if sources else "No sources available"

        # Display the formatted response and sources with enhanced styling
        st.markdown(f"<div class='response-box'><strong>Response:</strong> {response_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='source-box'>{formatted_sources}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()