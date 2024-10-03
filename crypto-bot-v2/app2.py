import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from db import log_prompt_response

# Load environment variables.
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Set Qdrant connection URL and collection name
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "compliance_dB"

# Prompt template
PROMPT_TEMPLATE = """
Based on the following context, provide a comprehensive and well-organized response to the question. Ensure your answer includes all relevant headings and subheadings exactly as they appear in the context. Organize your response with clear headings and bullet points for each key section.

1. Include All Contextual Headings: Ensure every heading and subheading from the context is represented in your response.
2. Accuracy and Completeness: Address all relevant details (well explained) under each heading as outlined in the context.
3. Clarity and Structure: Use bullet points to present information clearly under each heading.
4. Insightfulness: Provide a thorough answer that accurately reflects the information from the context.
5. Completeness: Try to utilize all of the information present in the context and explain it well.
6. Details: Try to give a detailed background for each point so that user gets all the surrounding information for the particular query.
7. If you cannot find anything in the {context} regarding {question}, don't answer anything, just apologize.
Context:
{context}

---

Question: {question}

Give the most informed, thoughtful, and contextually relevant answer possible, or respond with an apology if the question is irrelevant, nonsensical, or no relevant information can be found in the database.
"""

# Initialize chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize OpenAI model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):  # User messages without avatar
            st.markdown(f"<div style='text-align: right'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):  # Assistant messages with avatar
            st.markdown(message["content"])

# Accept user input for queries
if prompt := st.chat_input("Enter your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=None):  # User input without avatar
        st.markdown(f"<div style='text-align: right'>{prompt}</div>", unsafe_allow_html=True)

    # Prepare the Qdrant client
    qdrant_client = QdrantClient(url=QDRANT_URL)

    # Prepare the embeddings function
    embedding_function = OpenAIEmbeddings()

    # Generate embeddings for the query text
    query_embedding = embedding_function.embed_query(prompt)

    # Perform a similarity search in Qdrant
    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=20,
        search_params=SearchParams(hnsw_ef=456, exact=False)
    )

    # If no results are found, respond with a default message
    if len(results) == 0:
        response = "Sorry, I couldn't find relevant information in the database for your query."
    else:
        # Construct the context by joining the contents of the top results
        context_text = "\n\n---\n\n".join([result.payload['page_content'] for result in results])

        # Prepare the prompt with the retrieved context and the user's query
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        full_prompt = prompt_template.format(context=context_text, question=prompt)

        # Generate a response from OpenAI based on the prompt
        model = ChatOpenAI()
        response = model.predict(full_prompt)

    # Log the prompt and response in the database
    log_prompt_response(prompt, response)

    # Display assistant response
    with st.chat_message("ai",avatar = None):
        st.markdown(response)

    # Add the assistant's response to the message history
    st.session_state.messages.append({"role": "assistant", "content": response})




