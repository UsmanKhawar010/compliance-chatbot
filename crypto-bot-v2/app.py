
# import streamlit as st
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import SearchParams
# import openai
# import os
# import sqlite3
# from datetime import datetime
# from dotenv import load_dotenv
# from db import init_db, log_prompt_response

# # Load environment variables.
# load_dotenv()
# db_initialized = False

# # Set OpenAI API key
# openai.api_key = os.environ['OPENAI_API_KEY']
# #print(os.environ.get('OPENAI_API_KEY'))

# # Set Qdrant connection URL and collection name
# QDRANT_URL = "http://localhost:6333"
# QDRANT_COLLECTION = "compliance_dB"


# PROMPT_TEMPLATE = """


# Based on the following context, provide a comprehensive and well-organized response to the question. Ensure your answer includes all relevant headings and subheadings exactly as they appear in the context. Organize your response with clear headings and bullet points for each key section.

# 1. Include All Contextual Headings: Ensure every heading and subheading from the context is represented in your response.
# 2. Accuracy and Completeness: Address all relevant details (well explained) under each heading as outlined in the context.
# 3. Clarity and Structure: Use bullet points to present information clearly under each heading.
# 4. Insightfulness: Provide a thorough answer that accurately reflects the information from the context.
# 5. Completeness: Try to utilize all of the information present in the context and explain it well.
# 6. Details: Try to give a detailed background for each point so that user gets all the surrounding information for the particular query.

# ---

# If the question seems irrelevant, vague, or nonsensical (such as 'hello', 'hi', or other spammy content), respond with the following: "Sorry, I don't have relevant information for that query."

# If the question is legitimate but no relevant information is found in the context, respond with the following: "Sorry, I couldn't find relevant information in the database for your query."

# Context:
# {context}

# ---

# Question: {question}

# Give the most informed, thoughtful, and contextually relevant answer possible, or respond with an apology if the question is irrelevant, nonsensical, or no relevant information can be found in the database.
# """



# def main():

#     global db_initialized
#     if not db_initialized:
#         init_db()
#         db_initialized = True
    
    
#     # Sleek, classy message as a heading with improved styling
#     st.markdown("""
#     <style>
#     .sleek-header {
#         font-size: 28px;
#         color: #FFFFFF; /* Pure white color */
#         font-family: 'Arial', sans-serif;
#         text-align: center;
#         font-weight: 700; /* Bold font weight */
#         margin-bottom: 30px;
#     }
#     .stTextInput label {
#         font-size: 18px; /* Adjust label font size */
#         color: #CCCCCC; /* Subtle color for input label */
#     }
#     .stTextInput input {
#         border-radius: 10px;
#         padding: 10px;
#         border: 1px solid #444444; /* Slightly rounded input */
#         background-color: #222222; /* Dark background for input box */
#         color: #FFFFFF; /* White text inside input */
#     }
#     .response-box {
#         font-size: 16px;
#         color: #E0E0E0; /* Light grey color */
#         background-color: #2C2C2C; /* Darker background for the response */
#         padding: 15px;
#         border-radius: 10px;
#         margin-top: 20px;
#         line-height: 1.6;
#     }
#     .section-header {
#         font-size: 22px;
#         color: #FFD700; /* Gold color for section headers */
#         font-weight: bold;
#         margin-top: 20px;
#     }
#     .source-box {
#         font-size: 14px;
#         color: #AAAAAA; /* Light grey color for sources */
#         margin-top: 10px;
#     }
#     </style>
#     <h3 class='sleek-header'>Hi, I am a Crypto Bot. What would you like me to answer?</h3>
#     """, unsafe_allow_html=True)

#     # Input field for query text
#     query_text = st.text_input("Enter your question:", value="")

#     # Button to trigger the search and response generation
#     if st.button("Submit"):
#         if not query_text:
#             st.error("Please enter a valid question.")
#             return

#         # Prepare the Qdrant client
#         qdrant_client = QdrantClient(url=QDRANT_URL)

#         # Prepare the embeddings function
#         embedding_function = OpenAIEmbeddings()

#         # Generate embeddings for the query text
#         query_embedding = embedding_function.embed_query(query_text)

#         # Perform a similarity search in Qdrant
#         results = qdrant_client.search(
#             collection_name=QDRANT_COLLECTION,
#             query_vector=query_embedding,  # Search using the generated query embedding
#             limit=20, # Return the top 20 results
#             search_params=SearchParams(hnsw_ef=456, exact=False)  # Search parameters #128
#         )

#         # Check if results were found
#         if len(results) == 0:
#             st.error("Unable to find matching results.")
#             return

#         # Display the raw results from Qdrant search
#         # st.markdown("<h4 class='section-header'>Raw Search Results:</h4>", unsafe_allow_html=True)
#         # for result in results:
#         #     st.write(result.payload['page_content'])

#         # Construct the context by joining the contents of the top results
#         context_text = "\n\n---\n\n".join([result.payload['page_content'] for result in results])

#         # Prepare the prompt with the retrieved context and the user's query
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query_text)

#         # Generate a response from OpenAI based on the prompt
#         model = ChatOpenAI()
#         response_text = model.predict(prompt)
#         log_prompt_response(query_text, response_text)
        
        
#         st.markdown(f"<div class='response-box'><strong>Response:</strong> {response_text}</div>", unsafe_allow_html=True)


#         if "Sorry" not in response_text:
#             # Extract sources from the results metadata (if any)
#             sources = [result.payload.get("metadata", {}).get("source", None) for result in results]
#             formatted_sources = f"Sources: {', '.join([source for source in sources if source])}" if sources else "No sources available"

#             # Display the sources with enhanced styling
#             st.markdown(f"<div class='source-box'>{formatted_sources}</div>", unsafe_allow_html=True)

        

# if __name__ == "__main__":
#     main()






import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
import openai
import os
from dotenv import load_dotenv
from db import init_db, log_prompt_response

# Load environment variables.
load_dotenv()
db_initialized = False

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

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

---

If the question seems irrelevant, vague, or nonsensical (such as 'hello', 'hi', or other spammy content), respond with the following: "Sorry, I don't have relevant information for that query."

If the question is legitimate but no relevant information is found in the context, respond with the following: "Sorry, I couldn't find relevant information in the database for your query."

Context:
{context}

---

Question: {question}

Give the most informed, thoughtful, and contextually relevant answer possible, or respond with an apology if the question is irrelevant, nonsensical, or no relevant information can be found in the database.
"""

# Initialize session state to store query and response history
if 'history' not in st.session_state:
    st.session_state['history'] = []  # A list of (query, response) tuples

def main():
    global db_initialized
    if not db_initialized:
        init_db()
        db_initialized = True

    # CSS for fixed input box at the bottom and scrollable query history
    st.markdown("""
    <style>
    .sleek-header {
        font-size: 28px;
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
        text-align: center;
        font-weight: 700;
        margin-bottom: 30px;
    }
    .scroll-box {
        max-height: 500px;
        overflow-y: scroll;
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 10px;
        color: #FFFFFF;
    }
    .bottom-input {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 20px;
        background-color: #111111;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display query history
    st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)

    # Reverse order to make sure new queries appear at the top
    if st.session_state['history']:
        for query, response in reversed(st.session_state['history']):
            st.markdown(f"**Query:** {query}")
            st.markdown(f"**Response:** {response}")
            st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)

    # Move input field to the bottom of the page using fixed CSS
    with st.form(key='query_form', clear_on_submit=True):
        query_text = st.text_input("Enter your question:")
        submit_button = st.form_submit_button("Submit")

    # Handling submission of queries
    if submit_button and query_text:
        # Prepare the Qdrant client
        qdrant_client = QdrantClient(url=QDRANT_URL)

        # Prepare the embeddings function
        embedding_function = OpenAIEmbeddings()

        # Generate embeddings for the query text
        query_embedding = embedding_function.embed_query(query_text)

        # Perform a similarity search in Qdrant
        results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=20,
            search_params=SearchParams(hnsw_ef=456, exact=False)
        )

        # Check if results were found
        if len(results) == 0:
            response_text = "Sorry, I couldn't find relevant information in the database for your query."
        else:
            # Construct the context by joining the contents of the top results
            context_text = "\n\n---\n\n".join([result.payload['page_content'] for result in results])

            # Prepare the prompt with the retrieved context and the user's query
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            # Generate a response from OpenAI based on the prompt
            model = ChatOpenAI()
            response_text = model.predict(prompt)

        # Log the query and response
        log_prompt_response(query_text, response_text)

        # Store the query and response in session state history (new entries added at the top)
        st.session_state['history'].insert(0, (query_text, response_text))

        # Rerun the script after the first query submission to display the results immediately
        st.experimental_rerun()

if __name__ == "__main__":
    main()
