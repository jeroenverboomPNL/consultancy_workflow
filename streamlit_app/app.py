# app.py
import os

import dotenv
import streamlit as st
import logging
# Import the backend code
from backend import BackEndManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the backend
dotenv.load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY_POSTNL")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_POSTNL")

backend_manager = BackEndManager(api_key, endpoint)
backend_manager.init_back_end()

st.title("Interview Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Your message"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send the message to the interviewer assistant and get the response
    response = backend_manager.assistant_manager.chat_with_assistant(
        assistant_name='interview_assistant',
        user_message=prompt,
        chat_history=st.session_state.messages[:-1],  # Exclude the current user message
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Allow file upload
uploaded_file = st.file_uploader("Upload a file", type=["txt", "docx", "pdf", "pptx", "xlsx", "csv"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.write("File uploaded:", uploaded_file.name)

# Add 'Process Interview' button
if st.button('Process interview'):
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        # Get the uploaded file and chat history
        uploaded_file = st.session_state.uploaded_file
        chat_history = st.session_state.messages

        # Process the interview
        result = backend_manager.assistant_manager.process_interview(
            uploaded_file=uploaded_file,
            chat_history=chat_history,
        )

        # Display the assistant's response directly
        st.markdown(result)

    else:
        st.write("Please upload a file before processing.")
