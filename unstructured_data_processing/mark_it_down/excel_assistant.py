import os
import tempfile
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging

logger = logging.getLogger(__name__)
file_path = "/Users/jeroen.verboom/PycharmProjects/consultancy_workflow/unstructured_data_processing/mark_it_down/spring_volumes_test.md"
prompt = "I want you to tell me how many parcels were processed by MAD in the category Parcel Benelux and in the CGN Parcel UPU combined."

# Load environment variables
load_dotenv('/Users/jeroen.verboom/PycharmProjects/consultancy_workflow/streamlit_app/.env')

# Initialize Azure OpenAI client
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2024-05-01-preview",
#     timeout=180.0,
# )

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_POSTNL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_POSTNL"),
    api_version="2024-05-01-preview",
    timeout=180.0,
)

file = client.files.create(
  file=open(file_path, "rb"),
  purpose="assistants"
)

# Get the assistant ID
assistant_id = "asst_8op1eHo2skW3Hnd5yDjDwwR0"

# Create a thread
thread = client.beta.threads.create()


# Add the user message with the file attached
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=f"Please examine the excel file below very carefully. Make sure you do not lose any information. Now follow these instructions:\n{prompt}",
    attachments=[{"file_id": file.id, "tools": [{"type": "file_search"}]}],
)

# Run the assistant on the thread
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant_id,
    instructions="",  # Additional instructions if any
    tools=[{"type": "file_search"}],
    # tool_choice="required",
    poll_interval_ms=3000,  # Poll every 3 seconds
)

# if the run is completed, return the assistant's response
if run.status == "completed":
    # Retrieve messages and find the assistant's reply
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for message in messages:
        if message.role == "assistant":
            content = message.content[0]
            if content.type == "text":
                response = content.text.value
                print(content.text.value)


elif run.status in ['cancelling', 'cancelled', 'failed', 'incomplete', 'expired']:
    logger.error(f"Run failed with status: {run.status}")
    raise RuntimeError(f"Error: Run failed with status: {run.last_error.message}")