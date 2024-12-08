# Import necessary libraries
import os  # Provides functions to interact with the operating system
from dotenv import load_dotenv  # Used to load environment variables from a .env file
from openai import OpenAI, pydantic_function_tool  # OpenAI API client and tool integration
from pydantic import BaseModel, Field  # For creating data models with validation
import json  # Provides JSON parsing and handling
from typing import Literal, TypedDict, List, Optional  # Type hints for improved code readability and validation

# Route to the assistants that are identified by the Router model
def chat_with_assistant(assistant_id: str, chat_history: list) -> str:

    # Create a thread
    thread = client.beta.threads.create()

    # Add previous messages to the thread
    for message in chat_history:
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role=message["role"],
            content=message["content"],
        )

    # Run the assistant on the thread
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        instructions="",  # You can add any additional instructions here
    )

    if run.status == "completed":
        # Retrieve messages
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        # Find the assistant's reply
        for message in messages:
            if message.role == "assistant":
                content = message.content[0]
                if content.type == "text":
                    return content.text.value
        print("ERROR: Assistant did not return a response.")
        return "Error: Assistant did not return a response."
    else:
        print(f"ERROR: Run failed with status: {run.status}")
        return f"Error: Run failed with status: {run.status}"

def route_to_assistant(router_response: dict):
    # Check the Router fields to determine the appropriate task
    if router_response['standardise']:
        assistant_id = 'asst_Dglhdzs7dFMgL5TNeYrmsp4o'
        return chat_with_assistant(assistant_id=assistant_id, chat_history=chat_history)

    if router_response['format']:
        return "Format data"

    if router_response['extract']:
        return "Extract information"

    if router_response['map']:
        return "Map data"

    else:
        return "No task identified"

# PART 1 ------------------------------------------------------------------------------------
# Define a data model for routing tasks using Pydantic
class Router(BaseModel):
    # Optional field for standardizing text (e.g., replacing specific words or patterns with standard forms)
    standardise: Optional[bool] = Field(
        default=None,
        description="Smart find-and-replace that replaces words or patterns with their standardised form."
    )
    # Optional field for formatting data (e.g., adjusting case or splitting/merging strings)
    format: Optional[bool] = Field(
        default=None,
        description="Format data in terms of case, paddings, and merge/split"
    )
    # Optional field for extracting information (e.g., specific data points from text)
    extract: Optional[bool] = Field(
        default=None,
        description="Used for extracting specific pieces of information from input data."
    )
    # Optional field for mapping data to a new structure (e.g., converting data to match a schema)
    map: Optional[bool] = Field(
        default=None,
        description="Used for mapping input data to a new structure or schema."
    )
    # Required field for reasoning, which explains why the selected options are suitable for the task
    reasoning: str = Field(
        ...,
        description="Explain why the selected options are the most appropriate for the task."
    )
    # Required field for the certainty of task selection, represented as a float (e.g., 0.85 for 85%)
    certainty_of_correct_task: float






# START OF THE PROGRAM --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PART 1 START --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client using the API key from the environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a list of tools that use the Router model for validating and handling task-related requests
tools = [pydantic_function_tool(Router)]

# Create a list to hold messages for the chat interaction
chat_history = []

# Add a system message to guide the assistant's behavior and add a user message that contains a task request
chat_history.append({"role": "assistant", "content": "You are a helpful assistant. You have to choose which task you can be relevant to perform given the user query."})
chat_history.append({"role": "user", "content": "Can you help me replace the word e.g. with example? The value is 'For this one it is e.g. 2024-10-01'"})

# Use the OpenAI client to create a chat completion request
response = client.chat.completions.create(
    model='gpt-4o-2024-08-06',  # Specify the model to use
    messages=chat_history,  # Pass the messages for the interaction
    tools=tools  # Include the tools for task routing
)

# Extract the tool function response from the API output
router_response = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

# Print the results in a user-friendly format
print("\n".join([f"{key}: {value}" for key, value in router_response.items()]))


# PART 2 --------------------------------------------------------------------------------------------------------------------------------------------------------------------

assistant_response = route_to_assistant(router_response)
print(assistant_response)

chat_history.append({"role": "assistant", "content": assistant_response})

a=2