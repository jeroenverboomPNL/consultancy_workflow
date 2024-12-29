# backend.py
import logging
from typing import Literal, TypedDict
from openai import AzureOpenAI, pydantic_function_tool
from pydantic import BaseModel, Field
from json_repair import repair_json
import json
import time
from typing import Optional
import logging



class HypothesisEvaluation(BaseModel):

    reasoning: str

    # Optional field for standardizing text (e.g., replacing specific words or patterns with standard forms)
    score_1: Optional[bool] = Field(
        default=None,
        description=(
            "Minimal or no effort to reinforce the strategic direction. "
            "Leadership actions are either absent or misaligned, causing confusion or a lack of clarity across the organization."
        )
    )
    # Optional field for formatting data (e.g., adjusting case or splitting/merging strings)
    score_2: Optional[bool] = Field(
        default=None,
        description=(
            "Limited and inconsistent efforts to reinforce the strategic direction. "
            "Some actions are aligned but lack follow-through or coherence, leading to fragmented understanding within the organization."
        )
    )
    score_3: Optional[bool] = Field(
        default=None,
        description=(
            "Moderate and consistent reinforcement of the strategic direction. "
            "Leadership actions are aligned and reliable, creating a baseline understanding and buy-in across most parts of the organization."
        )
    )
    score_4: Optional[bool] = Field(
        default=None,
        description=(
            "High degree of consistent reinforcement of the strategic direction. "
            "Leadership demonstrates clear alignment, fostering broad understanding and a sense of urgency to act on the strategy."
        )
    )
    score_5: Optional[bool] = Field(
        default=None,
        description=(
            "Exceptional and innovative reinforcement of the strategic direction. "
            "Leadership culture actively seizes every opportunity to embed the strategy, ensuring the entire workforce deeply understands it. "
            "Creative and impactful approaches are used to build alignment and momentum."
        )
    )
    # Required field for the certainty of task selection, represented as a float (e.g., 0.85 for 85%)
    certainty_of_correct_score_assignment: float

# Define tool object using openai pydanctic_function_tool
tool_obj = pydantic_function_tool(HypothesisEvaluation)



class AzureOpenAIClient:
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.client = None
        self.logger = logging.getLogger(__name__)
        self.initialize_client()

    def initialize_client(self):
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-05-01-preview",
            timeout=180.0,
        )
        self.logger.info("AzureOpenAI client initialized.")


class VectorStoreManager:
    def __init__(self, client: AzureOpenAI):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def create_and_fill_vector_store(self, vector_store_name: str, file_paths: list) -> str:
        vector_store = self.client.beta.vector_stores.create(name=vector_store_name)
        file_streams = [open(path, "rb") for path in file_paths]
        file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )
        self.logger.info(f"Vector store '{vector_store_name}' status: {file_batch.status}")
        return vector_store.id

class Router(BaseModel):
    task: Literal["standardise", "format", "extract", "map"]

tools = [pydantic_function_tool(Router)]

class AssistantManager:
    def __init__(self, client: AzureOpenAI):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.vector_store_manager = VectorStoreManager(client)
        self.assistants = {}
        self.initialize_assistants()



    def initialize_assistants(self):
        assistant_names = ["interview_assistant", "json_assistant", "hypothesis1_3_structured_output", "summary_assistant"]
        missing_assistants = self.validate_assistants_exist(assistant_names)
        self.create_assistants(missing_assistants)

        # Update assistant IDs
        assistants_list = self.client.beta.assistants.list().data
        for assistant in assistants_list:
            self.assistants[assistant.name] = assistant

    def validate_assistants_exist(self, assistant_names: list) -> list:
        missing_assistants = []
        assistants_list = [a.name for a in self.client.beta.assistants.list().data]
        for assistant_name in assistant_names:
            if assistant_name not in assistants_list:
                self.logger.warning(f"Assistant '{assistant_name}' does not exist.")
                missing_assistants.append(assistant_name)
            else:
                self.logger.info(f"Assistant '{assistant_name}' exists.")
        return missing_assistants

    def create_assistants(self, missing_assistants: list):
        if not missing_assistants:
            self.logger.info("All assistants are already created.")
            return

        # Create missing assistants
        for assistant_name in missing_assistants:
            if assistant_name == 'interview_assistant':
                self.create_interview_assistant()
            elif assistant_name == 'json_assistant':
                self.create_json_assistant()
            elif assistant_name == 'summary_assistant':
                self.create_summary_assistant()
            elif assistant_name == 'hypothesis1_3_structured_output':
                self.create_hypothesis_assistant()

    def create_interview_assistant(self):
        assistant = self.client.beta.assistants.create(
            name="interview_assistant",
            instructions="""
            You are a Domain-Specific Generative AI Assistant focused on knowledge absorption and interviewing the user to create documentation.
            Your capabilities include: Absorb and understand detailed information from users about their projects or expertise.
            Ask open-ended and clarifying questions to gather comprehensive knowledge.
            Ensure no information is missed by being curious and thorough in understanding the complete project.

            Use casual language with fillers (um, uh, 'like' and 'you know') to mimic natural speech.
            Sound sincerely excited, curious and sympathetic. Incorporate humor, smalltalk and open ended questions.
            Be extremely enthusiastic, showing excitement and eagerness to learn more about the topic.

            IMPORTANT:
            - Summarise the story of the user as concise as possible, like super super super short and compact. The fewer words you use, the better.
            - Focus ONLY on understanding the user's project by asking a maximum of 3 follow-up questions each time.
            - NEVER write long text outputs.
            """,
            tools=[],
            model="cbs-test-deployment",
            # model="gpt4o-consultancy-project",
        )
        self.logger.info(f"Assistant 'interview_assistant' created with ID: {assistant.id}")

    def create_json_assistant(self):
        assistant = self.client.beta.assistants.create(
            name="json_assistant",
            instructions="""
                    You will get a large text file as input ans will have to structure it for me.
                """,
            model="cbs-test-deployment",
            # model="gpt4o-consultancy-project",
            tools=[tool_obj])
        self.logger.info(f"Assistant 'interview_assistant' created with ID: {assistant.id}")


    def create_summary_assistant(self):
        # file_paths = ["./streamlit_app/test.txt"]
        # vector_store_id = self.vector_store_manager.create_and_fill_vector_store(
        #     vector_store_name="vs_hypothesis1_3_structured_output",
        #     file_paths=file_paths,
        # )
        assistant = self.client.beta.assistants.create(
            name="summary_assistant",
            instructions="""
            You are tasked with reading the files provided and making an extensive summary of the content.
            I want you to note down the key points and provide a structured summary of the content.
            Be careful, do not lose any information!
            """,
            tools=[{"type": "file_search"}],
            model="cbs-test-deployment",
            # model="gpt4o-consultancy-project",
            # tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        self.logger.info(f"Assistant 'hypothesis_1_3_assistant' created with ID: {assistant.id}")

    def create_hypothesis_assistant(self):
        file_paths = ["./streamlit_app/test.txt"]
        vector_store_id = self.vector_store_manager.create_and_fill_vector_store(
            vector_store_name="vs_hypothesis1_3_structured_output",
            file_paths=file_paths,
        )
        assistant = self.client.beta.assistants.create(
            name="hypothesis1_3_structured_output",
            instructions="""
            You are an expert in identifying strategy execution issues.
            You need to test the following hypothesis: 'Hypothesis 1.3: Leadership does not consistently reinforce the strategic direction, leading to confusion or a lack of urgency'.
            Read the transcript and assess how the leadership scores on this hypothesis.
            First, write out your reasoning, make a conclusion, and finally score them on a Likert scale from seven points.
            """,
            tools=[{"type": "file_search"}],
            model="cbs-test-deployment",
            # model="gpt4o-consultancy-project",
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        self.logger.info(f"Assistant 'hypothesis_1_3_assistant' created with ID: {assistant.id}")

    def chat_with_assistant(self, assistant_name: str, chat_history: list) -> str:

        # retrieve assistant ID
        assistant_id = self.assistants.get(assistant_name).id

        if not assistant_id:
            self.logger.error(f"Assistant '{assistant_name}' not found.")
            return "Error: Assistant not found."

        # Create a thread
        thread = self.client.beta.threads.create()

        # Add previous messages to the thread
        for message in chat_history:
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role=message["role"],
                content=message["content"],
            )

        # Run the assistant on the thread
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="",  # You can add any additional instructions here
        )

        if run.status == "completed":
            # Retrieve messages
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            # Find the assistant's reply
            for message in messages:
                if message.role == "assistant":
                    content = message.content[0]
                    if content.type == "text":
                        return content.text.value
            self.logger.error("Assistant did not return a response.")
            return "Error: Assistant did not return a response."
        else:
            self.logger.error(f"Run failed with status: {run.status}")
            return f"Error: Run failed with status: {run.status}"


    def process_interview(self, assistant_name: str, uploaded_file, chat_history):

        # Get the assistant ID
        assistant = self.assistants.get(assistant_name)
        assistant_id = assistant.id

        # Reset the file pointer to the beginning
        uploaded_file.seek(0)

        # Upload the file
        file = self.client.files.create(file=uploaded_file, purpose="assistants")

        # Create a thread
        thread = self.client.beta.threads.create()

        # Add previous messages to the thread
        for message in chat_history:
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role=message["role"],
                content=message["content"],
            )

        # Add the user message with the file attached
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Please process the attached interview transcript.",
            attachments=[{"file_id": file.id, "tools": [{"type": "file_search"}]}],
        )

        # Run the assistant on the thread
        run = self.client.beta.threads.runs.create_and_poll(
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
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            for message in messages:
                if message.role == "assistant":
                    content = message.content[0]
                    if content.type == "text":
                        return content.text.value


        elif run.status in ['cancelling', 'cancelled', 'failed', 'incomplete', 'expired']:
            self.logger.error(f"Run failed with status: {run.status}")
            raise RuntimeError(f"Error: Run failed with status: {run.last_error.message}")


    def parse_to_json(self, assistant_name: str, input: str):
        # Get the assistant ID
        assistant = self.assistants.get(assistant_name)
        assistant_id = assistant.id

        # Create a thread
        thread = self.client.beta.threads.create()

        # Add the user message to the thread
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=input
        )

        # Run the assistant on the thread
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="",  # Additional instructions if any
            tool_choice="required",
            tools=[tool_obj],
            poll_interval_ms=3000,  # Poll every 3 seconds
        )

        # if the run is completed, return the assistant's response
        if run.status == "completed":
            # Retrieve messages and find the assistant's reply
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            for message in messages:
                if message.role == "assistant":
                    content = message.content[0]
                    if content.type == "text":
                        return content.text.value

        # If the run requires action, it means we've used a tool meaning we asked for a structured output.
        # We can cancel the run and return the structured output
        if run.status == "requires_action":

            # Get the string of the tool output. It should be JSON but I've built in a check to handle non-JSON strings
            json_string = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments

            # If the JSON string is not valid, try to repair it. If it can't be repaired, return an error message
            try:
                json_dict = json.loads(json_string)
            except:
                try:
                    json_dict = repair_json(json_string)
                except Exception as e:
                    ValueError(f"ERROR: Could not repair JSON string. Error: {e}")
                    quit()

            # We canceL the run and return the repaired JSON string
            run = self.client.beta.threads.runs.cancel(
                thread_id=thread.id,
                run_id=run.id
            )
            return json_dict

        elif run.status in ['cancelling', 'cancelled', 'failed', 'incomplete', 'expired']:
            self.logger.error(f"Run failed with status: {run.status}")
            raise RuntimeError(f"Error: Run failed with status: {run.last_error.message}")



class BackEndManager:
    def __init__(self, api_key: str, endpoint: str):
        self.logger = logging.getLogger(__name__)
        self.azure_client = AzureOpenAIClient(api_key, endpoint)
        self.assistant_manager = AssistantManager(self.azure_client.client)

    def init_back_end(self):
        # Initialization is handled in AssistantManager
        pass






