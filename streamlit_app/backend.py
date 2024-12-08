# backend.py
import logging
from openai import AzureOpenAI

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


class AssistantManager:
    def __init__(self, client: AzureOpenAI):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.vector_store_manager = VectorStoreManager(client)
        self.assistants = {}
        self.initialize_assistants()

    def initialize_assistants(self):
        assistant_names = ["interview_assistant", "summary_assistant", "hypothesis_1_3_assistant"]
        missing_assistants = self.validate_assistants_exist(assistant_names)
        self.create_assistants(missing_assistants)

        # Update assistant IDs
        assistants_list = self.client.beta.assistants.list().data
        for assistant in assistants_list:
            self.assistants[assistant.name] = assistant.id

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
            elif assistant_name == 'summary_assistant':
                self.create_summary_assistant()
            elif assistant_name == 'hypothesis_1_3_assistant':
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
        )
        self.logger.info(f"Assistant 'interview_assistant' created with ID: {assistant.id}")

    def create_summary_assistant(self):
        assistant = self.client.beta.assistants.create(
            name="summary_assistant",
            instructions="""
            Make a summary of the given text. Ensure that the summary is concise and captures the main points of the text.
            Do not lose any information; it is fine to be a bit longer but make sure you capture all the main points.
            """,
            tools=[],
            model="cbs-test-deployment",
        )
        self.logger.info(f"Assistant 'summary_assistant' created with ID: {assistant.id}")

    def create_hypothesis_assistant(self):
        file_paths = ["/path/to/your/document.docx"]
        vector_store_id = self.vector_store_manager.create_and_fill_vector_store(
            vector_store_name="hypothesis1_3_vs_id",
            file_paths=file_paths,
        )
        assistant = self.client.beta.assistants.create(
            name="hypothesis_1_3_assistant",
            instructions="""
            You are an expert in identifying strategy execution issues.
            You need to test the following hypothesis: 'Hypothesis 1.3: Leadership does not consistently reinforce the strategic direction, leading to confusion or a lack of urgency'.
            Read the transcript and assess how the leadership scores on this hypothesis.
            First, write out your reasoning, make a conclusion, and finally score them on a Likert scale from seven points.
            """,
            tools=[{"type": "file_search"}],
            model="cbs-test-deployment",
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        self.logger.info(f"Assistant 'hypothesis_1_3_assistant' created with ID: {assistant.id}")

    def chat_with_assistant(self, assistant_name: str, user_message: str, chat_history: list) -> str:
        assistant_id = self.assistants.get(assistant_name)
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

        # Add the new user message
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message,
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

    def process_interview(self, uploaded_file, chat_history):
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

        assistant_id = self.assistants.get('hypothesis_1_3_assistant')
        if not assistant_id:
            self.logger.error("Hypothesis assistant ID not found.")
            return "Error: Hypothesis assistant not found."

        # Run the assistant on the thread
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="",  # Additional instructions if any
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


class BackEndManager:
    def __init__(self, api_key: str, endpoint: str):
        self.logger = logging.getLogger(__name__)
        self.azure_client = AzureOpenAIClient(api_key, endpoint)
        self.assistant_manager = AssistantManager(self.azure_client.client)

    def init_back_end(self):
        # Initialization is handled in AssistantManager
        pass
