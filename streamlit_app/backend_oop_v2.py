import os
import json
import logging
import time
from openai import AzureOpenAI
import tempfile
from io import BytesIO
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from dotenv import load_dotenv

# Install or import these libraries if needed:
#   pip install markitdown PyMuPDF python-docx openai pydantic pillow
# Adjust references to your local environment as necessary.

# Pydantic for data models
from pydantic import BaseModel, Field

# The MarkItDown library for .docx -> .md
from markitdown import MarkItDown

# PyMuPDF for PDF -> images
import fitz  # PyMuPDF
# PIL for image handling
from PIL import Image

################################################################################
# 1. Data Models for Hypothesis Evaluations
################################################################################

class Hypothesis1Evaluation(BaseModel):
    reasoning: str
    score_1: Optional[bool] = Field(
        default=None,
        description="Minimal or no effort to reinforce the strategic direction. Leadership actions are absent or misaligned."
    )
    score_2: Optional[bool] = Field(
        default=None,
        description="Limited and inconsistent efforts to reinforce the strategic direction. Some actions are aligned but lack coherence."
    )
    score_3: Optional[bool] = Field(
        default=None,
        description="Moderate and consistent reinforcement of the strategic direction. Leadership actions are reliable and aligned."
    )
    score_4: Optional[bool] = Field(
        default=None,
        description="High degree of consistent reinforcement. Leadership shows clear alignment, fostering broader understanding."
    )
    score_5: Optional[bool] = Field(
        default=None,
        description="Exceptional and innovative reinforcement of the strategic direction. Leadership culture actively embeds the strategy."
    )
    certainty_of_correct_score_assignment: float


class AgileSafeMaturityEvaluation(BaseModel):
    reasoning: str
    score_1: Optional[bool] = Field(
        default=None,
        description="Minimal or no implementation of SAFe practices. Teams operate in silos with little alignment."
    )
    score_2: Optional[bool] = Field(
        default=None,
        description="Limited and inconsistent SAFe implementation. Some teams adopt elements, but it's not standardized."
    )
    score_3: Optional[bool] = Field(
        default=None,
        description="Moderate and consistent SAFe implementation. Most teams adhere to principles, processes are becoming standardized."
    )
    score_4: Optional[bool] = Field(
        default=None,
        description="High degree of SAFe practice implementation. Teams and leadership demonstrate strong alignment."
    )
    score_5: Optional[bool] = Field(
        default=None,
        description="Exceptional SAFe implementation. The organization fully embodies SAFe principles at all levels."
    )
    certainty_of_correct_score_assignment: float


# For structured JSON outputs with OpenAI
from openai import pydantic_function_tool
hypothesis1_tool = pydantic_function_tool(Hypothesis1Evaluation)
hypothesis2_tool = pydantic_function_tool(AgileSafeMaturityEvaluation)

################################################################################
# 2. LLM Client Interface
################################################################################

class ILLMClient(ABC):
    """
    Interface for interacting with an LLM (AzureOpenAI or OpenAI).
    Must be able to:
      1) create & run 'assistants' (the .beta.assistants approach)
      2) perform direct chat completions (chat.completion API).
    """

    @abstractmethod
    def get_existing_assistant_id(self, name: str) -> Optional[str]:
        pass

    @abstractmethod
    def create_assistant(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Any]] = None,
        tool_resources: Optional[Dict[str, Any]] = None
    ) -> str:
        pass

    @abstractmethod
    def run_assistant(
        self,
        assistant_name: str,
        messages: List[Dict[str, str]],
        attachments: Optional[List[Dict[str, Any]]] = None,
        instructions: str = "",
        tool_choice: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        poll_interval_ms: int = 3000
    ) -> str:
        pass

    @abstractmethod
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Directly invoke the chat.completion API with a set of messages, returning the response from the LLM.
        """
        pass

    @abstractmethod
    def process_image_to_text(self, image_path: str) -> str:
        """
        Process an image to text using the LLM's vision capabilities.
        """
        pass

################################################################################
# 3. Concrete AzureOpenAIClient (Mock Implementation)
################################################################################

class AzureOpenAIClient(ILLMClient):
    """
    Mocked AzureOpenAI client.
    In a real-world scenario, you'd integrate with the actual Azure or OpenAI Python library
    and supply real credentials (e.g., openai.api_type, openai.api_key, openai.api_base).
    """

    def __init__(self, api_key: str, endpoint: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-05-01-preview",
            timeout=180.0,
        )
        self.assistants_cache: Dict[str, str] = {a.name: a.id for a in self.client.beta.assistants.list().data}
        self.logger.info("AzureOpenAIClient initialized.")



    def get_existing_assistant_id(self, name: str) -> Optional[str]:
        return self.assistants_cache.get(name)

    def create_assistant(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Any]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        model: str = "cbs-test-deployment"
    ) -> str:
        """
        Creates a new assistant or retrieves an existing one if already created.

        Args:
            name (str): Name of the assistant.
            instructions (str): Instructions for the assistant.
            tools (Optional[List[Dict[str, Any]]]): List of tools for the assistant.
            tool_resources (Optional[Dict[str, Any]]): Resources for tools.
            model (str): The model to use for the assistant.

        Returns:
            str: The ID of the created or existing assistant.
        """
        existing_id = self.get_existing_assistant_id(name)
        if existing_id:
            self.logger.info(f"Assistant '{name}' already exists with ID={existing_id}. Reusing it.")
            return existing_id

        # Prepare the assistant creation payload
        assistant_payload = {
            "name": name,
            "instructions": instructions,
            "model": model,
        }

        # extend the payload with tools and tool_resources if provided
        if tools:
            assistant_payload["tools"] = tools
        if tool_resources:
            assistant_payload["tool_resources"] = tool_resources

        # Create the assistant by passing the payload dictionary
        assistant = self.client.beta.assistants.create(**assistant_payload)

        # Add the new assistant to the cache
        self.assistants_cache[name] = assistant.id
        self.logger.info(f"Created assistant '{name}' with ID={assistant.id} (Mock).")
        return assistant.id

    def run_assistant(
            self,
            assistant_name: str,
            messages: List[Dict[str, str]],
            attachments: Optional[List[Dict[str, Any]]] = None,
            instructions: str = "",
            tool_choice: Optional[str] = None,
            tools: Optional[List[Any]] = None,
            poll_interval_ms: int = 3000
    ) -> str:

        # Fetch Assistant ID by name
        assistant_id = self.get_existing_assistant_id(assistant_name)
        if assistant_id not in self.assistants_cache.values():
            return f"Assistant '{assistant_name}' not recognized. Please first create it."


        self.logger.info(
            f"run_assistant: assistant={assistant_name}, messages={messages}, attachments={attachments}, tools={tools}")

        # This function expects a dict with file_names ad file_paths.
        uploaded_files = []
        if attachments:
            for attachment in attachments:
                self.logger.info(f"Uploading file: {attachment['file_name']} to messages payload..")
                file = self.client.files.create(
                    file=open(attachment["file_path"], "rb"),
                    purpose="assistants"
                )
                uploaded_files.append({"file_id": file.id, "tools": [{"type": "file_search"}]})

        # Create a thread
        thread = self.client.beta.threads.create()

        # Add messages and attachments to the thread
        for message in messages:

            # Create message payload dictionary, add attachments if any
            message_payload = {
                "role": message["role"],
                "content": message["content"],
            }
            if len(uploaded_files) > 0:  # Attach uploaded files to the user's message
                message_payload["attachments"] = uploaded_files

            # Pass the payload dict to the create message method. We use the ** because we want it to be flexible.
            # With/without attachments, the method should work.
            self.logger.info(f"Adding messages to thread: {message_payload}")
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                **message_payload,
            )

        # Run the assistant on the thread
        self.logger.info(f"Running assistant '{assistant_name}' on the thread..")
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions=instructions,  # You can add any additional instructions here
        )

        if run.status == "completed":
            self.logger.info("Run completed successfully.")
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


    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_format: Optional[BaseModel] = None
    ) -> str:

        # Prepare the chat_completion payload
        chat_completion_payload = {model:'cbs-test-deployment',
                                   messages:messages}

        # Add response_format if provided
        if response_format:
            chat_completion_payload['response_format'] = response_format

        # Call the chat completion API
        response = self.client.chat.completions.create(**chat_completion_payload)

        # Handle structured response format
        if response_format:
            # Parse the response into the Pydantic model
            return response.choices[0].message.parsed

        # Return the plain message content if no response_format is provided
        return response.choices[0].message.content

    def process_image_to_text(self, image_path: str) -> str:
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string
        base64_image = encode_image(image_path)

        response = self.client.chat.completions.create(
            model="cbs-test-deployment",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                                    Describe in full detail what you see on this image. 
                                    Make sure you do not lose any information. 
                                    Respect the format of the input page your output. 
                                    Never add more information than you can find in the image

                                    Visual elements:
                                    - Diagrams: indicate diagrams with [Diagram] and ensure you describe the overal design shown in the diagram. Next describe all the diagram's elements you see how they interact with each other.
                                    - Charts: indicate charts with [Chart] and describe the information you see. I want you to make an objective description of the data displayed in the chart. Make sure not to lose any information.
                                    - Tables:
                                    Ensure to write out tables you see in the image as well. I want you to follow the format of the example table
                                    <example table>
                                    **Filtered Dataset:**

                                    | Prediction | Depot | Ritnaam             | Paketshift | Weekdag | Land | DeliveryWithinTimeFrame |
                                    |------------|-------|---------------------|------------|---------|------|--------------------------|
                                    | 3,689      | 21    | Brugge              | Jade Koala | 1       | Friday | NL | 0.8008                   |
                                    | 3,690      | 21    | Brugge              | Magenta Yak| 1       | Friday | NL | 0.0588                   |
                                    | 3,691      | 21    | Brugge              | Coral Monkey|1       | Friday | NL | 0.7373                   |
                                    | 3,692      | 21    | Brugge              | Orange Starfish| 1  | Friday | NL | 0.9672                   |
                                    | 3,693      | 21    | Brugge              | Gunmetal Horse| 1  | Friday | NL | 0.7565                   |
                                    | 3,694      | 21    | Brugge              | Cyan Lobster| 1     | Friday | NL | 0.9369                   |
                                    | 3,695      | 21    | Brugge              | Periwinkle Hippo| 1 | Friday | NL | 0.4924                   |
                                    | 3,696      | 21    | Brugge              | Buttercup Kangaroo| 1       | Friday | NL | 0.6632                   |
                                    | 3,697      | 21    | Brugge              | Denim Cat   | 1      | Friday | NL | 0.3133                   |
                                    | 3,698      | 21    | Brugge              | Ruby Moose  | 1      | Friday | NL | 0.4259                   |
                                    </example table>

                                        """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content



################################################################################
# 4. The Page-by-Page PDF to Markdown Approach
################################################################################

def pdf_pages_to_markdown(pdf_path: str, client: ILLMClient) -> str:
    """
    1) Convert each PDF page to an image (using PyMuPDF).
    2) For each image, call 'client.chat_completion' with the relevant instructions.
    3) Combine all resulting text into a single Markdown output.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Converting PDF to images and sending to LLM for page-level analysis: {pdf_path}")

    temp_folder = os.path.join(tempfile.gettempdir(), "pdf_images_for_openai")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        # scale to have a better resolution for the OCR/vision
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_filename = os.path.join(temp_folder, f"{os.path.basename(pdf_path)}_page_{page_number+1}.png")
        pix.save(image_filename)
        image_paths.append(image_filename)

    responses = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {}
        for i, img_path in enumerate(image_paths):
            future = executor.submit(client.process_image_to_text, img_path)
            future_map[future] = i

        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                resp_text = fut.result()
                # Label each page in the final output
                responses.append((idx, f"## Page {idx+1}\n\n{resp_text}"))
            except Exception as e:
                logger.error(f"Error in page {idx+1} extraction: {e}")

    # Sort by index
    responses.sort(key=lambda x: x[0])
    final_markdown = "\n\n".join([item[1] for item in responses])
    return final_markdown



################################################################################
# 5. DocumentProcessor (DOCX -> MarkItDown, PDF -> pdf_pages_to_markdown)
################################################################################

class DocumentProcessor:
    def __init__(self, storage_dir: str, llm_client: ILLMClient):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage_dir = storage_dir
        self.llm_client = llm_client

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def process_files(self, uploaded_files: List[BytesIO], filenames: List[str]) -> List[str]:
        """
        Takes a list of file-likes and associated filenames, converts each to Markdown,
        and stores the result. Returns a list of paths to the stored Markdown.
        """
        result_paths = []
        for file_obj, fname in zip(uploaded_files, filenames):
            file_ext = os.path.splitext(fname)[1].lower()
            tmp_path = self._save_temp_file(file_obj, fname)

            if file_ext in [".docx", ".doc"]:
                md_text = self._convert_docx_to_markdown(tmp_path)
            elif file_ext == ".pdf":
                md_text = self._convert_pdf_to_markdown(tmp_path)
            else:
                md_text = f"Unsupported file extension '{file_ext}'. No conversion performed."

            md_file_path = self._store_markdown(md_text, fname)
            result_paths.append(md_file_path)

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return result_paths

    def _save_temp_file(self, file_obj: BytesIO, orig_filename: str) -> str:
        temp_filename = f"{time.time()}_{os.path.basename(orig_filename)}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        with open(temp_path, "wb") as f:
            f.write(file_obj.read())
        file_obj.seek(0)
        return temp_path

    def _convert_docx_to_markdown(self, docx_path: str) -> str:
        """
        Leverages the MarkItDown library to convert docx -> markdown.
        """
        try:
            md = MarkItDown()
            result = md.convert(docx_path)
            return result.text_content
        except Exception as e:
            self.logger.error(f"Error converting docx with MarkItDown: {e}")
            return f"[Error in docx->md conversion: {e}]"

    def _convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Uses the PDF2Image2OpenAI approach:
        1) Convert PDF pages to images,
        2) For each image, call chat_completion,
        3) Combine into Markdown.
        """
        try:
            md_text = pdf_pages_to_markdown(pdf_path, self.llm_client)
            return md_text
        except Exception as e:
            self.logger.error(f"Error converting PDF via pdf_pages_to_markdown: {e}")
            return f"[Error in PDF->md conversion: {e}]"

    def _store_markdown(self, text: str, orig_filename: str) -> str:
        base_name = os.path.splitext(os.path.basename(orig_filename))[0].replace(" ", "_")
        md_filename = f"{base_name}_{int(time.time())}.md"
        md_path = os.path.join(self.storage_dir, md_filename)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)

        self.logger.info(f"Stored markdown at {md_path}.")
        return md_path

################################################################################
# 6. AssistantDefinition, Factory, Orchestrator
################################################################################

class AssistantDefinition(BaseModel):
    name: str
    instructions: str
    model: str
    tools: Optional[List[Any]] = None
    tool_resources: Optional[Dict[str, Any]] = None


class AssistantFactory:
    def __init__(self, llm_client: ILLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_assistant(self, definition: AssistantDefinition) -> str:
        return self.llm_client.create_assistant(
            name=definition.name,
            instructions=definition.instructions,
            tools=definition.tools,
            tool_resources=definition.tool_resources
        )


class AssistantOrchestrator:
    def __init__(self, llm_client: ILLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)

    def chat_text(self, assistant_name: str, messages: List[Dict[str, str]], attachments: Dict[str, str] = None) -> str:

        # Prepare the run payload
        run_payload = {
            "assistant_name": assistant_name,
            "messages": messages,
        }
        # Add attachments if any
        if attachments:
            run_payload["attachments"] = attachments

        # Run the assistant
        response = self.llm_client.run_assistant(**run_payload)
        return response

    def parse_to_json(
        self,
        assistant_name: str,
        user_input: str,
        tools: Optional[List[Any]]
    ) -> Any:

        messages = [{"role": "user", "content": user_input}]
        response = self.llm_client.run_assistant(
            assistant_name, messages, tool_choice="required", tools=tools
        )
        return response


################################################################################
# 7. BackEndManager
################################################################################

class BackEndManager:
    def __init__(self, api_key: str, endpoint: str, storage_dir: str):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create the LLM client
        self.llm_client = AzureOpenAIClient(api_key, endpoint)

        # Create the Assistant factory and orchestrator
        self.assistant_factory = AssistantFactory(self.llm_client)
        self.orchestrator = AssistantOrchestrator(self.llm_client)

        # Create the DocumentProcessor, now with a reference to the llm_client
        self.doc_processor = DocumentProcessor(storage_dir, self.llm_client)

    def init_assistants(self, assistant_defs: List[AssistantDefinition]):
        for definition in assistant_defs:
            if definition.name not in self.llm_client.assistants_cache.keys():
                self.assistant_factory.create_assistant(definition)

    def process_documents_for_hypotheses(
        self,
        uploaded_files: List[BytesIO],
        filenames: List[str],
        hypothesis_assistant_name: str,
        tools: Optional[List[Any]]
    ) -> dict:
        md_paths = self.doc_processor.process_files(uploaded_files, filenames)
        all_results = {}
        for md_path in md_paths:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            # Use parse_to_json with hypothesis and agile tools
            result = self.orchestrator.parse_to_json(
                assistant_name=hypothesis_assistant_name,
                user_input=md_content,
                tools=tools
            )
            all_results[md_path] = result
        return all_results

    def chat_with_assistant(self, assistant_name: str, user_text: str, attachments: Dict[str, str] = None) -> str:
        history = [{"role": "user", "content": user_text}]
        return self.orchestrator.chat_text(assistant_name=assistant_name, messages=history, attachments=attachments)

    def parse_text_with_assistant(self, assistant_name: str, text_to_parse: str, tools) -> Any:
        return self.orchestrator.parse_to_json(
            assistant_name=assistant_name,
            user_input=text_to_parse,
            tools=tools
        )


# ################################################################################
# # 8. Demo Main
# ################################################################################

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load the environment variables from a .env file
    load_dotenv()

    # Mock usage demonstration
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY_POSTNL")
    ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_POSTNL")
    STORAGE_DIR = os.path.join(tempfile.gettempdir(), "md_storage")

    # Initialize the backend manager
    backend = BackEndManager(api_key=API_KEY, endpoint=ENDPOINT, storage_dir=STORAGE_DIR)

    # Define the assistants
    interview_def = AssistantDefinition(
        name="OOP_interview_assistant",
        instructions="You are an interview assistant. Provide a friendly greeting.",
        model="cbs-test-deployment",
    )
    hypothesis1_def = AssistantDefinition(
        name="OOP_hypothesis1_assistant",
        instructions="You evaluate leadership and agile maturity from doc content.",
        model="cbs-test-deployment",
        tools=[hypothesis1_tool]
    )
    hypothesis2_def = AssistantDefinition(
        name="OOP_hypothesis2_assistant",
        instructions="You evaluate leadership and agile maturity from doc content.",
        model="cbs-test-deployment",
        tools=[hypothesis2_tool]
    )

    backend.init_assistants([interview_def, hypothesis1_def, hypothesis2_def])

    # Chat Example
    print("\n--- Chat Example ---")
    resp = backend.chat_with_assistant("OOP_interview_assistant", "Hello, can you help me?")
    # resp = backend.chat_with_assistant("OOP_interview_assistant", "Hello, can you help me?", attachments={"file_name": "sample.pdf", "file_path": "sample.pdf"})
    print("Interview Assistant:", resp)

    # Simulate PDF Bytes
    pdf_bytes = BytesIO(b"%PDF-1.4 This is a mock PDF content")
    # Simulate DOCX Bytes
    docx_bytes = BytesIO(b"FakeDOCXBinaryData...")

    print("\n--- Hypothesis on Documents ---")
    results = backend.process_documents_for_hypotheses(
        uploaded_files=[pdf_bytes, docx_bytes],
        filenames=["sample.pdf", "sample.docx"],
        hypothesis_assistant_name="hypothesis_assistant"
    )
    print(json.dumps(results, indent=2))

    print("\nAll done.")
