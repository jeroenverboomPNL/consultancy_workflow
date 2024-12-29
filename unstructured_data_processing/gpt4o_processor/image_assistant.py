import os
import tempfile
from dotenv import load_dotenv
from openai import AzureOpenAI
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv('/Users/jeroen.verboom/PycharmProjects/consultancy_workflow/streamlit_app/.env')

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    timeout=180.0,
)


def convert_pdf_to_images(pdf_path):
    """Convert each page of a PDF to an image using PyMuPDF."""
    pdf = fitz.open(pdf_path)
    for page_number in range(len(pdf)):
        # Get the page
        page = pdf[page_number]
        # Render the page to an image (default resolution: 72 dpi)
        # pix = page.get_pixmap(matrix=fitz.Matrix(5, 5)) # 4.17 for 300dpi
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 4.17 for 300dpi
        # Save the image
        pix.save(f"./images/page_{page_number + 1}.png")

    # get all absolute paths for the images
    images = [os.path.abspath(f"./images/page_{page_number + 1}.png") for page_number in range(len(pdf))]
    return images


def convert_ppt_to_images(ppt_file, output_folder):
    from pptx import Presentation
    from pptx.util import Inches
    from PIL import Image


    # Open the PowerPoint presentation
    presentation = Presentation('20240402_PostNL_GenAI CoE_Workshop 1 _vOutcomes.pptx')

    # Loop through slides and save each as an image
    for slide_index, slide in enumerate(presentation.slides):
        image_path = f"{output_folder}/slide_{slide_index + 1}.jpg"

        # Create a white image or with any color
        slide_image = Image.new("RGB", (1920, 1080), "white")

        # Save the image (initially it will be empty, needs a viewer to render content)
        slide_image.save(image_path)

    print("Slides have been converted to images.")


def process_image(image_path, client):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt4o-consultancy-project",
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



def stack_outputs(outputs):
    """Combine all assistant responses into a single output."""
    return "\n\n".join(outputs)


def process_pdf_file(file_path, max_workers=4):
    """Main function to process PDF or PPT files."""
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        images = convert_pdf_to_images(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or PPT/PPTX file.")

    responses = []

    # --------- Asyc execution ------------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(process_image, image, client): idx for idx, image in
                          enumerate(images)}
        for future in as_completed(future_to_page):
            idx = future_to_page[future]
            try:
                response = future.result()
                responses.append((idx, f"--- Page {idx + 1} ---\n{response}"))
                print(f"Completed processing page {idx + 1}")
            except Exception as exc:
                print(f"Page {idx + 1} generated an exception: {exc}")

    # Sort responses by page number
    responses_sorted = sorted(responses, key=lambda x: x[0])
    final_output = stack_outputs([resp for idx, resp in responses_sorted])
    return final_output




if __name__ == "__main__":
    # Process a PDF file
    file_path = "20240517_PostNL_GenAI Taskforce_Summary_Deck.pdf"
    output = process_pdf_file(file_path)
    print(output)
    print("Processing complete.")