import gradio as gr
import pytesseract
import cv2
import numpy as np
import os
import openai
from dotenv import load_dotenv

# Ensure the NLTK resource is available
import nltk

nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up the path to Tesseract executable (Update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image_for_ocr(image):
    """Preprocess the image for OCR to improve accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to double the original size for better accuracy
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Remove noise with median blur
    gray = cv2.medianBlur(gray, 3)

    # Apply adaptive thresholding
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    # Apply dilation and erosion to remove small noise
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)

    return gray


def extract_text_from_image(image):
    """Extract text from an image using OCR."""
    processed_image = preprocess_image_for_ocr(image)
    # Use Tesseract to do OCR on the image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng')
    return text


def classify_content(query):
    """Use OpenAI to classify the content based on specified parameters."""
    prompt = (
        f"The following text has been extracted from a drug fact label. Analyze the content and answer the questions below:\n\n"
        f"{query}\n\n"
        "Questions:\n"
        "1. What is the active ingredient and its purpose?\n"
        "2. What are the primary uses of this medication?\n"
        "3. List any warnings or precautions mentioned.\n"
        "4. What are the directions for use?\n"
        "5. What are the storage conditions?\n"
        "6. Identify any inactive ingredients listed.\n"
        "7. Who is the targeted audience for this drug (e.g., Adults, Children, Elderly, Healthcare Professionals)?\n"
        "8. Is there any specific information about overdose management?\n"
        "9. Does this label mention any specific patient advice (e.g., consultation with doctor, prohibited activities)?\n"
        "10. Does the content have a disclaimer? (Yes or No). If yes, mention the disclaimer.\n"
        "11. Is there any information on drug interactions?\n"
        "12. Does the content require a reviewer or approver? (Yes or No).\n"
        "13. Any other relevant details per healthcare or regulatory guidelines."
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are an assistant that specializes in analyzing and categorizing content extracted from images representing slides, videos, or documents. Provide clear and concise answers to the questions based on the provided text."},
            {"role": "user", "content": prompt}
        ]
    )
    assistant_response = response.choices[0].message.content.strip()
    return assistant_response


def calculate_score(text):
    """Calculate score using VADER with default thresholds."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']

    # Default thresholds
    positive_threshold = 0.05
    neutral_threshold = -0.05

    # Use default values for scoring
    if compound_score >= positive_threshold:
        return 0  # Positive sentiment
    elif compound_score > neutral_threshold:
        return -1  # Neutral sentiment
    else:
        return -3  # Negative sentiment


def process_image(image):
    """Process the uploaded image and return the analysis."""
    try:
        # Convert PIL image to OpenCV format
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        extracted_text = extract_text_from_image(image)
        if not extracted_text.strip():
            return "No text found in the image. Please try another image.", None

        classified_content = classify_content(extracted_text)
        score = calculate_score(classified_content)
        return classified_content, score
    except Exception as e:
        return f"Error processing image: {str(e)}", None


# Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Classified Content"),
        gr.Textbox(label="Sentiment Score"),
    ],
    title="Image Text Extraction and Analysis",
    description="Upload an image (e.g., a slide from a presentation, a frame from a video, or a page from a PDF) to extract text and analyze its content.",
)

if __name__ == "__main__":
    interface.launch()
