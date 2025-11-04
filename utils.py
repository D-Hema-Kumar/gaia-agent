import os
import re

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from transformers import pipeline

# constants
TASK_DATA_PATH = "./task_data/"
load_dotenv()


def format_response(response: str) -> str:
    formatted_response = re.search(r"FINAL ANSWER:\s*(.*)", response, re.IGNORECASE)

    if formatted_response:
        return formatted_response.group(1).strip()

    else:
        return response


def transcribe_audio_file(file_path: str) -> str:
    """Reads audio files and transcribes"""
    asr_pipeline = pipeline(
        "automatic-speech-recognition", model="openai/whisper-base", language="en"
    )
    result = asr_pipeline(file_path, return_timestamps=True)
    return result["text"]


def analyze_image(image_path: str, question_text: str) -> str:
    """
    Answers quetions about a given image.
    Use when an image is given to you in the question.
    Args:
        image_path (str): path to where the image is stored.
        question (str): The question you are asking about the image.
    """
    print(f"Analyzing image at:{image_path}")
    try:
        with open(image_path, "rb") as f:
            file_content = f.read()
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=file_content, mime_type="image/png"),
                question_text,
            ],
        )

        return response.text
    except Exception as e:
        return f"Error {str(e)}"


def preprocess_file_for_agent(task_text: str, task_file_name: str) -> str:
    """
    Detects file modality and converts content to text for the agent.

    Args:
        task_text: Text with the actual question.
        task_file_name: The filename attached to the question.

    Returns:
        str: The full question text plus extracted file content (if any).
    """

    if not task_file_name:
        return task_text

    file_path = os.path.join(TASK_DATA_PATH, task_file_name)
    if not os.path.exists(file_path):
        print(f"File NOT found locally: {file_path}")
        return task_text

    extension = os.path.splitext(file_path)[-1].lower()

    try:
        if extension in [".txt", ".py", ".json", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
            file_content = file_content.replace('if __name__ == "__main__":', "")
            task_description = f"{task_text}\n\nAttached file content:\n{file_content}"
            return task_description

        elif extension in [".xlsx", ".csv"]:
            if extension == ".csv":
                file_content = pd.read_csv(file_path)
            else:
                file_content = pd.read_excel(file_path)
            task_description = f"{task_text}\n\nAttached file content:\n{file_content.to_string(index=False)}"
            return task_description

        elif extension in [".mp3"]:
            print("mp3 file is being processed")
            file_content = transcribe_audio_file(file_path)
            task_description = f"{task_text}\n\nAttached file content:\n{file_content}"
            return task_description

        elif extension in [".jpeg", ".jpg", ".png"]:
            answer = analyze_image(image_path=file_path, question_text=task_text)
            task_description = f"{task_text}\n\nanalyzed file content:{answer}"  # TODO: implement vision capanbilities
            return task_description

    except Exception as e:
        print(f"Error processing {task_file_name}: {e}")
        return task_text


if __name__ == "__main__":
    test_question = {
        "task_id": "1f975693-876d-457b-a649-393859e79bf3",
        "question": "Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(\n\nCould you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order.",
        "Level": "1",
        "file_name": "1f975693-876d-457b-a649-393859e79bf3.mp3",
    }
    test_question = {
        "task_id": "cca530fc-4052-43b2-b130-b30968d8aa44",
        "question": "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.",
        "Level": "1",
        "file_name": "cca530fc-4052-43b2-b130-b30968d8aa44.png",
    }
    task_description = preprocess_file_for_agent(
        task_text=test_question["question"], task_file_name=test_question["file_name"]
    )
    print(f"The task description for the given quesntion\n\n{task_description}")
    # client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # for model in client.models.list():
    #    print(model.name)
