import os
import re

import pandas as pd
from transformers import pipeline

# constants
TASK_DATA_PATH = "./task_data/"


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
            with open(file_path, "rb") as f:
                file_content = f.read()
            task_description = f"{task_text}\n\nAttached file content:"  # TODO: implement vision capanbilities
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
    task_description = preprocess_file_for_agent(
        task_text=test_question["question"], task_file_name=test_question["file_name"]
    )
    print(f"The task description for the given quesntion\n\n{task_description}")
