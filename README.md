# GAIA Agent

A multi-modal AI agent designed to tackle the GAIA Benchmark challenges.  
It can process text, images, audio, Excel/CSV files, and Python code to answer diverse questions. Thie code is structured to be run on Hugging face spaces.

## Features

- **Text Understanding:** Answer natural language questions.
- **Image Analysis:** Uses Google Gemini Vision API to reason about images.
- **Audio Processing:** Transcribes audio using OpenAI Whisper.
- **Python Execution:** Executes Python scripts or snippets safely via PythonREPL tool.
- **Tabular Data Handling:** Processes `.csv` and `.xlsx` files for arithmetic, sums, and analysis.
- **Extensible Tool System:** Uses LangGraph nodes and LangChain tools to structure reasoning and tool usage.

## Getting Started

### Requirements

```bash
pip install -r requirements.txt


### Project Structure
├─ app.py             # Main app entry point
├─ utils.py           # Utility functions
├─ tools.py           # Tool definitions (Wiki, Gemini Vision, Python REPL, etc.)
├─ prompt.txt         # System prompt for the assistant
├─ task_data/         # Local storage of task-related files
├─ requirements.txt   # Python dependencies
```