# Multilingual Voice-Based Document QA System 🗣️📄
This project is an interactive web application built with Streamlit for a comparative study of different transformer-based language models. It allows users to upload a PDF document, ask questions using either text or voice, and receive answers in both text and speech, enabling a direct comparison between Foundation, Indic, and International language models.

(Replace the placeholder with a screenshot of your running application)

# 🚀 Objective
The core objective of this project is to implement and analyze the performance of three distinct types of language models in a real-world Question-Answering (QA) task:

Foundation Model (English): A powerful, general-purpose model.

Indic Language Model (Sanskrit): A model designed for Indian languages.

International Language Model (Japanese): A model tailored for a non-English, non-Indic language.

The system evaluates these models on their ability to extract relevant answers from a document based on user queries provided via voice or text.

# ✨ Features
Multi-Language Support: Seamlessly switch between English, Sanskrit, and Japanese.

Dynamic Model Loading: Automatically loads the appropriate model based on the selected language.

Dual Input Methods: Ask questions by typing in a text box or by recording your voice directly in the browser.

Dual Output Methods: Receive answers as clear text and as a playable audio file.

PDF Document Processing: Upload your own PDF documents or use the provided default documents for a quick start.

Interactive UI: Built with Streamlit for a clean, user-friendly, and responsive web interface.

Built-in Evaluation:

Automated: Calculate ROUGE scores against a reference answer.

Human: Use interactive sliders to evaluate answer correctness, fluency, and voice clarity.

Context Viewer: Inspect the full text extracted from the PDF and the raw context used for generating the answer.

# 🤖 Models Used
The system leverages the following models from the Hugging Face Hub:

Language

Model Type

Model Name

English

Foundation (Generative)

google/flan-t5-base

Sanskrit

Indic (Extractive)

google/muril-base-cased

Japanese

International (Extractive)

cl-tohoku/bert-base-japanese-whole-word-masking

# 📂 Project Structure
voice-qa-system/
|
├── app.py                  # Main Streamlit application UI
├── README.md               # This file
|
├── core/
|   ├── __init__.py
|   ├── qa_pipeline.py      # PDF processing and QA logic
|   └── voice_handler.py    # Speech-to-Text and Text-to-Speech
|
├── assets/
|   ├── pdfs/
|   |   ├── moon_en.pdf
|   |   ├── moon_sa.pdf
|   |   └── moon_ja.pdf
|   └── temp_audio/         # Temporary storage for generated audio
|       └── .gitkeep
|
└── requirements.txt        # Project dependencies

# ⚙️ Setup and Installation
Follow these steps to set up and run the project locally.

1. Prerequisites
Python 3.9+

pip package manager

2. Clone the Repository
git clone <your-repository-url>
cd voice-qa-system

3. Create a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies
Install all required packages from the requirements.txt file.

pip install -r requirements.txt

Special Note for Windows Users (PyAudio):
The PyAudio library (for microphone access) can be difficult to install on Windows. If you encounter an error, install it manually:

Find your Python version (e.g., 3.11) and system type (64-bit).

Download the appropriate pre-compiled "wheel" (.whl) file from this trusted repository.

Install it using pip: pip install path/to/your/downloaded/PyAudio-file.whl.

Special Note for Japanese Model:
The Japanese model requires extra dependencies. If you encounter issues, ensure they are installed:

pip install fugashi[unidic-lite] mecab-python3

# ▶️ How to Run
Once the setup is complete, run the following command from the root directory of the project:

streamlit run app.py

Your web browser will automatically open with the application running. On the first run for each language, the corresponding model will be downloaded from Hugging Face, which may take a few minutes. Subsequent runs will be much faster as the models are cached.

# 📋 How to Use the Application
Select a Language: Use the sidebar to choose between English, Sanskrit, and Japanese.

Provide a Document: Either use the default "Moon" PDF (recommended) or upload your own PDF document.

Ask a Question:

Text: Type your question into the text box and click "Submit".

Voice: Click the "Start Recording" button, grant microphone access if prompted, and speak your question. The system will automatically detect when you stop speaking.

Review the Answer: The application will display the text answer and an audio player for the voice response.

Evaluate:

Optionally provide a "gold standard" answer to see the ROUGE score.

Use the sliders to perform a human evaluation of the result.

Click "Ask Another Question" to reset the interface for a new query.
