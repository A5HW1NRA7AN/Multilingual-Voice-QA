import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import time

# Ensure the temporary audio directory exists
if not os.path.exists("assets/temp_audio"):
    os.makedirs("assets/temp_audio")

def listen_and_transcribe(lang="en-US"):
    """
    Captures audio from the microphone and transcribes it using Whisper.

    Args:
        lang (str): The language code for transcription (e.g., 'en-US', 'ja-JP').
                    Note: Whisper auto-detects, but this can be a hint.

    Returns:
        str: The transcribed text, or an error message.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please ask your question.")
        r.pause_threshold = 1.0 # seconds of non-speaking audio before a phrase is considered complete
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        st.info("Transcribing speech...")
        # Using the whisper model for transcription
        text = r.recognize_whisper(audio, language=lang.split('-')[0])
        st.success(f"Transcribed Question: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Whisper could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Whisper service; {e}")
        return None

def text_to_speech(text, lang_code="en"):
    """
    Converts text to speech using gTTS and returns the path to the audio file.

    Args:
        text (str): The text to be converted to speech.
        lang_code (str): The language code for gTTS (e.g., 'en', 'sa', 'ja').

    Returns:
        str: The file path of the generated audio file.
    """
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        # Use a timestamp to create a unique filename
        timestamp = int(time.time())
        audio_file = f"assets/temp_audio/response_{lang_code}_{timestamp}.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Failed to generate speech: {e}")
        return None