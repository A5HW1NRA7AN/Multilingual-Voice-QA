import streamlit as st
from core.qa_pipeline import load_model_components, extract_text_from_pdf, get_answer
from core.voice_handler import listen_and_transcribe, text_to_speech
from rouge_score import rouge_scorer
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual Voice QA System",
    page_icon="🎙️",
    layout="wide"
)

# --- Model and Language Configuration ---
MODEL_CONFIG = {
    "English": {"model_name": "google/flan-t5-base", "lang_code": "en", "stt_lang": "en-US"},
    "Sanskrit": {"model_name": "google/muril-base-cased", "lang_code": "sa", "stt_lang": "sa-IN"},
    # --- REVERTING TO THE MOST STABLE & CANONICAL JAPANESE MODEL ---
    "Japanese": {"model_name": "cl-tohoku/bert-base-japanese-whole-word-masking", "lang_code": "ja", "stt_lang": "ja-JP"},
}

DEFAULT_PDF_PATHS = {
    "English": "assets/pdfs/moon_en.pdf",
    "Sanskrit": "assets/pdfs/moon_sa.pdf",
    "Japanese": "assets/pdfs/moon_ja.pdf",
}

# --- Session State Initialization ---
if 'answer_data' not in st.session_state:
    st.session_state.answer_data = None
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'ui_state' not in st.session_state:
    st.session_state.ui_state = "idle"

# --- UI: Sidebar ---
st.sidebar.title("⚙️ Configuration")
selected_language = st.sidebar.selectbox("Choose Language", list(MODEL_CONFIG.keys()), key="language_select")

st.sidebar.markdown(f"**Model:** `{MODEL_CONFIG[selected_language]['model_name']}`")
st.sidebar.info("The selected model will be downloaded and cached on its first use.")

use_default_pdf = st.sidebar.checkbox("Use Default 'Moon' PDF", value=True)
uploaded_file = st.sidebar.file_uploader("Or Upload your own PDF", type="pdf", disabled=use_default_pdf)

# --- UI: Main App ---
st.title("🗣️ Comparative Study of Voice-Based Document QA")
st.markdown("Select a language, ask a question via text or voice, and get a spoken answer from the document.")

# --- Functions to run QA ---
def process_question(model_comps, question_text, doc_text):
    st.session_state.question = question_text
    answer_data = get_answer(model_comps, st.session_state.question, doc_text)
    st.session_state.answer_data = answer_data
    st.session_state.ui_state = "done"

# Load model components
model_components = load_model_components(MODEL_CONFIG[selected_language]["model_name"])

# Handle PDF input
pdf_text = None
if use_default_pdf:
    default_path = DEFAULT_PDF_PATHS[selected_language]
    if os.path.exists(default_path):
        with open(default_path, "rb") as f:
            pdf_text = extract_text_from_pdf(f)
    else:
        st.error(f"Default PDF not found at: {default_path}")
elif uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)

# --- Main Interaction Area ---
if pdf_text:
    with st.expander("View Full Text from PDF"):
        st.text_area("Extracted Text", pdf_text, height=250)
    
    st.header("1. Ask a Question")

    if st.session_state.ui_state == "idle":
        text_question = st.text_input("Type your question here:", key="text_q")
        if st.button("Submit Text Question"):
            if text_question:
                process_question(model_components, text_question, pdf_text)
                st.rerun()
            else:
                st.warning("Please enter a question.")

        st.markdown("<h5 style='text-align: center; color: grey;'>OR</h5>", unsafe_allow_html=True)
        
        if st.button("🎤 Start Recording"):
            st.session_state.ui_state = "recording"
            st.rerun()

    if st.session_state.ui_state == "recording":
        stt_lang_code = MODEL_CONFIG[selected_language]["stt_lang"]
        transcribed_question = listen_and_transcribe(lang=stt_lang_code)
        if transcribed_question:
            process_question(model_components, transcribed_question, pdf_text)
        else:
            st.session_state.ui_state = "idle"
        st.rerun()

    if st.session_state.ui_state == "done":
        st.write(f"**Your Question:** *{st.session_state.question}*")
        
        st.header("2. Answer from Document")
        ans_data = st.session_state.answer_data
        
        st.success(f"**Answer:** {ans_data['answer']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Model Confidence Score", value=f"{ans_data['score']:.2%}")
        
        with col2:
            st.write("**Voice Output:**")
            tts_lang_code = MODEL_CONFIG[selected_language]["lang_code"]
            audio_file_path = text_to_speech(ans_data['answer'], lang_code=tts_lang_code)
            if audio_file_path:
                st.audio(audio_file_path, format='audio/mp3')

        st.header("3. Evaluation Metrics")
        st.markdown("---")
        
        with st.expander("Show Raw Answer Context"):
            st.json(ans_data)

        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            st.subheader("🤖 Automated Evaluation (ROUGE)")
            reference_answer = st.text_input("Provide a 'Gold Standard' reference answer (optional):")
            if reference_answer and ans_data['answer']:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(reference_answer, ans_data['answer'])
                
                rouge_df = pd.DataFrame({
                    'Metric': ['Precision', 'Recall', 'F-Measure'],
                    'ROUGE-1': [scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure],
                    'ROUGE-2': [scores['rouge2'].precision, scores['rouge2'].recall, scores['rouge2'].fmeasure],
                    'ROUGE-L': [scores['rougeL'].precision, scores['rougeL'].recall, scores['rougeL'].fmeasure],
                }).set_index('Metric')
                st.dataframe(rouge_df.style.format("{:.2f}"))
        
        with eval_col2:
            st.subheader("🧑‍💻 Human Evaluation")
            correctness = st.slider("Answer Correctness", 1, 5, 3, help="Is the answer factually correct according to the text?")
            fluency = st.slider("Answer Fluency", 1, 5, 3, help="Is the answer grammatically correct and easy to understand?")
            voice_clarity = st.slider("Voice Output Clarity", 1, 5, 3, help="How clear and natural was the pronunciation?")
            
            human_scores = pd.DataFrame({
                'Metric': ['Correctness', 'Fluency', 'Voice Clarity'],
                'Score': [correctness, fluency, voice_clarity]
            })
            st.bar_chart(human_scores.set_index('Metric'))
        
        if st.button("🔄 Ask Another Question"):
            st.session_state.question = ""
            st.session_state.answer_data = None
            st.session_state.ui_state = "idle"
            st.rerun()

else:
    st.warning("Please upload a PDF or select the default option to begin.")
