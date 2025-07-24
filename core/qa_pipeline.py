import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pdfplumber

# Use Streamlit's cache to load models only once
@st.cache_resource(show_spinner=False) # Spinner is handled in the main app now
def load_model_components(model_name, tokenizer_name=None):
    """
    Loads all necessary components for a model: pipeline, raw model, or tokenizer.
    This is a more flexible approach to handle different model types.
    """
    if tokenizer_name is None:
        tokenizer_name = model_name
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.info(f"Loading model: {model_name} on {device}...")

    # Load FLAN-T5 for a generative approach
    if "flan-t5" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        st.success(f"Generative Model '{model_name}' loaded successfully!")
        return {"model": model, "tokenizer": tokenizer, "pipeline": None, "device": device}
    
    # Load other models into an extractive QA pipeline
    else:
        qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=tokenizer_name,
            device=device
        )
        st.success(f"Extractive Model '{model_name}' loaded successfully!")
        return {"pipeline": qa_pipeline, "model": None, "tokenizer": None}


def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file using pdfplumber for robustness."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def get_answer(model_components, question, pdf_text):
    """
    Finds an answer using the appropriate method (generative for T5, extractive for others).
    The extractive method now uses a more robust top-k search across all chunks.
    """
    # --- Generative Path for FLAN-T5 ---
    if model_components.get("model") and "t5" in model_components["model"].name_or_path:
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        device = model_components["device"]
        
        input_text = f"question: {question} context: {pdf_text}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        with st.spinner("Generating answer with FLAN-T5..."):
            outputs = model.generate(**inputs, max_length=200, num_beams=4, early_stopping=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'answer': answer if answer else "FLAN-T5 could not generate an answer.",
            'score': 1.0,
            'context': "Answer was generated, not extracted. The model used the full document as context."
        }
    
    # --- Robust Extractive Path for BERT-based Models ---
    else:
        qa_pipeline = model_components["pipeline"]
        max_chunk_len = 512
        overlap = 100

        chunks = []
        if pdf_text and isinstance(pdf_text, str):
            for i in range(0, len(pdf_text), max_chunk_len - overlap):
                chunks.append(pdf_text[i:i + max_chunk_len])
        
        if not chunks:
            return {'answer': "The document appears to be empty.", 'score': 0, 'context': "N/A"}

        all_candidates = []
        with st.spinner(f"Searching for top answers in {len(chunks)} document chunks..."):
            for chunk in chunks:
                try:
                    results = qa_pipeline(question=question, context=chunk, top_k=3, handle_impossible_answer=True)
                    
                    # --- BUG FIX: Manually add the context to each result ---
                    # The pipeline result doesn't include the context it was given, so we add it back.
                    for result in results:
                        result['context'] = chunk
                    
                    all_candidates.extend(results)
                except Exception as e:
                    print(f"Error processing a chunk: {e}")
                    continue
        
        valid_candidates = [cand for cand in all_candidates if cand.get('answer')]

        if not valid_candidates:
            return {'answer': "Sorry, I couldn't find a confident answer in the document.", 'score': 0, 'context': "N/A"}

        best_answer = max(valid_candidates, key=lambda x: x['score'])
        
        return {
            'answer': best_answer['answer'],
            'score': best_answer['score'],
            'context': best_answer['context']
        }
