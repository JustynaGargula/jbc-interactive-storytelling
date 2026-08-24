import json
from typing import List, Optional
import uuid
import pandas as pd
import streamlit as st
from models import Document

def print_llm_usage(
        provider: str,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
) -> None:
    """
    Prints successful LLM usage to the Streamlit server console.
    """
    print(
        "[LLM] "
        f"provider={provider} | "
        f"model={model} | "
        f"prompt_tokens={prompt_tokens} | "
        f"completion_tokens={completion_tokens} | "
        f"total_tokens={total_tokens} | "
        f"cost={cost}",
        flush=True,
    )

def display_error(error_code: int, page_text_part: dict, e: Exception):
    st.error(page_text_part.get(f"error_{error_code}"))
    st.write(page_text_part.get(f"info_{error_code}"))
    with st.expander(page_text_part.get("other_error_details")):
        st.code(str(e))

def convert_data_to_dataframe(data: List[Document]) -> pd.DataFrame:
    """
    Konwertuje listę dokumentów na DataFrame Pandas do wykorzystania np. w osi czasu.

    :param data: Lista dokumentów do konwersji
    :type data: List[Document]
    :return: DataFrame zawierający dane dokumentów
    :rtype: DataFrame
    """
    timeline_data = []
    if not data:
        return None

    for doc in data:
        timeline_data.append({
            'title': doc.title,
            'year': doc.year,
            'date_display': doc.get_date_display(),
            'subjects': ', '.join(doc.subjects[:3]),  # pierwsze 3 tematy
            'type': doc.type,
            'url': doc.identifier,
        })

    df = pd.DataFrame(timeline_data)
    df = df[df['year'].notna()] # usuwa wiersze bez roku
    df = df.sort_values('year')
    return df

def show_button_status(placeholder, text):
    with placeholder:
        st.info(text)

def get_or_create_session_id() -> str:
    """
    Retrieves the existing session ID from the Streamlit session state.
    If it does not exist, generates a new UUID4 and stores it.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    return st.session_state.session_id

def load_page_text_in_chosen_language(language: str) -> dict:
    translation_file_path = f"locales/{language}.json"
    with open(translation_file_path, "r", encoding="utf-8") as f:
        json_text = json.load(f)
        st.session_state["page_text"] = json_text

