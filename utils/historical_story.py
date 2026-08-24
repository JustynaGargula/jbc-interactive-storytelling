from typing import List, Optional
from models import Document
import streamlit as st
from .llm import handle_llm


def generate_historical_story_from_data(data: List[Document], user_prompt) -> Optional[str]:
    """
    Generuje historyczną opowieść na podstawie podanych dokumentów.

    :param data: Lista dokumentów do wygenerowania historycznej opowieści
    :type data: List[Document]
    :param user_prompt: Prompt wprowadzony przez użytkownika
    :type user_prompt: str
    :return: Wygenerowana historyczna opowieść lub None, jeśli dane są puste
    :rtype: str | None
    """
    if not data:
        return None
    page_text_part = st.session_state["page_text"].get("utils_generate_historical_story_from_data")
    prompt = f"{page_text_part.get('prompt_pt1')} {data}{page_text_part.get('prompt_pt2')}"
    if user_prompt:
        prompt += f" {page_text_part.get('prompt_pt3')} {user_prompt}"
    response_text = handle_llm(prompt)
    return response_text
