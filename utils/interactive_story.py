import json
from pathlib import Path
import re
from typing import List, Optional
import streamlit as st
from models import Document
from .llm import handle_llm


def generate_interactive_story_from_data(data: List[Document], user_prompt: Optional[str] = None) -> Optional[str]:
    """
    Generuje interaktywną opowieść na podstawie podanych dokumentów.

    :param data: Lista dokumentów do wygenerowania interaktywnej opowieści
    :type data: List[Document]
    :param user_prompt: Dodatkowy opis od użytkownika o tym, co chce przeczytać
    :type user_prompt: Optional[str]
    :return: Wygenerowana interaktywna opowieść lub None, jeśli dane są puste
    :rtype: str | None
    """
    if not data:
        return None
    page_text_part = st.session_state["page_text"].get("utils_generate_interactive_story_from_data")

    story_template_path = Path("data/stories/story_template.json")
    with open(story_template_path, "r", encoding="utf-8") as f:
        story_template = json.load(f)
    prompt = f"{page_text_part.get('prompt_pt1')} {data}{page_text_part.get('prompt_pt2')} {str(story_template)}"
    if user_prompt:
        prompt += f" {page_text_part.get('prompt_pt3')} {user_prompt}"
    response_text = handle_llm(prompt)
    try:
        if response_text is not None:
            response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip())
            return json.loads(response_text)
        else:
            return None
    except Exception as e:
        print(f"Błąd podczas analizy odpowiedzi JSON: {e}")
        return response_text

def display_interactive_story(story: str):
    """
    Wyświetla interaktywną opowieść.
    :param story: Tekst interaktywnej opowieści do wyświetlenia
    :type story: str
    """
    page_text = st.session_state["page_text"].get("utils_display_interactive_story")
    st.header(story.get("title"))
    st.write(story.get("description"))

    if st.session_state.get("choices_path") is None:
        st.session_state["choices_path"] = []
        if story.get("choices"):
            choices = story.get("choices")
            story_depth = 1
            while choices[0].get("choices"):
                story_depth += 1
                choices = choices[0].get("choices")
        else:
            story_depth = 0
        st.session_state["story_depth"] = story_depth
        st.rerun() # odświeża strone, żeby załadować dane do paska wyboru

    if st.session_state["story_depth"] > 0 and len(st.session_state["choices_path"]) < st.session_state["story_depth"]:
        display_choices(get_choices_or_ending(story, "current"), page_text.get("choice_button"))
    else:
        with st.container(border=True):
            st.subheader(page_text.get("story_ending"), text_alignment="center")
            st.write(get_choices_or_ending(story, "ending"))

    with st.container(horizontal=True, horizontal_alignment="center"):
        if st.button(page_text.get("rewind_button")):
            if len(st.session_state["choices_path"]) > 0:
                st.session_state["choices_path"] = st.session_state["choices_path"][:-1]
                st.rerun()
        if st.button(page_text.get("reset_button")):
            reset_interactive_story_to_first_choice()
            st.rerun()

def get_choices_or_ending(story: str, type: str) -> Optional[List[dict]]:
    """
    Zwraca aktualne opcje wyboru na podstawie głębokości opowieści i zapisanej ścieżki wyborów.
    :param story: Tekst interaktywnej opowieści
    :type story: str
    :param type: Typ wyborów do zwrócenia ("current" lub "previous" lub "ending")
    :type type: str
    :return: Lista aktualnych opcji wyboru
    :rtype: List[dict]
    """
    choices = story.get("choices")
    if type == "previous":
        for choice_index in st.session_state["choices_path"][:-1]:
            choices = choices[choice_index].get("choices")
    elif type == "current":
        for choice_index in st.session_state["choices_path"]:
            choices = choices[choice_index].get("choices")
    elif type == "ending":
        for choice_index in st.session_state["choices_path"][:-1]:
            choices = choices[choice_index].get("choices")
        ending = choices[st.session_state["choices_path"][-1]].get("ending")
        return ending
    return choices

def display_choices(choices: List[dict], choice_text: str):
    """
    Wyświetla opcje wyboru dla interaktywnej opowieści.

    :param choices: Lista słowników reprezentujących opcje wyboru
    :type choices: List[dict]
    :param choice_text: Tekst do wyświetlenia na przycisku wyboru
    :type choice_text: str
    """
    for i, choice in enumerate(choices):
        with st.container(border=True):
            st.subheader(choice.get("option_title"))
            st.write(choice.get("option_description"))
            if st.button(choice_text, key=f"choice_{len(st.session_state['choices_path'])}_{i}"):
                st.session_state["choices_path"].append(i)
                st.rerun() # odświeża stronę, żeby pokazać kolejne opcje

def reset_interactive_story_completely():
    """
    Resetuje stan interaktywnej opowieści, umożliwiając rozpoczęcie od nowa.
    """
    st.session_state["story_depth"] = None
    st.session_state["choices_path"] = None
    st.session_state["interactive_story"] = None

def reset_interactive_story_to_first_choice():
    """
    Resetuje stan interaktywnej opowieści do pierwszego wyboru.
    """
    st.session_state["choices_path"] = []
