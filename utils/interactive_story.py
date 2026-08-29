import json
from typing import List, Optional
import streamlit as st
from models import Document
from .llm import get_llm_config, handle_llm
from google.genai import types

def generate_interactive_story_from_data(data: List[Document], story_depth: int, choices_per_chapter: int, user_prompt: Optional[str] = None) -> Optional[str]:
    """
    Generuje interaktywną opowieść na podstawie podanych dokumentów.

    :param data: Lista dokumentów do wygenerowania interaktywnej opowieści
    :type data: List[Document]
    :param story_depth: Głębokość opowieści (liczba etapów)
    :type story_depth: int
    :param choices_per_chapter: Liczba wyborów w każdym etapie opowieści
    :type choices_per_chapter: int
    :param user_prompt: Dodatkowy opis od użytkownika o tym, co chce przeczytać
    :type user_prompt: Optional[str]
    :return: Wygenerowana interaktywna opowieść lub None, jeśli dane są puste
    :rtype: str | None
    """
    if not data:
        return None
    page_text_part = st.session_state["page_text"].get("utils_generate_interactive_story_from_data")

    prompt = f"{page_text_part.get('prompt_pt1')} {data}{page_text_part.get('prompt_pt2')}"
    if user_prompt:
        prompt += f" {page_text_part.get('prompt_pt3')} {user_prompt}"

    story_schema = get_story_schema(story_depth, choices_per_chapter)
    response_text = handle_llm(prompt, interactive_story=True, story_schema=story_schema)

    if response_text is not None:
        return json.loads(response_text)
    else:
        print("Brak odpowiedzi od LLM lub błąd podczas generowania opowieści.")
        return None

def display_interactive_story(story: str):
    """
    Wyświetla interaktywną opowieść.
    :param story: Tekst interaktywnej opowieści do wyświetlenia
    :type story: str
    """
    page_text = st.session_state["page_text"].get("utils_display_interactive_story")
    st.header(story.get("title"), text_alignment="center")

    # initializing session variables
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

    # displaying story description or ending
    with st.container(border=True):
        description = st.empty()
    if len(st.session_state["choices_path"]) == 0:
        description.write(story.get("description"))
    elif st.session_state["story_depth"] > 0 and len(st.session_state["choices_path"]) < st.session_state["story_depth"]:
        prev_choice_id = st.session_state["choices_path"][-1]
        chosen_path_description = get_choices_or_ending(story, "previous")[prev_choice_id].get("option_description")
        description.write(chosen_path_description)
    else:
        story_end_text = page_text.get("story_ending")
        st.markdown(f"<div style='text-align:center; font-size:large;'> <i>✧ {story_end_text} ✧</i></div>", unsafe_allow_html=True)
        st.space("xsmall")
        description.write(get_choices_or_ending(story, "ending"))

    # displaying story choices or ending
    if st.session_state["story_depth"] > 0 and len(st.session_state["choices_path"]) < st.session_state["story_depth"]:
        display_choices(get_choices_or_ending(story, "current"), page_text.get("choice_button"))

    # rewind and reset buttons
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
    with st.container(horizontal=True):
        for i, choice in enumerate(choices):
            with st.container(border=True, height="stretch", vertical_alignment="center"):
                st.subheader(choice.get("option_title"))
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

def get_story_schema(story_depth: int, choices_per_chapter: int) -> dict:
    """
    Tworzy schemat JSON dla interaktywnej opowieści na podstawie głębokości i liczby wyborów w każdym rozdziale.

    :param depth: Głębokość opowieści (liczba rozdziałów)
    :type depth: int
    :param choices_per_chapter: Liczba wyborów w każdym rozdziale
    :type choices_per_chapter: int
    :return: Schemat JSON dla interaktywnej opowieści
    :rtype: dict
    """
    llm_provider = get_llm_config()["provider"]
    if llm_provider == "openrouter":
        story_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "choices": {
                    "type": "array",
                    "minItems": choices_per_chapter,
                    "maxItems": choices_per_chapter,
                    "items": create_choice_schema_for_openrouter(story_depth-1, choices_per_chapter),
                },
            },
            "required": [
                "title",
                "description",
                "choices",
            ],
            "additionalProperties": False,
        }
    elif llm_provider == "gemini":
        story_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "title": types.Schema(
                    type=types.Type.STRING,
                ),
                "description": types.Schema(
                    type=types.Type.STRING,
                ),
                "choices": types.Schema(
                    type=types.Type.ARRAY,
                    min_items=choices_per_chapter,
                    max_items=choices_per_chapter,
                    items=create_choice_schema_for_gemini(
                        story_depth - 1,
                        choices_per_chapter,
                    ),
                ),
            },
            required=[
                "title",
                "description",
                "choices",
            ],
        )
    return story_schema

def create_choice_schema_for_openrouter(depth: int, choices_per_chapter: int) -> dict:
    if depth <= 0:
        return {
            "type": "object",
            "properties": {
                "option_title": {"type": "string"},
                "ending": {"type": "string"},
            },
            "required": [
                "option_title",
                "ending",
            ],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "option_title": {"type": "string"},
            "option_description": {"type": ["string", "null"]},
            "ending": {"type": ["string", "null"]},
            "choices": {
                "type": "array",
                "minItems": choices_per_chapter,
                "maxItems": choices_per_chapter,
                "items": create_choice_schema_for_openrouter(depth - 1, choices_per_chapter),
            },
        },
        "required": [
            "option_title",
            "option_description",
            "ending",
            "choices",
        ],
        "additionalProperties": False,
    }

def create_choice_schema_for_gemini(depth: int, choices_per_chapter: int) -> dict:
    properties = {
        "option_title": types.Schema(
            type=types.Type.STRING,
        ),
        "option_description": types.Schema(
            type=types.Type.STRING,
        ),
    }

    if depth > 0:
        properties["ending"] = types.Schema(
            type=types.Type.STRING,
        )

        properties["choices"] = types.Schema(
            type=types.Type.ARRAY,
            min_items=choices_per_chapter,
            max_items=choices_per_chapter,
            items=create_choice_schema_for_gemini(
                depth - 1,
                choices_per_chapter,
            ),
        )
    else:
        properties["ending"] = types.Schema(
            type=types.Type.STRING,
        )

    return types.Schema(
        type=types.Type.OBJECT,
        properties=properties,
        required=list(properties.keys()),
    )