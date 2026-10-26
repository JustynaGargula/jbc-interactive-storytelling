import json
from typing import List, Optional
from models import Document
import streamlit as st
from .llm import get_llm_config, handle_llm
from .general import get_markdown_link_list
from google.genai import types


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
    story_schema = get_story_schema()

    response_text = handle_llm(prompt, story_schema)
    if response_text is not None:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print("Błąd podczas przetwarzania odpowiedzi od LLM.")
            print(f"Więcej o błędzie: {e}")
            stars = "*" * 20
            print(f"{stars}\n Wygenerowana opowieść: {response_text} \n{stars}")
            return None
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd: {e}")
            return None
    else:
        print("Brak odpowiedzi od LLM")
        return None

def display_historical_story(story):
    sources_word = st.session_state["page_text"].get("general").get("sources")

    for paragraph in story.get("paragraphs"):
        source_links = get_markdown_link_list(paragraph.get("source_ids"))
        col1, col2 = st.columns([0.9, 0.1])
        col1.write(paragraph.get("text"))
        col2.write(f"{sources_word}: {source_links}")

def get_story_schema():
    story_schema = None
    llm_provider = get_llm_config()["provider"]

    if llm_provider == "openrouter":
        story_schema = {
            "type": "object",
            "properties": {
                "paragraphs": {
                "type": "array",
                "description": "Lista akapitów tworzących opowieść historyczną. Opowieść musi zawierać od 2 do 10 akapitów.",
                "minItems": 2,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                    "text": {
                        "type": "string",
                        "description": "Treść akapitu opowieści historycznej."
                    },
                    "source_ids": {
                        "type": "array",
                        "description": "Lista identyfikatorów dokumentów źródłowych wykorzystanych do napisania tego akapitu. Używaj wyłącznie identyfikatorów przekazanych w danych wejściowych. Jeśli akapit nie korzysta z żadnego źródła, zwróć pustą tablicę.",
                        "items": {
                        "type": "string"
                        }
                    }
                    },
                    "required": [
                    "text",
                    "source_ids"
                    ],
                    "additionalProperties": False
                }
                }
            },
            "required": [
                "paragraphs"
            ],
            "additionalProperties": False
            }
    elif llm_provider == "gemini":
        story_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "paragraphs": types.Schema(
                    type=types.Type.ARRAY,
                    min_items=2,
                    max_items=10,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "text": types.Schema(
                                type=types.Type.STRING,
                            ),
                            "source_ids": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(
                                    type=types.Type.STRING,
                                ),
                            ),
                        },
                        required=[
                            "text",
                            "source_ids",
                        ],
                    ),
                ),
            },
            required=[
                "paragraphs",
            ],
        )

    return story_schema
