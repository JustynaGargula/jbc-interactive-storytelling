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

    if st.session_state.get("language") == "en":
        all_english_types = read_names_from_file("document_types", "en")
        all_polish_types = read_names_from_file("document_types", "pl")
        all_english_subjects = read_names_from_file("subjects", "en")
        all_polish_subjects = read_names_from_file("subjects", "pl")
        for doc in data:
            doc_type = translate_names_to_english([doc.type], all_english_types, all_polish_types)[0]
            doc_subjects = ', '.join(translate_names_to_english(doc.subjects, all_english_subjects, all_polish_subjects))
            timeline_data.append({
                'title': doc.title,
                'year': doc.year,
                'date_display': doc.get_date_display(),
                'subjects': doc_subjects,
                'type': doc_type,
                'url': doc.identifier,
            })
    else:
        for doc in data:
            subjects = ', '.join(doc.subjects)
            timeline_data.append({
                'title': doc.title,
                'year': doc.year,
                'date_display': doc.get_date_display(),
                'subjects': subjects,
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

def get_markdown_link_list(source_ids: List[str]) -> str:
    """
    Generates a markdown-formatted list of links based on the provided source IDs.

    :param source_ids: List of source identifiers (e.g., document IDs)
    :type source_ids: List[str]
    :return: A string containing markdown-formatted links
    :rtype: str
    """
    if not source_ids:
        return ""

    link_list = []
    for i, source_id in enumerate(source_ids):
        if source_id[0:5] == "https":
            urls = source_id.split("https")
            url  = "https" + urls[-1]
            link_list.append(f"[link{i+1}]({url})")

    return ", ".join(link_list)

def read_names_from_file(category, language) -> List[str] | dict[str, List[str]]:
    """
    Reads names from a text file and returns them as a list.
    :param category: The category of names to read ("subjects", "grouped_topics" or "document_types")
    :type category: str
    :param language: The language of the names to read ("pl" or "en")
    :type language: str

    :return: List of names
    :rtype: List[str] | dict[str, List[str]]
    """
    names = None
    if category == "subjects":
        names = []
        file_path = f"locales/subjects_{language}.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                names.append(line.strip())
    elif category == "grouped_topics":
        names = {}
        file_path = f"locales/grouped_topics_{language}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            names = json.load(f)

    elif category == "document_types":
        names = []
        file_path = f"locales/document_types_{language}.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                names.append(line.strip())
    return names

def translate_names_to_polish(names_to_translate: List[str], all_english_names: List[str], all_polish_names: List[str]) -> List[str]:
    """
    Translates a list of English names to Polish.

    :param names_to_translate: List of English names to translate
    :type names_to_translate: List[str]
    :param all_polish_names: List of all Polish names
    :type all_polish_names: List[str]
    :param all_english_names: List of all English names
    :type all_english_names: List[str]
    :return: List of translated Polish names
    :rtype: List[str]
    """
    translated_names = []
    for name in names_to_translate:
        if name in all_english_names:
            index = all_english_names.index(name)
            translated_names.append(all_polish_names[index])
        else:
            translated_names.append(name)
    return translated_names

def translate_names_to_english(names_to_translate: List[str], all_english_names: List[str], all_polish_names: List[str]) -> List[str]:
    """
    Translates a list of Polish names to English.

    :param names_to_translate: List of Polish names to translate
    :type names_to_translate: List[str]
    :param all_polish_names: List of all Polish names
    :type all_polish_names: List[str]
    :param all_english_names: List of all English names
    :type all_english_names: List[str]
    :return: List of translated Polish names
    :rtype: List[str]
    """
    translated_names = []
    for name in names_to_translate:
        if name in all_polish_names:
            index = all_polish_names.index(name)
            translated_names.append(all_english_names[index])
        else:
            translated_names.append(name)
    return translated_names