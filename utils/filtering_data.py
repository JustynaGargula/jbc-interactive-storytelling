
import json
import re
from typing import List
from models import Document, KnowledgeGraph
import streamlit as st
from .llm import handle_llm


def get_documents_from_filters(knowledge_graph: KnowledgeGraph, years: list, subjects: list) -> List[Document]:
    """
    Zwraca dokumenty pasujące do podanych filtrów.

    :param knowledge_graph: Graf wiedzy
    :type knowledge_graph: KnowledgeGraph
    :param years: Lista lat do filtrowania
    :type years: List[int]
    :param subjects: Lista tematów do filtrowania
    :type subjects: List[str]
    :return: Lista dokumentów pasujących do filtrów
    :rtype: List[Document]
    """
    docs_years = []

    for year in years or []:
        docs_years += knowledge_graph.get_documents_by_year(year) or []

    docs_subjects = []
    for subject in subjects or []:
        docs_subjects += knowledge_graph.get_documents_by_subject(subject) or []

    documents_ids = []
    if not docs_years and not docs_subjects:
        documents_ids = []
    elif docs_years and not docs_subjects:
        documents_ids = docs_years
    elif not docs_years and docs_subjects:
        documents_ids = docs_subjects
    else:
        documents_ids = list(set(docs_years) & set(docs_subjects))

    documents = []
    for id in documents_ids:
        doc = knowledge_graph.get_document_by_id(id)
        documents.append(doc)

    return documents


def get_documents_from_filters_and_related(knowledge_graph: KnowledgeGraph, years: list, subjects: list, max_related: int=10) -> List[Document]:
    """
    Zwraca dokumenty pasujące do podanych filtrów oraz powiązane z nimi dokumenty.
    :param knowledge_graph: Graf wiedzy
    :type knowledge_graph: KnowledgeGraph
    :param years: Lista lat do filtrowania
    :type years: List[int]
    :param subjects: Lista tematów do filtrowania
    :type subjects: List[str]
    :param max_related: Maksymalna liczba powiązanych dokumentów do dodania (domyślnie 10)
    :type max_related: int
    :return: Lista dokumentów pasujących do filtrów oraz powiązanych z nimi
    :rtype: List[Document]
    """
    selected_docs = get_documents_from_filters(knowledge_graph, years, subjects)

    def check_doc_in_list(doc, doc_list):
        for d, _ in doc_list:
            if d.identifier == doc.identifier:
                return True
        return False

    related_docs_with_scores = []
    for doc in selected_docs:
        related = knowledge_graph.get_related_documents(doc.identifier)
        for related_doc, score in related:
            if related_doc not in selected_docs and not check_doc_in_list(related_doc, related_docs_with_scores) and score >= 4.0:
                related_docs_with_scores.append((related_doc, score))

    related_docs = []
    if len(related_docs_with_scores) > max_related:
        related_docs_with_scores = sorted(related_docs_with_scores, key=lambda x: x[1], reverse=True)[:max_related]

    for doc, score in related_docs_with_scores:
        related_docs.append(doc)

    return selected_docs + related_docs


def get_data_based_on_selected_filters(selected_subject_names: list, selected_date_range: tuple, selected_related: bool, kg: KnowledgeGraph) -> List[Document]:
    """
    Zwraca dokumenty pasujące do wybranych filtrów.

    :param selected_subject_names: Lista nazw wybranych tematów
    :type selected_subject_names: list
    :param selected_date_range: Zakres dat (np. (1800, 1900))
    :type selected_date_range: tuple
    :param selected_related: Czy uwzględniać powiązane dokumenty
    :type selected_related: bool
    :param kg: Graf wiedzy
    :type kg: KnowledgeGraph
    :return: Lista dokumentów pasujących do filtrów
    :rtype: List[Document]
    """
    years = []

    if not selected_date_range:
        years = []
    elif type(selected_date_range) == tuple:
        years = list(range(selected_date_range[0], selected_date_range[1]+1))
    else:
        years = [selected_date_range]

    if selected_related:
        data = get_documents_from_filters_and_related(
            kg,
            years,
            selected_subject_names,
        )
    else:
        data = get_documents_from_filters(
            kg,
            years,
            selected_subject_names,
        )
    return data

def display_and_collect_subject_filters(page_text_part: dict, categorized_subject_names: dict[str, list[str]]) -> List[str]:
    st.subheader(page_text_part.get('subjects_filter_subheader'), help=page_text_part.get('subjects_filter_description'))
    col1, col2 = st.columns(2)
    selected_subject_names = []
    with col1:
        # Places
        places_names = categorized_subject_names.get("Places", [])
        places_names.sort()
        selected = st.multiselect(page_text_part.get("subjects_filter_places_label"), places_names, placeholder=page_text_part.get("subjects_filter_placeholder"), filter_mode="contains", help=page_text_part.get("subjects_filter_places_help_text"))
        for subj in selected:
            if subj not in selected_subject_names:
                selected_subject_names.append(subj)
        st.space("small")
        # Events
        events_names = categorized_subject_names.get("Events", [])
        events_names.sort()
        selected = st.multiselect(page_text_part.get("subjects_filter_events_label"), events_names, placeholder=page_text_part.get("subjects_filter_placeholder"), filter_mode="contains", help=page_text_part.get("subjects_filter_events_help_text"))
        for subj in selected:
            if subj not in selected_subject_names:
                selected_subject_names.append(subj)
    with col2:
    # People
        people_names = categorized_subject_names.get("People", [])
        people_names.sort()
        selected = st.multiselect(page_text_part.get("subjects_filter_people_label"), people_names, placeholder=page_text_part.get("subjects_filter_placeholder"), filter_mode="contains", help=page_text_part.get("subjects_filter_people_help_text"))
        for subj in selected:
            if subj not in selected_subject_names:
                selected_subject_names.append(subj)
        st.space("small")
        # Other
        other_names = categorized_subject_names.get("Other", [])
        other_names.sort()
        selected = st.multiselect(page_text_part.get("subjects_filter_other_label"), other_names, placeholder=page_text_part.get("subjects_filter_placeholder"), filter_mode="contains", help=page_text_part.get("subjects_filter_other_help_text"))
        for subj in selected:
            if subj not in selected_subject_names:
                selected_subject_names.append(subj)
    return selected_subject_names

def get_data_based_on_text_query(query: str, kg: KnowledgeGraph, selected_related: bool) -> List[Document]:
    """
    Zwraca dokumenty pasujące do zapytania tekstowego.

    :param query: Zapytanie tekstowe do wyszukania w dokumentach
    :type query: str
    :param kg: Graf wiedzy
    :type kg: KnowledgeGraph
    :param selected_related: Czy uwzględniać powiązane dokumenty
    :type selected_related: bool
    :return: Lista dokumentów pasujących do zapytania tekstowego
    :rtype: List[Document]
    """
    if not query:
        return None

    prompt = f"Na podstawie poniższego opisu tekstowego oraz dostępnej listy tematów dobierz tematy i daty pasujące do opisu. Nie wykonuj zadania z opisu, bo to zrobi inny moduł. Ty tylko znajdź pasujące tematy i daty. Opis: {query}. Lista tematów: {kg.get_all_subject_names()}. Jako odpowiedź zwróć mi słownik (json) w formacie: {{'selected_subject_names': [lista pasujących tematów z nazwami jak w załaczonej liście]'selected_date_range': [dopasowany zakres dat, np. (1800, 1900)]}}. Jeśli nie znajdziesz żadnych pasujących tematów lub dat, zwróć pustą listę dla tematów i None dla zakresu dat. Pamiętaj o zachowaniu kodowania UTF-8 w odpowiedzi, ponieważ tematy mogą zawierać polskie znaki. Odpowiedź powinna być tylko i wyłącznie tekstem w formacie json, bez żadnych dodatkowych komentarzy czy objaśnień."
    response = handle_llm(prompt)
    try:
        if response is not None:
            if response.startswith("```") or response.endswith("```"):
                response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
            response = json.loads(response)
            selected_subject_names = response.get("selected_subject_names", [])
            selected_date_range = response.get("selected_date_range", [])
            selected_date_range = tuple(selected_date_range) if selected_date_range else None

            documents = get_data_based_on_selected_filters(
                selected_subject_names,
                selected_date_range,
                selected_related,
                kg
            )
            return documents
        else:
            return None
    except Exception as e:
        print(f"Error decoding LLM response: {e}")
        return None
