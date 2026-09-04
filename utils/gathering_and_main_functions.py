import glob
import json
from pathlib import Path
from typing import List
import streamlit as st
from models import Document, KnowledgeGraph, Relation, Subject
from .downloading_data_and_creating_graph import build_kg_from_rdf, create_graph, export_kg_to_jsonld, get_ids, get_rdfs, save_jsonld_to_file, save_rdfs_to_file
from .filtering_data import display_and_collect_subject_filters, get_data_based_on_selected_filters, get_data_based_on_text_query
from .general import convert_data_to_dataframe, show_button_status
from .interactive_story import display_interactive_story, generate_interactive_story_from_data, reset_interactive_story_completely
from .historical_story import generate_historical_story_from_data
from .timeline import generate_timeline

@st.cache_data(show_spinner=False)
def import_knowledge_graph_from_jsonld_file(jsonld_file: str) -> KnowledgeGraph:
    """
    Importuje graf wiedzy z pliku JSON-LD.

    :param jsonld_file: ścieżka do pliku JSON-LD zawierającego graf wiedzy
    :type jsonld_file: str
    :return kg: zaimportowany graf wiedzy
    :rtype: KnowledgeGraph
    """
    with open(jsonld_file, "r", encoding="utf-8") as f:
        jsonld_graph = json.load(f)

    kg = KnowledgeGraph()

    for item in jsonld_graph.get("@graph", []):
        if "Document" in item.get("@type", []):
            # print(f"Importuję dokument: {item.get('title')} (ID: {item.get('@id')})")
            doc = Document(
                identifier=item.get("@id"),
                title=item.get("title"),
                description=item.get("description", ""),
                subjects=item.get("subject", []),
                date_raw=item.get("date", ""),
                creator=item.get("creator", ""),
                publisher=item.get("publisher", ""),
                type=item.get("type", ""),
            )
            kg.add_document(doc) # to doda inne pola związane z datą

        elif "Subject" in item.get("@type", []):
            subject_id = item.get("@id").split("/")[-1]
            kg.subjects[subject_id] = Subject(
                name=item.get("name"),
                documents=item.get("documents", [])
            )

        elif "Relation" in item.get("@type", []):
            kg.relations.append(Relation(
                source_id=item.get("sourceId"),
                target_id=item.get("targetId"),
                relation_type=item.get("relationType"),
                weight=item.get("weight", 1.0)
            ))

    print(f"Zaimportowano graf wiedzy z {jsonld_file}, wczytano {len(kg.documents)} dokumentów, {len(kg.subjects)} tematów i {len(kg.relations)} relacji.")
    return kg

@st.cache_data(show_spinner=False)
def get_knowledge_graph_from_ris(ris_files_directory_path: str,  rdfs_directory_path: str, allowed_centuries: List[int], jsonld_output_file: str, already_downloaded_rdfs: bool = False, already_saved_jsonld: bool = False) -> KnowledgeGraph:
    """
    Tworzy graf wiedzy na podstawie pliku RIS i folderu z rdfami.

    :param ris_files_directory_path: Ścieżka do folderu z plikami RIS
    :type ris_files_directory_path: str
    :param rdfs_directory_path: Ścieżka do folderu z plikami RDF
    :type rdfs_directory_path: str
    :param jsonld_output_file: Ścieżka do pliku wyjściowego JSON-LD
    :type jsonld_output_file: str
    :param already_downloaded_rdfs: Czy pliki RDF zostały już pobrane
    :type already_downloaded_rdfs: bool
    :param already_saved_jsonld: Czy graf JSON-LD został już zapisany
    :type already_saved_jsonld: bool
    :return: Graf wiedzy
    :rtype: KnowledgeGraph
    """

    files = []
    for file in glob.glob(f"{ris_files_directory_path}" + "/*.ris"):
        files.append(file)
    ids = get_ids(files)
    rdfs_path = Path(rdfs_directory_path)

    if not already_downloaded_rdfs or not (rdfs_path.exists() and rdfs_path.is_dir() and any(rdfs_path.iterdir())):
        rdfs = get_rdfs(ids)
        save_rdfs_to_file(rdfs, ids, rdfs_directory_path)

    g = create_graph(rdfs_directory_path)
    # utils.save_data_to_one_file(g, "turtle", ".ttl")

    kg = build_kg_from_rdf(g, allowed_centuries)
    print(f"Wczytano {len(kg.documents)} dokumentów do grafu wiedzy.")

    if not already_saved_jsonld:
        jsonld_graph = export_kg_to_jsonld(kg)
        save_jsonld_to_file(jsonld_graph, jsonld_output_file)

    return kg

def display_interface_top_part():
    """
    Wyświetla górną część interfejsu użytkownika (tytuł i opis) w aplikacji Streamlit.
    """
    page_text_part = st.session_state["page_text"].get("utils_display_interface_top_part")
    st.title(page_text_part.get("main_header"))
    st.write(page_text_part.get("main_description"))
    st.space("small")


def display_interface_main_part(all_subject_names: List[str], dates__range: tuple, kg: KnowledgeGraph):
    """
    Wyświetla główną część interfejsu użytkownika w aplikacji Streamlit, umożliwiając wybór filtrów i generowanie opowieści lub osi czasu.

    :param all_subject_names: Lista wszystkich nazw tematów
    :type all_subject_names: List[str]
    :param dates__range: Zakres lat (np. (1800, 1900))
    :type dates__range: tuple
    :param kg: Graf wiedzy
    :type kg: KnowledgeGraph
    """
    page_text_part = st.session_state["page_text"].get("utils_display_interface_main_part")
    global interactive_story
    global user_query
    user_query = None
    interactive_story = None

    st.header(page_text_part.get("filters_header"))
    st.space("xsmall")

    # Elementy do wyboru, które potrzebują się odświeżać zanim formularz zostanie wysłany
    output_type, story_depth, choices_per_chapter = display_result_type_options(page_text_part)

    filters_choice = st.segmented_control(
        page_text_part.get("filters_choice_label"),
        options=page_text_part.get("filters_choice_options"),
        selection_mode="single", default=page_text_part.get("default_filters_choice"),
        width="stretch")
    st.space("xsmall")

    # Formularz z filtrami
    with st.form("filter_form", border=False):
        with st.container(border=True):
            # podstawowe filtry
            if filters_choice == page_text_part.get("filters_choice_options")[0]:
                if st.session_state.get("language") == "pl":
                    with open("locales/grouped_topics_pl.json", "r", encoding="utf-8") as f:
                        categorized_subject_names = json.load(f)
                    selected_subject_names = display_and_collect_subject_filters(page_text_part, categorized_subject_names)
                elif st.session_state.get("language") == "en":
                    all_english_subject_names = []
                    with open ("locales/subjects_en.txt", "r", encoding="utf-8") as f:
                        for line in f:
                            all_english_subject_names.append(line.strip())
                    with open("locales/grouped_topics_en.json", "r", encoding="utf-8") as f:
                        categorized_subject_names = json.load(f)

                    english_selected_subject_names = display_and_collect_subject_filters(page_text_part, categorized_subject_names)
                    st.write("*Notes: The subjects in English were tranlated by AI and may not be entirely accurate. The subjects in the generated story will be based on the Polish names, but you can select them using their English translations.*")

                    selected_subject_names = []
                    for subj in english_selected_subject_names:
                        index = all_english_subject_names.index(subj)
                        selected_subject_names.append(all_subject_names[index])

                fit_type = st.radio(page_text_part.get("fit_type_label"), page_text_part.get("fit_type_options"), horizontal=True, help=page_text_part.get("fit_type_help_text"))
                general_fit_type_names = ["or", "and"]
                fit_type = general_fit_type_names[page_text_part.get("fit_type_options").index(fit_type)]

                st.space("xxsmall")

                selected_date_range = st.slider(
                    page_text_part.get("date_range_label"),
                    min_value=dates__range[0],
                    max_value=dates__range[1],
                    value=dates__range,
                    help=page_text_part.get("date_range_help_text")
                )
                st.space("xxsmall")

            # zapytanie tekstowe
            elif filters_choice == page_text_part.get("filters_choice_options")[1]:
                user_query = st.text_area(page_text_part.get("query_filter_label"), height=200, placeholder=page_text_part.get("query_filter_placeholder"))
                st.space("xxsmall")

        # wspólne opcje
        selected_related = st.checkbox(page_text_part.get("related_documents_label"),
            help=page_text_part.get("related_documents_help_text"))
        st.space("xxsmall")

        button_status_placeholder = st.empty()
        generate_button = st.form_submit_button(page_text_part.get("generate_button_label"), on_click=show_button_status, args=(button_status_placeholder, page_text_part.get("button_clicked_info"),), type="primary", width="stretch")

    story_placeholder = st.empty()

    # generowanie
    if generate_button:
        with st.spinner(page_text_part.get("getting_data_spinner_text")):
            if filters_choice == page_text_part.get("filters_choice_options")[0]:
                data = get_data_based_on_selected_filters(
                    selected_subject_names,
                    selected_date_range,
                    selected_related,
                    kg,
                    fit_type=fit_type
                )
            elif filters_choice == page_text_part.get("filters_choice_options")[1]:
                data = get_data_based_on_text_query(user_query, kg, selected_related)
            if not data:
                if fit_type == "or":
                    st.warning(page_text_part.get("no_documents_warning"))
                else:
                    st.warning(page_text_part.get("no_documents_warning2"))
                return
            df = convert_data_to_dataframe(data)
        reset_interactive_story_completely()

        if output_type == page_text_part.get("historical_story"):
            with st.spinner(page_text_part.get("generating_story_spinner_text")):
                story = generate_historical_story_from_data(data, user_query)

            if story:
                st.divider()
                st.subheader(page_text_part.get("generated_story_header"))
                st.markdown(story)
            else:
                st.warning(page_text_part.get("no_story_warning"))

        elif output_type == page_text_part.get("interactive_story"):
            with st.spinner(page_text_part.get("generating_story_spinner_text")):
                story = generate_interactive_story_from_data(data, story_depth, choices_per_chapter, user_query)

            if story:
                st.session_state["interactive_story"] = story
            else:
                st.warning(page_text_part.get("no_story_warning"))

        elif output_type == page_text_part.get("timeline"):
            with st.spinner(page_text_part.get("generating_timeline_spinner_text")):
                timeline = generate_timeline(data)

            if timeline:
                st.divider()
                st.subheader(page_text_part.get("timeline_header"))
                st.plotly_chart(timeline, width="stretch")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(page_text_part.get("documents_number"), len(df))
                with col2:
                    st.metric(page_text_part.get("date_range_label"), f"{int(df['year'].min())} - {int(df['year'].max())}")
                with col3:
                    st.metric(page_text_part.get("documents_types_label"), len(df['type'].unique()))

            else:
                st.warning(page_text_part.get("no_timeline_warning"))

        else:
            st.error(page_text_part.get("no_story_type_warning"))

        if data:
            st.space("small")
            with st.expander(page_text_part.get("expander_text")):
                for idx, row in df.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"{row['subjects']}")
                    with col2:
                        st.text(row['date_display'])
                    with col3:
                        if row['url']:
                            st.link_button(page_text_part.get("open_document_button"), row['url'], width="stretch")
                    st.divider()
        button_status_placeholder = st.empty()

    if st.session_state.get("interactive_story"):
        with story_placeholder.container():
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(page_text_part.get("generated_story_header"))
            with col2:
                if st.session_state["story_depth"] and (st.session_state["choices_path"] is not None):
                    if st.session_state["story_depth"] == 0:
                        progress_value = 0
                        progress_text = f"{page_text_part.get('progress_label')} 0/0"
                    else:
                        progress_value = len(st.session_state["choices_path"]) / st.session_state["story_depth"]
                        progress_text = f"{page_text_part.get('progress_label')} {len(st.session_state['choices_path'])}/{st.session_state['story_depth']}"
                    st.progress(progress_value, text=progress_text)

            display_interactive_story(st.session_state.get("interactive_story"))

@st.fragment
def display_result_type_options(page_text_part):
    output_type = st.segmented_control(
        page_text_part.get("output_type_label"),
        page_text_part.get("output_type_options"),
        selection_mode="single", default=page_text_part.get("timeline"), width="stretch")

    if output_type == page_text_part.get("interactive_story"):
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            story_depth = st.number_input(page_text_part.get("story_depth_label"), min_value=1, max_value=5, value=3, step=1, help=page_text_part.get("story_depth_help_text"))
        with col2:
            choices_per_chapter = st.number_input(page_text_part.get("choices_per_chapter_label"), min_value=2, max_value=4, value=2, step=1, help=page_text_part.get("choices_per_chapter_help_text"))
    else:
        story_depth = None
        choices_per_chapter = None

    st.space("xxsmall")

    return output_type, story_depth, choices_per_chapter
    