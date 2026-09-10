import json

import streamlit as st
from .filtering_data import display_and_collect_subject_filters

def display_result_type_options(page_text_part: str, is_advanced_mode_on: bool):
    output_type = st.segmented_control(
        page_text_part.get("output_type_label"),
        page_text_part.get("output_type_options"),
        selection_mode="single", default=page_text_part.get("timeline"), width="stretch")

    if output_type == page_text_part.get("interactive_story") and is_advanced_mode_on:
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

def display_filters_choice(page_text_part):
    filters_choice = st.segmented_control(
        page_text_part.get("filters_choice_label"),
        options=page_text_part.get("filters_choice_options"),
        selection_mode="single", default=page_text_part.get("default_filters_choice"),
        width="stretch")
    st.space("xsmall")
    return filters_choice

def display_topics_choice(page_text_part, all_subject_names, is_advanced_mode_on):
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

    general_fit_type_names = ["or", "and"]
    if is_advanced_mode_on:
        fit_type = st.radio(page_text_part.get("fit_type_label"), page_text_part.get("fit_type_options"), horizontal=True, help=page_text_part.get("fit_type_help_text"))
        fit_type = general_fit_type_names[page_text_part.get("fit_type_options").index(fit_type)]
    else:
        fit_type = general_fit_type_names[0]

    st.space("xxsmall")

    return selected_subject_names, fit_type

def display_date_range_choice(page_text_part, dates_range):
    selected_date_range = st.slider(
        page_text_part.get("date_range_label"),
        min_value=dates_range[0],
        max_value=dates_range[1],
        value=dates_range,
        help=page_text_part.get("date_range_help_text")
    )
    st.space("xxsmall")
    return selected_date_range

def display_text_query(page_text_part):
    user_query = st.text_area(page_text_part.get("query_filter_label"), height=200, placeholder=page_text_part.get("query_filter_placeholder"))
    st.space("xxsmall")
    return user_query

def display_additional_documents_checkbox(page_text_part, is_advanced_mode_on):
    if is_advanced_mode_on:
        selected_related = st.checkbox(page_text_part.get("related_documents_label"),
            help=page_text_part.get("related_documents_help_text"))
        st.space("xxsmall")
    else:
        selected_related = False

    return selected_related