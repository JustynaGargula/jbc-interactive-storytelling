import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from utils import get_or_create_session_id

keys = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "m1", "m2"]
options_scale = ["1", "2", "3", "4", "5"]

def display_all_questions():
    with st.sidebar:
        global page_text_part
        page_text_part = st.session_state["page_text"].get("evaluation_module", {})
        scale_descriptions = page_text_part.get("scale_descriptions", "")
        job_options = page_text_part.get("demographics", {}).get("job_options", [])

        st.header(page_text_part.get("header"))
        st.write(f"{page_text_part.get('description')}\n{scale_descriptions}.")

        # Wybór motywu aplikacji
        st.subheader(page_text_part.get("theme_selection", {}).get("subheader"))
        st.pills(label=page_text_part.get("theme_selection", {}).get("comfort_label"), options=options_scale, selection_mode="single", key=keys[0])
        st.pills(label=page_text_part.get("theme_selection", {}).get("light_theme_readability_label"), options=options_scale, selection_mode="single", key=keys[1])
        st.pills(label=page_text_part.get("theme_selection", {}).get("light_theme_color_label"), options=options_scale, selection_mode="single", key=keys[2])

        # Wyszukiwanie i filtrowanie treści
        st.subheader(page_text_part.get("search_and_filtering", {}).get("subheader"))
        st.pills(label=page_text_part.get("search_and_filtering", {}).get("filter_choice_label"), options=options_scale, selection_mode="single", key=keys[3])
        st.pills(label=page_text_part.get("search_and_filtering", {}).get("topics_list_readability_label"), options=options_scale, selection_mode="single", key=keys[4])
        st.pills(label=page_text_part.get("search_and_filtering", {}).get("multi_topic_addition_label"), options=options_scale, selection_mode="single", key=keys[5])
        st.pills(label=page_text_part.get("search_and_filtering", {}).get("search_intuitiveness_label"), options=options_scale, selection_mode="single", key=keys[6])

        # Wybór narracji
        st.subheader(page_text_part.get("narrative_selection", {}).get("subheader"))
        st.pills(label=page_text_part.get("narrative_selection", {}).get("narration_type_intuitiveness_label"), options=options_scale, selection_mode="single", key=keys[7])

        # Dodatkowe opcje filtrowania
        st.subheader(page_text_part.get("additional_filter_options", {}).get("subheader"))
        st.pills(label=page_text_part.get("additional_filter_options", {}).get("related_documents_option_label"), options=options_scale, selection_mode="single", key=keys[8])

        # Oś czasu
        st.subheader(page_text_part.get("timeline", {}).get("subheader"))
        st.pills(label=page_text_part.get("timeline", {}).get("timeline_understanding_label"), options=options_scale, selection_mode="single", key=keys[9])
        st.pills(label=page_text_part.get("timeline", {}).get("timeline_visual_readability_label"), options=options_scale, selection_mode="single", key=keys[10])
        st.pills(label=page_text_part.get("timeline", {}).get("timeline_document_discovery_label"), options=options_scale, selection_mode="single", key=keys[11])
        st.pills(label=page_text_part.get("timeline", {}).get("timeline_document_navigation_label"), options=options_scale, selection_mode="single", key=keys[12])

        # Narracja historyczna
        st.subheader(page_text_part.get("historical_narrative", {}).get("subheader"))
        st.pills(label=page_text_part.get("historical_narrative", {}).get("historical_story_quality_label"), options=options_scale, selection_mode="single", key=keys[13])
        st.pills(label=page_text_part.get("historical_narrative", {}).get("historical_summary_quality_label"), options=options_scale, selection_mode="single", key=keys[14])

        # Narracja interaktywna
        st.subheader(page_text_part.get("interactive_narrative", {}).get("subheader"))
        st.pills(label=page_text_part.get("interactive_narrative", {}).get("interactive_path_intuitiveness_label"), options=options_scale, selection_mode="single", key=keys[15])
        st.pills(label=page_text_part.get("interactive_narrative", {}).get("interactive_choice_impact_label"), options=options_scale, selection_mode="single", key=keys[16])

        # Ogólne wrażenia z aplikacji 
        st.subheader(page_text_part.get("overall_app_experience", {}).get("subheader"))
        st.pills(label=page_text_part.get("overall_app_experience", {}).get("overall_app_rating_label"), options=options_scale, selection_mode="single", key=keys[17])
        st.text_area(label=page_text_part.get("optional_feedback", {}).get("label"), key="q_opt", placeholder=page_text_part.get("optional_feedback", {}).get("placeholder"))

        # Metryczka
        st.subheader(page_text_part.get("demographics", {}).get("subheader"))
        st.radio(label=page_text_part.get("demographics", {}).get("radio_label"), options=job_options, key=keys[18], index=None)
        st.text_input(label=page_text_part.get("demographics", {}).get("text_input_label"), key=keys[19], value=None)
        st.caption(page_text_part.get("demographics", {}).get("caption"))
        
        st.space("xsmall")

        questions_answered_percentage_float = check_number_of_answered_questions() / len(keys)
        st.progress(questions_answered_percentage_float, f"{check_number_of_answered_questions()} / {len(keys)} {page_text_part.get('questions_answered')}")

        if st.button(page_text_part.get("buttons", {}).get("submit")):
            save_evaluation()

def check_number_of_answered_questions():
    counter = 0
    for key in keys:
        if st.session_state[key] != None:
            counter += 1
    return counter


def save_evaluation() -> None:
    """
    Collects user answers, appends timestamp and session ID,
    and saves the complete record as a new row in Google Sheets.
    """
    # Fetch answers based on predefined keys
    answers = [st.session_state.get(k) for k in keys]

    if any(a is None for a in answers):
        st.warning(page_text_part.get("messages", {}).get("warning_incomplete"))
        return

    # Handle the optional comment
    optional_answer = st.session_state.get("q_opt") or page_text_part.get("messages", {}).get("optional_answer_default", "Brak dodatkowych uwag")

    # Insert optional answer before the last two demographic questions
    answers = answers[0:-2] + [optional_answer] + answers[-2:]

    # Get session ID and current timestamp
    session_id = get_or_create_session_id()
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Authorize and connect to Google Sheets
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).sheet1

    # Construct the data row.
    # Index 0: Timestamp (Column A)
    # Index 1: Session ID (Column B)
    # Index 2+: Survey answers (Columns C onwards)
    row = [current_timestamp, session_id] + answers

    # Append to the Google Sheet
    sheet.append_row(row)

    st.success(page_text_part.get("messages", {}).get("success_thank_you"))