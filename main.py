import os

import utils
import streamlit as st
# import evaluation_module

ris_directory_path = "data/data_ris"
rdfs_directory_path = "data/rdfs"
available_languages = {"polski": "pl", "english": "en"}
model = "gemini-3.1-flash-lite"  # modele: gemini-3-flash-preview, gemini-2.5-flash, gemini-3.1-flash-lite, 
allowed_centuries = [19, 20]

st.set_page_config(
    page_title="Interaktywne Opowieści JBC",
    page_icon="📚",
    layout="wide",
    # initial_sidebar_state="expanded"
)

if st.session_state.get("language") is None:
    st.session_state["language"] = "pl"

color_mode_caption_placeholder = st.empty()
col1, col2= st.columns([8, 2])

with col2:
    lan_name = st.selectbox(label="Zmień język / Change language 🌐", options=available_languages.keys(), index=0, filter_mode=None)
    st.session_state["language"] = available_languages.get(lan_name)
    utils.load_page_text_in_chosen_language(st.session_state["language"])

page_text = st.session_state["page_text"]
color_mode_caption_placeholder.caption(page_text.get("main_file").get("bottom_color_mode_caption"), text_alignment="right")

with col1:
    utils.display_interface_top_part()

with st.spinner(page_text.get("main_file").get("loading_spinner_text")):
    jsonld_output_file_pl = "data/jbc_knowledge_graph_pl.jsonld"
    if os.path.exists(jsonld_output_file_pl):
        kg = utils.import_knowledge_graph_from_jsonld_file(jsonld_output_file_pl)
    else:
        kg = utils.get_knowledge_graph_from_ris(ris_directory_path, rdfs_directory_path, allowed_centuries, jsonld_output_file_pl, already_downloaded_rdfs=True, already_saved_jsonld=False)
    all_subject_names = kg.get_all_subject_names()
    available_centuries = kg.get_all_centuries()
    dates__range = kg.get_dates_range()

utils.display_interface_main_part(all_subject_names, available_centuries, dates__range, kg, model)

# with st.sidebar:
#     evaluation_module.display_all_questions()
