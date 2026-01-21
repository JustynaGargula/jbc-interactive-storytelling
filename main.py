import utils
import streamlit as st

ris_file = ".\data\data_ris\dlibra.ris"
part = 1
rdfs_directory_path = f"./data/part{part}/*rdf"

st.set_page_config(
    page_title="Interaktywne Opowieści JBC",
    page_icon="📚",
    layout="wide",
)

utils.get_interface_top_part()

with st.spinner("Ładowanie danych do działania aplikacji... ⏳"):
    kg = utils.get_knowledge_graph_from_ris(ris_file, rdfs_directory_path, part, True, True)
    all_subject_names = kg.get_all_subject_names()
    available_centuries = kg.get_all_centuries()
    dates__range = kg.get_dates_range()

utils.get_interface_main_part(all_subject_names, available_centuries, dates__range, kg)

st.space("large")
st.caption("💡 **Zmiana motywu na ciemny lub jasny:** Kliknij 3 kropki - ⋮ - w prawym górnym rogu → Settings → Theme")