from typing import List, Optional

import pandas as pd
from models import Document
from utils.llm import handle_llm
from .general import convert_data_to_dataframe
import streamlit as st
import plotly.express as px

# if st.session_state.get("page_text"):
#     page_text_part = st.session_state["page_text"].get("utils_generate_timeline")
# else:
#     page_text_part = {"modal_header": None}

def generate_timeline(data: List[Document]) -> Optional[px.scatter]:
    """
    Generuje oś czasu na podstawie podanych dokumentów.

    :param data: Lista dokumentów do wygenerowania osi czasu
    :type data: List[Document]
    :return: Wygenerowana oś czasu lub None, jeśli dane są puste
    :rtype: px.scatter | None
    """
    if not data:
        return None

    page_text_part = st.session_state["page_text"].get("utils_generate_timeline")
    df = convert_data_to_dataframe(data)

    # Grupowanie dokumentów według typu i roku, aby ustalić pozycję na osi Y, żeby uniknąć nakładania się punktów
    STACK_STEP = 0.3    # odstęp Y między punktami w słupku (1.0 = pełny poziom)
    TYPE_GAP = 0.8      # przerwa między pasmami typów
    PX_PER_UNIT = 30    # ile pikseli odpowiada jednej jednostce osi Y

    df['stack_pos'] = df.groupby(['type', 'year']).cumcount()
    max_stack = df.groupby('type')['stack_pos'].max()
    type_base = {}
    current = 0.0
    for doc_type in df['type'].unique():
        type_base[doc_type] = current
        current += int(max_stack[doc_type]) * STACK_STEP + TYPE_GAP

    df['height'] = df['type'].map(type_base) + df['stack_pos'] * STACK_STEP
    total_levels = current

    fig = px.scatter(
        df,
        x='year',
        y='height',
        color='type',  # kolor według typu dokumentu
        hover_name='title',
        hover_data={
            'year': True,
            'subjects': True,
            'type': True,
            'height': False,
        },
        custom_data=['url'],
        # title=f'{page_text_part.get("title_pt1")} {len(df)} {page_text_part.get("title_pt2")}',
        labels={ 'type': page_text_part.get("type"), 'year': page_text_part.get("year"), 'date_display': page_text_part.get("date_display"), 'subjects': page_text_part.get("subjects") },
        size_max=15,
    )

    # Dostosuj wygląd
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))

    fig.update_layout(
        height=min(700, max(300, 150 + int(total_levels * PX_PER_UNIT))),
        showlegend=True,
        yaxis={'visible': False, 'showticklabels': False, 'range': [-0.5, total_levels]},
        xaxis={'title': page_text_part.get("year"), 'showgrid': True},
        hovermode='closest',
        clickmode='event+select',
    )
    return fig

def show_document_dialog(row: pd.Series) -> None:
    """
    Wyświetla okienko modalne z informacjami o dokumencie.

    :param doc: Dokument do wyświetlenia
    :type doc: Document
    """
    page_text_part = st.session_state["page_text"].get("utils_generate_timeline")
    st.write(f"**{page_text_part.get('title')}:** {row['title']}")
    st.write(f"**{page_text_part.get('type')}:** {row['type']}")
    st.write(f"**{page_text_part.get('date_display')}:** {row['date_display']}")
    st.write(f"**{page_text_part.get('subjects')}:**")
    subjects = row['subjects'].split(", ")
    for subject in subjects:
        st.write(f"* {subject}")
    if row['url']:
        st.link_button(page_text_part.get("open_document"), row['url'])

def display_timeline(fig: px.scatter, description: str, df: pd.DataFrame, page_text_part: dict) -> None:
    """
    Wyświetla oś czasu w aplikacji Streamlit.

    :param fig: Oś czasu do wyświetlenia
    :type fig: px.scatter
    :param description: Opis osi czasu
    :type description: str
    :param df: DataFrame zawierający dane do wyświetlenia
    :type df: pd.DataFrame
    :param page_text_part: Słownik z tekstami do wyświetlenia
    :type page_text_part: dict
    """
    page_text_part = st.session_state["page_text"].get("utils_generate_timeline")

    with st.container(border=True, horizontal=True):
        event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="timeline_chart",
        )
        st.markdown("", help=page_text_part.get("help_text"))

    # wyświetlanie metryk pod osią czasu
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(page_text_part.get("documents_number"), len(df))
    with col2:
        st.metric(page_text_part.get("date_range_label"), f"{int(df['year'].min())} - {int(df['year'].max())}")
    with col3:
        st.metric(page_text_part.get("documents_types_label"), len(df['type'].unique()))

    # wyświetlenie krótkiego opisu dokumentów na osi czasu
    st.space("small")
    st.markdown(f"**{page_text_part.get('description_label')}**")
    st.write(description)

    # obsługa kliknięcia w punkt na osi czasu
    points = event.selection.points if event and event.selection else []
    if not points:
        st.session_state.pop("last_opened_doc", None)
        return

    identifier = points[0]["customdata"][0]
    matches = df[df['url'] == identifier]

    if not matches.empty and st.session_state.get("last_opened_doc") != identifier:
        st.session_state["last_opened_doc"] = identifier
        title = st.session_state.get("page_text").get("utils_generate_timeline").get("modal_header")
        dialog = st.dialog(title, width="large")(show_document_dialog)
        dialog(matches.iloc[0])

def generate_timeline_description(df: pd.DataFrame) -> str:
    """
    Generuje opis osi czasu na podstawie znalezionych dokumentów.

    :param df: DataFrame zawierający dane dokumentów
    :type df: pd.DataFrame
    :return: Opis osi czasu
    :rtype: str
    """
    page_text_part = st.session_state["page_text"].get("utils_generate_timeline")
    prompt = f"{page_text_part.get('prompt_pt1')}{df.to_string(index=False)}\n{page_text_part.get('prompt_pt2')}\n\n"
    description = handle_llm(prompt)
    return description