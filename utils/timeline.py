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

def generate_timeline(df: pd.DataFrame) -> Optional[px.scatter]:
    """
    Generuje oś czasu na podstawie podanych dokumentów.

    :param df: DataFrame zawierający dane dokumentów
    :type df: pd.DataFrame
    :return: Wygenerowana oś czasu lub None, jeśli dane są puste
    :rtype: px.scatter | None
    """
    if df.empty:
        return None

    page_text_part = st.session_state["page_text"].get("utils_generate_timeline")

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

    :param row: Wiersz DataFrame zawierający informacje o dokumencie
    :type row: pd.Series
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
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())

    col1, col2 = st.columns([0.9, 0.1], gap="medium")
    with col1.container(border=True, horizontal=True):
        event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="timeline_chart",
        )
        st.markdown("", help=page_text_part.get("help_text"))

    # zmiana zakresu dat na osi czasu
    with col2:
        with st.form(key="timeline_date_select_form", border=False):
            with st.container(border=False, horizontal_alignment="center", width="content"):
                st.markdown(f"##### {page_text_part.get('date_change_header')}", help=page_text_part.get("date_change_help_text"))
                selected_start_date = st.number_input(page_text_part.get('start_date_label'), value=min_year, min_value=min_year, max_value=max_year)
                selected_end_date = st.number_input(page_text_part.get('end_date_label'), value=max_year, min_value=min_year, max_value=max_year)
                submitted = st.form_submit_button(page_text_part.get('apply_button_label'))
                if submitted:
                    if selected_end_date<selected_start_date:
                        st.warning(page_text_part.get("wrong_date_warning"))
                    else:
                        crop_timeline_to_date_range(df, selected_start_date, selected_end_date)
                        st.rerun()

    # wyświetlenie metryk pod osią czasu
    st.space("xxsmall")
    st.write(f"**{page_text_part.get('metrics_header')}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(page_text_part.get("documents_number"), len(df))
    with col2:
        st.metric(page_text_part.get("date_range_label"), f"{min_year} - {max_year}")
    with col3:
        st.metric(page_text_part.get("documents_types_label"), len(df['type'].unique()))

    # wyświetlenie krótkiego opisu dokumentów na osi czasu
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

def crop_timeline_to_date_range(df: pd.DataFrame, start_date: int, end_date: int) -> None:
    """
    Przycina dane osi czasu do określonego zakresu dat, a konkretnie tworzy nową oś czasu według podanego zakresu dat.

    :param df: DataFrame zawierający dane dokumentów
    :type df: pd.DataFrame
    :param start_date: Data początkowa
    :type start_date: int
    :param end_date: Data końcowa
    :type end_date: int
    """
    with st.spinner(st.session_state["page_text"].get("utils_display_interface_main_part").get("generating_timeline_spinner_text")):
        new_df = df.drop(df[(df['year'] < start_date) | (df['year'] > end_date)].index)
        new_fig = generate_timeline(new_df)
        st.session_state["timeline_fig"] = new_fig
