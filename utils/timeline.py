from typing import List, Optional
from models import Document
from .general import convert_data_to_dataframe
import streamlit as st
import plotly.express as px

def generate_timeline(data: List[Document]) -> Optional[str]:
    """
    Generuje oś czasu na podstawie podanych dokumentów.

    :param data: Lista dokumentów do wygenerowania osi czasu
    :type data: List[Document]
    :return: Wygenerowana oś czasu lub None, jeśli dane są puste
    :rtype: str | None
    """
    if not data:
        return None

    page_text_part = st.session_state["page_text"].get("utils_generate_timeline")
    df = convert_data_to_dataframe(data)
    type_heights = {doc_type: i for i, doc_type in enumerate(df['type'].unique())}
    df['height'] = df['type'].map(type_heights)


    fig = px.scatter(
        df,
        x='year',
        y='height',
        color='type',  # kolor według typu dokumentu
        hover_name='title',
        hover_data={
            'year': True,
            'date_display': True,
            'subjects': True,
            'type': True,
            'height': False,
        },
        title=f'{page_text_part.get("title_pt1")} {len(df)} {page_text_part.get("title_pt2")}',
        labels={ 'type': page_text_part.get("type"), 'year': page_text_part.get("year"), 'date_display': page_text_part.get("date_display"), 'subjects': page_text_part.get("subjects") },
        size_max=15
    )

    # Dostosuj wygląd
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))

    fig.update_layout(
        height=400,
        showlegend=True,
        yaxis={'visible': False, 'showticklabels': False},  # ukryj oś Y
        xaxis={'title': page_text_part.get("year"), 'showgrid': True},
        hovermode='closest'
    )
    return fig
