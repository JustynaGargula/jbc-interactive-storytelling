import re     # regular expressions
import requests
from rdflib import Graph
import glob
from typing import List, Optional
from collections import defaultdict
import json
from models import Document, KnowledgeGraph
import streamlit as st
from google import genai
from google.api_core import exceptions
import plotly.express as px
import pandas as pd
from pathlib import Path
import roman
import uuid

SEARCH_URL = "https://jbc.bj.uj.edu.pl/dlibra/results?q=&action=SimpleSearchAction&type=-6&qf1=collections%3A188&qf2=collections%3A201&qf3=Subject%3Aspo%C5%82ecze%C5%84stwo&qf4=Subject%3Adruki%20ulotne%2020%20w.&qf5=Subject%3Adruki%20ulotne%2019%20w.&ipp=50"
    # parametr, które można dodać: "&ipp=50" to liczba wyników na stronie (50 tu, domyślnie jst 25), a "&p=0" oznacza numer strony (pierwsza ma nr 0)
RDF_URL = "https://jbc.bj.uj.edu.pl/dlibra/rdf.xml?type=e&id="

def get_ids(files: List[str]) -> List[str]:
    """
    Wyciąga id dokumentów z podanych plików w standardzie RIS.

    Id jest wyciągane z wierszy z adresem url - tag "UR" np. `UR  - http://jbc.bj.uj.edu.pl/dlibra/publication/edition/510136`.

    :param files: lista ścieżek do plików zawierających dane w standardzie RIS
    :type files: List[str]

    :return ids: lista id obiektów w JBC
    """

    ids = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("UR"):
                    match = re.search("/edition/(\\d+)", line)
                    if match:
                        ids.append(match.group(1))
    print(f"Znaleziono {len(ids)} ID.")
    return ids


def get_rdfs(ids: List[str]) -> List[bytes]:
    """
    Pobiera dokumenty rdf według podanych id.

    :param ids: lista id obiektów w JBC
    :type ids: List[str]

    :return rdfs: lista pobranych dokumentów w formacie rdf
    """
    rdfs = []
    for id in ids:
        print(f"Pobieram rdf dla id={id}")
        r = requests.get(RDF_URL+str(id))
        if r.ok:
            rdfs.append(r.content)
    print(f"Pobrano {len(rdfs)} rdfów.")
    return rdfs


def save_rdfs_to_file(rdfs: List[bytes], ids: List[str], path: str = "./data/rdfs"):
    """
    Zapisuje rdfy do plików w folerze `/data/rdfs`, chyba że podano inaczej.

    :param rdfs: lista pobranych danych o obiektach w formacie rdf
    :type rdfs: List[bytes]
    :param ids: lista id obiektów w JBC odpowiadającym podanym danym RDF
    :type ids: List[str]
    :param path: ścieżka do folderu, w którym zapisywane są pliki rdf, domyślnie "./data/rdfs"
    :type path: str
    """
    rdfs_path = Path(path)
    if not rdfs_path.exists():
        rdfs_path.mkdir(parents=True, exist_ok=True)

    for i, id in enumerate(ids):
        with open(f"{path}/{id}.rdf", "wb") as f:
            f.write(rdfs[i])


def create_graph(directory_path_with_rdfs: str) -> Graph:
    """
    Tworzy graf z danymi z plików `.rdf`.

    :param directory_path_with_rdfs: ścieżka do folderu zawierającego pliki `.rdf`
    :type directory_path_with_rdfs: str

    :return graph: obiekt grafu RDFLib zawierający dane z podanych plików
    """
    graph = Graph()

    for rdf_file in glob.glob(f"{directory_path_with_rdfs}" + "/*.rdf"):
        graph.parse(rdf_file)
    print(f"Łącznie wczytano {len(graph)} trójek.")

    return graph


def save_data_to_one_file(graph: Graph, format="turtle", file_extension=".ttl"):
    """
    Zapisuje graf do jednego pliku w podanym formacie.

    :param graph: obiekt grafu RDFLib zawierający dane
    :type graph: Graph
    :param format: format zapisu (np. "turtle", "xml", "nt"), domyślnie "turtle"
    :type format: str
    :param file_extension: rozszerzenie pliku wynikowego (np. ".ttl", ".xml", ".nt"), domyślnie ".ttl"
    :type file_extension: str
    """
    graph.serialize(f"./data/merged_graph{file_extension}", format=format)


def build_kg_from_rdf(rdf_graph: Graph, allowed_centuries: List[int]) -> KnowledgeGraph:
    """
    Buduje graf wiedzy z grafu RDF.

    :param rdf_graph: graf stworzony przez bibliotekę RDFLib zawierający dane RDF
    :type rdf_graph: Graph
    :param allowed_centuries: lista dozwolonych stuleci, z których zostaną wczytane dokumenty (np. [19, 20])
    :type allowed_centuries: List[int]

    :return kg: zbudowany graf wiedzy
    :rtype: KnowledgeGraph
    """
    kg = KnowledgeGraph()

    docs_data = defaultdict(lambda: {
        'title': None,
        'identifier': None,
        'description': None,
        'date': None,
        'subjects': [],
        'creator': None,
        'publisher': None,
        'type': None,
    })

    query = """
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    SELECT ?doc ?title ?date ?description ?subject ?identifier ?creator ?publisher ?type
    WHERE {
        ?doc dc:title ?title .
        OPTIONAL { ?doc dc:date ?date }
        OPTIONAL { ?doc dc:description ?description }
        OPTIONAL { ?doc dc:subject ?subject }
        OPTIONAL { ?doc dc:identifier ?identifier }
        OPTIONAL { ?doc dc:creator ?creator }
        OPTIONAL { ?doc dc:publisher ?publisher }
        OPTIONAL { ?doc dc:type ?type }
    }
    """

    # zapisanie danych w odpowiednich zmiennych
    for row in rdf_graph.query(query):
        doc_uri = str(row.doc)

        if row.title:
            docs_data[doc_uri]['title'] = str(row.title)
        if row.date and not docs_data[doc_uri]['date']:
            docs_data[doc_uri]['date'] = str(row.date)
        if row.description:
            docs_data[doc_uri]['description'] = str(row.description)
        if row.subject:
            docs_data[doc_uri]['subjects'].append(str(row.subject))
        if row.identifier:
            docs_data[doc_uri]['identifier'] = str(row.identifier)
        if row.creator:
            docs_data[doc_uri]['creator'] = str(row.creator)
        if row.publisher:
            docs_data[doc_uri]['publisher'] = str(row.publisher)
        if row.type:
            docs_data[doc_uri]['type'] = str(row.type)

    # tworzenie obiektów Document i dodawanie ich do KnowledgeGraph
    for doc_uri, data in docs_data.items():
        if data['title'] and data['identifier']:
            doc = Document(
                identifier=data['identifier'],
                title=data['title'],
                description=data['description'] or "",
                subjects=data['subjects'],
                date_raw=data['date'] or  "",
                creator=data['creator'] or  "",
                publisher=data['publisher'] or  "",
                type=data['type'] or  "",
            )
            if doc.century in allowed_centuries:
                kg.add_document(doc)

    # budowanie relacji
    kg.build_relations()

    return kg

def export_kg_to_jsonld(kg: KnowledgeGraph):
    """
    Eksportuje graf wiedzy do formatu JSON-LD.

    :param kg: graf wiedzy
    :type kg: KnowledgeGraph
    :return graph: graf w formacie JSON-LD
    :rtype: dict
    """
    context = {
    "@context": {
        "@vocab": "http://jbc.bj.uj.edu.pl/vocab/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "title": "dc:title",
        "subject": "dc:subject",
        "date": "dc:date",
        "description": "dc:description",
        "identifier": "dc:identifier",
        "creator": "dc:creator",
        "publisher": "dc:publisher",
        "type": "dc:type",
        "hasRelation": "http://jbc.bj.uj.edu.pl/vocab/hasRelation",
        "relationType": "http://jbc.bj.uj.edu.pl/vocab/relationType",
        "relatedTo": "http://jbc.bj.uj.edu.pl/vocab/relatedTo",
        "weight": "http://jbc.bj.uj.edu.pl/vocab/weight",
        "year": "http://jbc.bj.uj.edu.pl/vocab/year",
        "century": "http://jbc.bj.uj.edu.pl/vocab/century",
        "year_end": "http://jbc.bj.uj.edu.pl/vocab/yearEnd",
        "isApproximate": "http://jbc.bj.uj.edu.pl/vocab/isApproximate",
        "isRange": "http://jbc.bj.uj.edu.pl/vocab/isRange",
        "dateDisplay": "http://jbc.bj.uj.edu.pl/vocab/dateDisplay",
    }
}

    documents = []
    for doc in kg.documents.values():
        doc_obj = {
            "@id": doc.identifier,
            "@type": "Document",
            "title": doc.title,
            "description": doc.description,
            "date": doc.date_raw,
            "dateDisplay": doc.get_date_display(),
            "year": doc.year,
            "century": doc.century,
            "subject": doc.subjects,
            "creator": doc.creator,
            "publisher": doc.publisher,
            "type": doc.type,
            "hasRelation": []
        }

        if doc.year_end:
            doc_obj["yearEnd"] = doc.year_end
        if doc.is_approximate:
            doc_obj["isApproximate"] = True
        if doc.is_range:
            doc_obj["isRange"] = True


        for rel in kg.relations:
            if rel.source_id == doc.identifier:
                doc_obj["hasRelation"].append({
                    "@type": "Relation",
                    "relationType": rel.relation_type,
                    "relatedTo": rel.target_id,
                    "weight": rel.weight,
                })
            elif rel.target_id == doc.identifier:
                doc_obj["hasRelation"].append({
                    "@type": "Relation",
                    "relationType": rel.relation_type,
                    "relatedTo": rel.source_id,
                    "weight": rel.weight,
                })
        documents.append(doc_obj)


    subjects = []
    for subj_key, subj in kg.subjects.items():
        subj_obj = {
            "@id": f"http://jbc.bj.uj.edu.pl/subject/{subj_key}",
            "@type": "Subject",
            "name": subj.name,
            "documents": subj.documents,
        }
        subjects.append(subj_obj)

    graph = {
        "@context": context["@context"],
        "@graph": documents + subjects
    }
    print(f"Wyeksportowano:")
    print(f"  - Dokumentów: {len(documents)}")
    print(f"  - Subjects: {len(subjects)}")
    print(f"  - Relacji: {len(kg.relations)}")

    return graph


def save_jsonld_to_file(jsonld_graph: dict, output_file: str):
    """
    Zapisuje graf JSON-LD do pliku.

    :param jsonld_graph: graf w formacie JSON-LD
    :type jsonld_graph: dict
    :param output_file: ścieżka do pliku wyjściowego
    :type output_file: str
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(jsonld_graph, f, ensure_ascii=False, indent=2)

    print(f"Zapisano graf do {output_file}")

@st.cache_data(show_spinner=False)
def get_knowledge_graph_from_ris(ris_files_directory_path: str,  rdfs_directory_path: str, allowed_centuries: List[int], already_downloaded_rdfs: bool = False, already_saved_jsonld: bool = False) -> KnowledgeGraph:
    """
    Tworzy graf wiedzy na podstawie pliku RIS i folderu z rdfami.

    :param ris_files_directory_path: Ścieżka do folderu z plikami RIS
    :type ris_files_directory_path: str
    :param rdfs_directory_path: Ścieżka do folderu z plikami RDF
    :type rdfs_directory_path: str
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
        save_jsonld_to_file(jsonld_graph, "data/jbc_knowledge_graph.jsonld")

    return kg

def get_documents_from_filters(knowledge_graph: KnowledgeGraph, years: list, centuries: list, subjects: list) -> List[Document]:
    """
    Zwraca dokumenty pasujące do podanych filtrów.

    :param knowledge_graph: Graf wiedzy
    :type knowledge_graph: KnowledgeGraph
    :param years: Lista lat do filtrowania
    :type years: List[int]
    :param centuries: Lista stuleci do filtrowania
    :type centuries: List[int]
    :param subjects: Lista tematów do filtrowania
    :type subjects: List[str]
    :return: Lista dokumentów pasujących do filtrów
    :rtype: List[Document]
    """
    docs_years = []

    for year in years or []:
        docs_years += knowledge_graph.get_documents_by_year(year) or []

    docs_centuries = []
    for century in centuries or []:
        docs_centuries += knowledge_graph.get_documents_by_century(century) or []

    docs_subjects = []
    for subject in subjects or []:
        docs_subjects += knowledge_graph.get_documents_by_subject(subject) or []

    documents_ids = []
    if not docs_years and not docs_centuries and not docs_subjects:
        documents_ids = []
    elif docs_years and not docs_centuries and not docs_subjects:
        documents_ids = docs_years
    elif not docs_years and docs_centuries and not docs_subjects:
        documents_ids = docs_centuries
    elif not docs_years and not docs_centuries and docs_subjects:
        documents_ids = docs_subjects
    elif docs_years and docs_centuries and not docs_subjects:
        documents_ids = list(set(docs_years) & set(docs_centuries))
    elif docs_years and not docs_centuries and docs_subjects:
        documents_ids = list(set(docs_years) & set(docs_subjects))
    elif not docs_years and docs_centuries and docs_subjects:
        documents_ids = list(set(docs_centuries) & set(docs_subjects))
    else:
        documents_ids = list(set(docs_years) & set(docs_centuries) & set(docs_subjects))

    documents = []
    for id in documents_ids:
        doc = knowledge_graph.get_document_by_id(id)
        documents.append(doc)

    return documents


def get_documents_from_filters_and_related(knowledge_graph: KnowledgeGraph, years: list, centuries: list, subjects: list, max_related: int=10) -> List[Document]:
    """
    Zwraca dokumenty pasujące do podanych filtrów oraz powiązane z nimi dokumenty.
    :param knowledge_graph: Graf wiedzy
    :type knowledge_graph: KnowledgeGraph
    :param years: Lista lat do filtrowania
    :type years: List[int]
    :param centuries: Lista stuleci do filtrowania
    :type centuries: List[int]
    :param subjects: Lista tematów do filtrowania
    :type subjects: List[str]
    :param max_related: Maksymalna liczba powiązanych dokumentów do dodania (domyślnie 10)
    :type max_related: int
    :return: Lista dokumentów pasujących do filtrów oraz powiązanych z nimi
    :rtype: List[Document]
    """
    selected_docs = get_documents_from_filters(knowledge_graph, years, centuries, subjects)

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


def get_data_based_on_selected_filters(selected_subject_names: list, selected_centuries: list, selected_date_range: tuple, selected_related: bool, kg: KnowledgeGraph) -> List[Document]:
    """
    Zwraca dokumenty pasujące do wybranych filtrów.

    :param selected_subject_names: Lista nazw wybranych tematów
    :type selected_subject_names: list
    :param selected_centuries: Lista wybranych stuleci
    :type selected_centuries: list
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
            selected_centuries,
            selected_subject_names,
        )
    else:
        data = get_documents_from_filters(
            kg,
            years,
            selected_centuries,
            selected_subject_names,
        )
    return data


def handle_llm(prompt: str, model: str) -> Optional[str]:
    """
    Obsługuje komunikację z modelem językowym Gemini i zarządza błędami.

    :param prompt: Tekst zapytania do modelu językowego
    :type prompt: str
    :param model: Model językowy do użycia (domyślnie "gemini-3-flash-preview")
    :type model: str
    :return: Odpowiedź modelu językowego lub None w przypadku błędu
    :rtype: str | None
    """
    page_text_part = st.session_state["page_text"].get("utils_handle_llm")
    try:
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        client = genai.Client()

        response = client.models.generate_content(
            model=model, contents=prompt
        )

        return response.text

    except exceptions.ResourceExhausted as e:
        # 429 - przekroczono limit requestów
        st.error(page_text_part.get("error_429"))
        st.info(page_text_part.get("info_429"))
        return None

    except exceptions.ServiceUnavailable as e:
        # 503 - serwer przeciążony
        st.error(page_text_part.get("error_503"))
        st.info(page_text_part.get("info_503"))
        return None

    except exceptions.InvalidArgument as e:
        # 400 - błędne dane wejściowe
        st.error(page_text_part.get("error_400"))
        st.code(str(e))
        return None

    except exceptions.PermissionDenied as e:
        # 403 - problem z kluczem API
        st.error(page_text_part.get("error_403"))
        return None

    except Exception as e:
        # Inne nieoczekiwane błędy
        st.error(f"{page_text_part.get('other_error')}{type(e).__name__}")
        with st.expander(page_text_part.get("other_error_details")):
            st.code(str(e))
        return None


def generate_interactive_story_from_data(data: List[Document], model: str) -> Optional[str]:
    """
    Generuje interaktywną opowieść na podstawie podanych dokumentów.

    :param data: Lista dokumentów do wygenerowania interaktywnej opowieści
    :type data: List[Document]
    :param model: Model językowy do użycia
    :type model: str
    :return: Wygenerowana interaktywna opowieść lub None, jeśli dane są puste
    :rtype: str | None
    """
    if not data:
        return None
    page_text_part = st.session_state["page_text"].get("utils_generate_interactive_story_from_data")
    prompt = f"{page_text_part.get('prompt_pt1')} {data}{page_text_part.get('prompt_pt2')}"

    response_text = handle_llm(prompt, model=model)
    return response_text


def generate_historical_story_from_data(data: List[Document], model: str) -> Optional[str]:
    """
    Generuje historyczną opowieść na podstawie podanych dokumentów.

    :param data: Lista dokumentów do wygenerowania historycznej opowieści
    :type data: List[Document]
    :param model: Model językowy do użycia
    :type model: str
    :return: Wygenerowana historyczna opowieść lub None, jeśli dane są puste
    :rtype: str | None
    """
    if not data:
        return None
    page_text_part = st.session_state["page_text"].get("utils_generate_historical_story_from_data")
    prompt = f"{page_text_part.get('prompt_pt1')} {data}{page_text_part.get('prompt_pt2')}"
    response_text = handle_llm(prompt, model=model)
    return response_text

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

    for doc in data:
        timeline_data.append({
            'title': doc.title,
            'year': doc.year,
            'date_display': doc.get_date_display(),
            'subjects': ', '.join(doc.subjects[:3]),  # pierwsze 3 tematy
            'type': doc.type,
            'url': doc.identifier,
        })

    df = pd.DataFrame(timeline_data)
    df = df[df['year'].notna()] # usuwa wiersze bez roku
    df = df.sort_values('year')
    return df

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


def display_interface_top_part():
    """
    Wyświetla górną część interfejsu użytkownika (tytuł i opis) w aplikacji Streamlit.
    """
    page_text_part = st.session_state["page_text"].get("utils_display_interface_top_part")
    st.title(page_text_part.get("main_header"))
    st.write(page_text_part.get("main_description"))
    st.space("small")


def display_interface_main_part(all_subject_names: List[str], all_centuries: List[str], dates__range: tuple, kg: KnowledgeGraph, model: str = "gemini-3-flash-preview"):
    """
    Wyświetla główną część interfejsu użytkownika w aplikacji Streamlit, umożliwiając wybór filtrów i generowanie opowieści lub osi czasu.

    :param all_subject_names: Lista wszystkich nazw tematów
    :type all_subject_names: List[str]
    :param all_centuries: Lista wszystkich wieków
    :type all_centuries: List[str]
    :param dates__range: Zakres lat (np. (1800, 1900))
    :type dates__range: tuple
    :param kg: Graf wiedzy
    :type kg: KnowledgeGraph
    """
    page_text_part = st.session_state["page_text"].get("utils_display_interface_main_part")
    st.header(page_text_part.get("filters_header"))

    selected_subject_names = st.multiselect(page_text_part.get("subjects_filter_label"), all_subject_names, placeholder=page_text_part.get("subjects_filter_placeholder"))
    st.space("xxsmall")

    all_roman_centuries = [roman.toRoman(c) for c in all_centuries]
    selected_centuries = [ roman.fromRoman(c) for c in st.pills(page_text_part.get("centuries_filter_label"), all_roman_centuries, selection_mode="multi") ]
    st.space("xxsmall")

    selected_date_range = st.slider(
        page_text_part.get("date_range_label"),
        min_value=dates__range[0],
        max_value=dates__range[1],
        value=dates__range
    )
    st.space("xxsmall")

    output_type = st.segmented_control(
        page_text_part.get("output_type_label"),
        page_text_part.get("output_type_options"),
        selection_mode="single", default=page_text_part.get("timeline"))
    st.space("xxsmall")

    selected_related = st.checkbox(page_text_part.get("related_documents_label"))
    st.space("xxsmall")

    if st.button(page_text_part.get("generate_button_label")):
        with st.spinner(page_text_part.get("getting_data_spinner_text")):
            data = get_data_based_on_selected_filters(
                selected_subject_names,
                selected_centuries,
                selected_date_range,
                selected_related,
                kg
            )
            df = convert_data_to_dataframe(data)

        if output_type == page_text_part.get("historical_story"):
            with st.spinner(page_text_part.get("generating_story_spinner_text")):
                story = generate_historical_story_from_data(data, model)

            if story:
                st.divider()
                st.subheader(page_text_part.get("generated_story_header"))
                st.markdown(story)
            else:
                st.warning(page_text_part.get("no_documents_warning"))

        elif output_type == page_text_part.get("interactive_story"):
            with st.spinner(page_text_part.get("generating_story_spinner_text")):
                story = generate_interactive_story_from_data(data, model)

            if story:
                st.divider()
                st.subheader(page_text_part.get("generated_story_header"))
                st.markdown(story)
            else:
                st.warning(page_text_part.get("no_documents_warning"))

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
                st.warning(page_text_part.get("no_documents_warning"))

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