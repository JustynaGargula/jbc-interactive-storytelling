import re     # regular expressions
import requests
from rdflib import Graph
import glob
from typing import List, Optional, Dict
from collections import defaultdict
import json
from models import Document, Subject, Relation, KnowledgeGraph
import streamlit as st
from google import genai
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

SEARCH_URL = "https://jbc.bj.uj.edu.pl/dlibra/results?q=&action=SimpleSearchAction&type=-6&qf1=collections%3A188&qf2=collections%3A201&qf3=Subject%3Aspo%C5%82ecze%C5%84stwo&qf4=Subject%3Adruki%20ulotne%2020%20w.&qf5=Subject%3Adruki%20ulotne%2019%20w.&ipp=50"
    # parametr, które można dodać: "&ipp=50" to liczba wyników na stronie (50 tu, domyślnie jst 25), a "&p=0" oznacza numer strony (pierwsza ma nr 0)
RDF_URL = "https://jbc.bj.uj.edu.pl/dlibra/rdf.xml?type=e&id="

def get_ids(file: str) -> List[str]:
    """
    Wyciąga id dokumentów z podanego pliku w standardzie RIS.

    Id jest wyciągane z wierszy z adresem url - tag "UR" np. `UR  - http://jbc.bj.uj.edu.pl/dlibra/publication/edition/510136`.

    :param file: ścieżka do pliku zawierającego dane w standardzie RIS
    :type file: str

    :return ids: lista id obiektów w JBC
    """

    ids = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("UR"):
                match = re.search("/edition/(\d+)", line)
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


def save_rdfs_to_file(rdfs: List[bytes], ids: List[str], part: int):
    """
    Zapisuje rdfy do plików w folerze `/data/partX`, gdzie X jest numerem określonym parametrem *part*.

    :param rdfs: lista pobranych danych o obiektach w formacie rdf
    :type rdfs: List[bytes]
    :param ids: lista id obiektów w JBC odpowiadającym podanym danym RDF
    :type ids: List[str]
    :param part: numer części (folderu), do którego zapisywane są pliki (partia danych)
    :type part: int
    """
    for i, id in enumerate(ids):
        with open(f"./data/part{part}/{id}.rdf", "wb") as f:
            f.write(rdfs[i])


def create_graph(directory_path_with_rdfs: str) -> Graph:
    """
    Tworzy graf z danymi z plików `.rdf`.

    :param directory_path_with_rdfs: ścieżka do folderu zawierającego pliki `.rdf`
    :type directory_path_with_rdfs: str

    :return graph: obiekt grafu RDFLib zawierający dane z podanych plików
    """
    graph = Graph()

    for rdf_file in glob.glob(directory_path_with_rdfs):
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


def build_kg_from_rdf(rdf_graph: Graph) -> KnowledgeGraph:
    """
    Buduje graf wiedzy z grafu RDF.

    :param rdf_graph: graf stworzony przez bibliotekę RDFLib zawierający dane RDF
    :type rdf_graph: Graph

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

def get_documents_from_filters(knowledge_graph, years, centuries, subjects):
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


def get_documents_from_filters_and_related(knowledge_graph, years, centuries, subjects, max_related=5):
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

@st.cache_data
def get_knowledge_graph_from_ris(ris_file: str,  rdfs_directory_path: str, part: int = 1, already_downloaded_rdfs: bool = False, already_saved_jsonld: bool = False) -> KnowledgeGraph:

    ids = get_ids(ris_file)

    if not already_downloaded_rdfs:
        rdfs = get_rdfs(ids)
        save_rdfs_to_file(rdfs, ids, part)

    g = create_graph(rdfs_directory_path)
    # utils.save_data_to_one_file(g, "turtle", ".ttl")

    kg = build_kg_from_rdf(g)
    print(f"Wczytano {len(kg.documents)} dokumentów do grafu wiedzy.")

    if not already_saved_jsonld:
        jsonld_graph = export_kg_to_jsonld(kg)
        save_jsonld_to_file(jsonld_graph, "data/jbc_knowledge_graph.jsonld")

    return kg

def generate_story_from_data(data):
    prompt = f"Kontekst: Skorzystaj przede wszystkim z tych danych: {data}. Zadanie: wygenereuj interaktywną opowieść na ich podstawie."

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    )
    return response.text

def get_data_based_on_selected_filters(selected_subject_names, selected_centuries, selected_date_range, selected_related, kg):
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

def generate_timeline(data):

    # przygotowanie danych do wykresu
    timeline_data = []
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
        title=f'Oś czasu {len(df)} dokumentów',
        labels={'year': 'Rok'},
        size_max=15
    )

    # Dostosuj wygląd
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))

    fig.update_layout(
        height=400,
        showlegend=True,
        yaxis={'visible': False, 'showticklabels': False},  # ukryj oś Y
        xaxis={'title': 'Rok', 'showgrid': True},
        hovermode='closest'
    )

    return fig, df
