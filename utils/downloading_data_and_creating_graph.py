
from collections import defaultdict
import glob
import json
from pathlib import Path
import re
from typing import List
from rdflib import Graph
import requests
from models import Document, KnowledgeGraph


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
        "sourceId": "http://jbc.bj.uj.edu.pl/vocab/sourceId",
        "targetId": "http://jbc.bj.uj.edu.pl/vocab/targetId",
        "relationType": "http://jbc.bj.uj.edu.pl/vocab/relationType",
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
            "subject": doc.subjects,
            "creator": doc.creator,
            "publisher": doc.publisher,
            "type": doc.type,
        }

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

    # Store relations as separate objects with explicit source and target IDs
    relations = []
    for i, rel in enumerate(kg.relations):
        rel_obj = {
            "@id": f"http://jbc.bj.uj.edu.pl/relation/{i}",
            "@type": "Relation",
            "sourceId": rel.source_id,
            "targetId": rel.target_id,
            "relationType": rel.relation_type,
            "weight": rel.weight,
        }
        relations.append(rel_obj)

    graph = {
        "@context": context["@context"],
        "@graph": documents + subjects + relations
    }
    print(f"Wyeksportowano:")
    print(f"  - Dokumentów: {len(documents)}")
    print(f"  - Subjects: {len(subjects)}")
    print(f"  - Relacji: {len(relations)}")

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
