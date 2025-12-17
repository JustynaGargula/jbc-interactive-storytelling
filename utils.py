import re       # regular expressions
import requests
from rdflib import Graph
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict
from typing import Tuple
import json

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


def normalize(date_str: str) -> Optional[Dict]:
    """
    Normalizuje podaną datę w formie tekstowej do struktury zawierającej rok, wiek itp.

    Dla zakresu dat (np. "[1990-1999]") zwracany jest domyślnie rok początkowy, a w dodatkowym polu rok końcowy.
    Dla wieku (np. "19 w.") zwracany jest środkowy rok wieku (np. 1850).

    :param date_str: data w formie tekstowej
    :type date_str: str

    :return: słownik z znormalizowaną datą, np.:
    {   'original': 'ca 1999',
        'year': 1999,
        'century': 20,
        'is_approximate': True
    }
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # Wzorzec 1: zwykły rok (np. "1999")
    match = re.match(r"^(\d{4})$", date_str)
    if match:
        year = int(match.group(1))
        return {
            'original': date_str,
            'year': year,
            'century': (year - 1) // 100 + 1,
        }

    # Wzorzec 2: daty przybliżone (np. "ca 1999")
    match = re.match(r"^(ca|circa|c\.)\s*(\d{4})$", date_str, re.IGNORECASE)
    if match:
        year = int(match.group(2))
        return {
            'original': date_str,
            'year': year,
            'century': (year - 1) // 100 + 1,
            'is_approximate': True,
        }

    # Wzorzec 3: zakresy dat (np. "[1990-1999]")
    match = re.match(r"^\[?(\d{4})\s*-\s*(\d{4})\]?$", date_str)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return {
            'original': date_str,
            'year': start_year,
            'end_year': end_year,
            'century': (start_year - 1) // 100 + 1,
            'is_range': True,
        }

    # Wzorzec 4: wiek (np. "19 w.")
    match = re.match(r"^(\d{1,2})\s*w\.$", date_str)
    if match:
        century = int(match.group(1))
        year = (century - 1) * 100 + 50 # środek wieku
        return {
            'original': date_str,
            'year': year,
            'century': century,
            'is_approximate': True,
        }

    # Wzorzec 5: data w nawiasach kwadratowych (np. "[1990]")
    match = re.match(r"^\[?(\d{4})\s*\]?$", date_str)
    if match:
        year = int(match.group(1))
        return {
            'original': date_str,
            'year': year,
            'century': (year - 1) // 100 + 1,
        }

    return {
        'original': date_str,
        'year': None,
        'century': None,
        'is_unparsed': True,
    }


@dataclass
class Document:
    """Reprezentuje dokument w grafie wiedzy."""
    identifier: str
    title: str
    description: str = ""
    subjects: List[str] = field(default_factory=list)
    date_raw: str = ""
    date_normalized: Optional[Dict] = None
    year: Optional[int] = None
    century: Optional[int] = None
    creator: Optional[str] = None
    publisher: Optional[str] = None
    type: str = ""
    year_end: Optional[int] = None  # ← dla zakresów
    century: Optional[int] = None
    decade: Optional[int] = None
    is_approximate: bool = False     # o dacie
    is_range: bool = False           # o dacie
    is_unparsed: bool = False        # o dacie

    def __post_init__(self):
        if self.date_raw:
            self.date_normalized = normalize(self.date_raw)
            if self.date_normalized:
                self.year = self.date_normalized['year']
                self.century = self.date_normalized['century']
                self.year_end = self.date_normalized.get('year_end')
                self.is_approximate = self.date_normalized.get('is_approximate', False)
                self.is_range = self.date_normalized.get('is_range', False)
                self.is_unparsed = self.date_normalized.get('is_unparsed', False)

    def get_date_display(self) -> str:
        """Zwraca czytelny opis daty dla LLM"""
        if self.is_unparsed:
            return f"date uncertain: {self.date_raw}"
        if self.is_range and self.year_end:
            return f"{self.year}-{self.year_end}"
        if self.is_approximate:
            return f"circa {self.year}"
        if self.year:
            return str(self.year)
        return "date unknown"


@dataclass
class Subject:
    """Reprezentuje subject w grafie wiedzy."""
    name: str
    documents: List[str] = field(default_factory=list)

    def add_document(self, doc_id: str):
        """Dodaje dokument do listy dokumentów powiązanych z tym subjectem.
        :param doc_id: ID dokumentu do dodania
        :type doc_id: str
        """
        if doc_id not in self.documents:
            self.documents.append(doc_id)


class Relation:
    """Reprezentuje relację między dwoma dokumentami w grafie wiedzy."""
    def __init__(self, source_id: str, target_id: str, relation_type: str, weight: float):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.weight = weight


class KnowledgeGraph:
    """Reprezentuje graf wiedzy z dokumentami, subjectami i relacjami."""
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.subjects: Dict[str, Subject] = {}
        self.relations: List[Relation] = []

        self.by_year: Dict[int, List[str]] = defaultdict(list)
        self.by_century: Dict[int, List[str]] = defaultdict(list)
        self.by_subject: Dict[str, List[str]] = defaultdict(list)


    def add_document(self, doc: Document):
        """
        Dodaje dokument do grafu wiedzy i aktualizuje indeksy pomocnicze.

        :param doc: dokument do dodania
        :type doc: Document
        """
        self.documents[doc.identifier] = doc

        if doc.year is not None:
            self.by_year[doc.year].append(doc.identifier)
        if doc.century is not None:
            self.by_century[doc.century].append(doc.identifier)

        for subject_name in doc.subjects:
            subject_key = subject_name.lower()
            self.by_subject[subject_key].append(doc.identifier)

            if subject_key not in self.subjects:
                self.subjects[subject_key] = Subject(name=subject_name)
            self.subjects[subject_key].add_document(doc.identifier)


    def build_relations(self):
        """
        Buduje relacje między dokumentami na podstawie wspólnych cech.

        Relacje są tworzone na podstawie:
        1. wspólnych subjects (relacja "shared_subject", waga 1.0)
        2. chronologicznej bliskości (relacja "close_next", waga 1.0)
        3. tego samego wieku (relacja "same_century", waga 0.5)
        """
        # 1. wspólne subjects
        for subject_key, doc_ids in self.by_subject.items():
            if len(doc_ids) > 1:
                for i, doc_id1 in enumerate(doc_ids):
                    for doc_id2 in doc_ids[i+1:]:
                        if doc_id1 != doc_id2:
                            self.relations.append(Relation(doc_id1, doc_id2, "shared_subject", 1.0))


        # 2. chronologicznie
        sorted_years = sorted(self.by_year.keys())
        for i in range(len(sorted_years) - 1):
            year1 = sorted_years[i]
            year2 = sorted_years[i + 1]
            doc_ids1 = self.by_year[year1]
            doc_ids2 = self.by_year[year2]
            if year2 - year1 <= 5:      # odległość do 5 lat
                for doc_id1 in doc_ids1:
                    for doc_id2 in doc_ids2:
                        self.relations.append(Relation(doc_id1, doc_id2, "close_next", 1.0))


        # 3. te same century
        for century, doc_ids in self.by_century.items():
            if len(doc_ids) > 1:
                for i, doc_id1 in enumerate(doc_ids):
                    for doc_id2 in doc_ids[i+1:]:
                        if doc_id1 != doc_id2:
                            self.relations.append(Relation(doc_id1, doc_id2, "same_century", 0.5))


    def get_related_documents(self, doc_id: str, max_results: int = 5) -> List[Tuple[str, float]]:
        """
        Zwraca listę powiązanych dokumentów z podanym dokumentem na podstawie relacji w grafie wiedzy.

        :param doc_id: ID dokumentu
        :type doc_id: str
        :param max_results: maksymalna liczba wyników do zwrócenia
        :type max_results: int

        :return: lista krotek (dokument, score (suma) powiązania)
        :rtype: List[Tuple[str, float]]
        """
        related = defaultdict(float)

        for relation in self.relations:
            if relation.source_id == doc_id:
                related[relation.target_id] += relation.weight
            elif relation.target_id == doc_id:
                related[relation.source_id] += relation.weight

        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        result = []
        for doc_id, score in sorted_related[:max_results]:
            result.append((self.documents[doc_id], score))

        return result


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