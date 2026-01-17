from collections import defaultdict
from typing import Tuple
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import re

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
            self.date_normalized = self.normalize(self.date_raw)
            if self.date_normalized:
                self.year = self.date_normalized['year']
                self.century = self.date_normalized['century']
                self.year_end = self.date_normalized.get('year_end')
                self.is_approximate = self.date_normalized.get('is_approximate', False)
                self.is_range = self.date_normalized.get('is_range', False)
                self.is_unparsed = self.date_normalized.get('is_unparsed', False)

    def normalize(self, date_str: str) -> Optional[Dict]:
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

