# JBC Interactive Storytelling

## Opis projektu

Projekt integruje metadane dokumentów życia społecznego JBC (Jagiellońska Biblioteka Cyfrowa) w graf wiedzy, tworzy powiązania semantyczne i wykorzystuje LLM do generowania narracji tematycznych lub chronologicznych. Interfejs umożliwia eksplorację kolekcji w formie osi czasu lub ciągłych opowieści, wspierając kontekstowe odkrywanie treści w badaniach historycznych i edukacyjnych.

## Użyte technologie

- Python
- Streamlit
- Gemini API (google-genai)

## Uruchomienie aplikacji

1. Zainstaluj wymagane biblioteki.
   - Upewnij się, że masz odpowiednią wersję Pythona: 3.9 lub nowszy `python --version`
   - Stwórz wirtualne środowisko: `python -m venv venv`
   - Aktywuj to wirtualne środowisko: `source venv/scripts/activate`
   - Zainstaluj wymagane biblioteki: `pip install -r requirements.txt`

2. Pobierz własny klucz do Gemini API.
   - Utwórz klucz tutaj: [link](https://aistudio.google.com/app/apikey)
   - Zapisz ten klucz jako zmienną środowiskową: [instrukcja](https://ai.google.dev/gemini-api/docs/api-key?hl=pl#set-api-env-var)

3. Uruchom ponownie terminal i/lub IDE. Upewnij się, że masz aktywne wirtualne środowisko (`source venv/scripts/activate`).

4. Uruchom aplikację komendą `streamlit run main.py`. Otworzy się ona w przeglądarce pod adresem `http://localhost:8501/`.

5. Aby pobrać więcej danych (albo dane z innej kategorii) z Jagiellońskiej Biblioteki Cyfrowej:
   - Wejdź na stronę: [jbc.bj.uj.edu.pl](https://jbc.bj.uj.edu.pl/dlibra/results).
   - Wybierz odpowiednie filtry.
   - Na koniec linka strony wklej `&ipp=50`, gdzie zamiast `50` wpisz liczbę dokumentów, które chcesz pobrać.
   - Po załadowaniu wyników na stronie kliknij w prawym górnym rogu (pod paskiem wyszukiwania) `Dodaj wszystkie obiekty z listy do bibliografii`.
   - Pobrany plik umieść w folderze `/data` jako `dlibra.ris`.
   - Usuń folder `/data/rdfs`.
   - Uruchom aplikację komendą `streamlit run main.py`. Dla dużej liczby dokumentów pobranie i przetworzenie ich może długo zająć.

## Struktura projektu

- `main.py` - główny plik aplikacji Streamlit.
- `utils.py` - plik z funkcjami pomocniczymi do przetwarzania danych, generowania grafu wiedzy i interfejsu użytkownika.
- `models.py` - plik z klasami reprezentującymi strukturę dokumentów oraz grafu wiedzy i operacje na nim.
- `requirements.txt` - plik z listą wymaganych bibliotek Python.
- `data/` - folder zawierający pliki danych (np. plik RIS z metadanymi JBC).
- `.streamlit/config.toml` - plik dostosowujący styl aplikacji Streamlit.
- `jupyter_notebooks/` - folder z notatnikami Jupyter używanymi do eksperymentów i analizy danych przed implementacją aplikacji.
