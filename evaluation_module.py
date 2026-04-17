import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import uuid

keys = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "m1", "m2"]

options_scale = ["1", "2", "3", "4", "5"]
scale_descriptions = '''
* 1 - bardzo źle,
* 2 - źle,
* 3 - neutralnie,
* 4 - dobrze,
* 5 - bardzo dobrze'''

job_options = [
    "badaczem / badaczką z UJ",
    "badaczem / badaczką spoza UJ",
    "pracownikiem / pracowniczką instytucji GLAM (biblioteka, muzeum, archiwum) z UJ",
    "pracownikiem / pracowniczką instytucji GLAM (biblioteka, muzeum, archiwum) spoza UJ",
    "studentem / studentką nauk humanistycznych z UJ",
    "studentem / studentką nauk humanistycznych spoza UJ",
    "studentem / studentką nauk społecznych z UJ",
    "studentem / studentką nauk społecznych spoza UJ",
    "studentem / studentką nauk ścisłych z UJ",
    "studentem / studentką nauk ścisłych spoza UJ",
    "inne"
]

def display_all_questions():
    st.header("Moduł oceny")
    st.markdown(f"Poniższe pytania dotyczą różnych aspektów aplikacji. Odpowiedz proszę na każde pytanie wybierając liczbę od 1 do 5, gdzie {scale_descriptions}.")

    # Wybór motywu aplikacji
    st.subheader("Wybór motywu aplikacji")
    st.pills(label="Komfort wyboru motywu aplikacji (jasny/ciemny)", options=options_scale, selection_mode="single", key=keys[0])
    st.pills(label="Czytelność motywu jasnego", options=options_scale, selection_mode="single", key=keys[1])
    st.pills(label="Kolorystyka motywu jasnego", options=options_scale, selection_mode="single", key=keys[2])

    # Wyszukiwanie i filtrowanie treści
    st.subheader("Wyszukiwanie i filtrowanie treści")
    st.pills(label="Łatwość wyboru filtrów do tematu opowieści", options=options_scale, selection_mode="single", key=keys[3])
    st.pills(label="Czytelność listy tematów", options=options_scale, selection_mode="single", key=keys[4])
    st.pills(label="Łatwość dodawania wielu tematów jednocześnie", options=options_scale, selection_mode="single", key=keys[5])
    st.pills(label="Jak oceniasz intuicyjność obecnego systemu wyszukiwania?", options=options_scale, selection_mode="single", key=keys[6])

    # Wybór narracji
    st.subheader("Wybór narracji")
    st.pills(label="Intuicyjność wyboru typu narracji (historyczna / interaktywna / oś czasu)", options=options_scale, selection_mode="single", key=keys[7])

    # Dodatkowe opcje filtrowania
    st.subheader("Dodatkowe opcje filtrowania")
    st.pills(label="Czy opcja „Uwzględnij dokumenty powiązane z tematami i/lub datami” jest dla Ciebie zrozumiała? ", options=options_scale, selection_mode="single", key=keys[8])

    # Oś czasu
    st.subheader("Oś czasu")
    st.pills(label="Oceń na ile funkcje osi czasu są dla Ciebie zrozumiałe?", options=options_scale, selection_mode="single", key=keys[9])
    st.pills(label="Jak oceniasz czytelność graficzną osi czasu?", options=options_scale, selection_mode="single", key=keys[10])
    st.pills(label="Jak oceniasz łatwość odnajdywania interesujących dokumentów na osi czasu?", options=options_scale, selection_mode="single", key=keys[11])
    st.pills(label="Jak oceniasz łatwość lokalizowania konkretnych dokumentów i przechodzenia do ich źródła w JBC (skany PDF)?", options=options_scale, selection_mode="single", key=keys[12])

    # Narracja historyczna
    st.subheader("Narracja historyczna")
    st.pills(label="Jak oceniasz wygenerowane opowieści historyczne?", options=options_scale, selection_mode="single", key=keys[13])
    st.pills(label="Jak oceniasz jakość i wiarygodność podsumowań historycznych?", options=options_scale, selection_mode="single", key=keys[14])

    # Narracja interaktywna
    st.subheader("Narracja interaktywna")
    st.pills(label="Jak oceniasz intuicyjność wyboru ścieżki narracji interaktywnej?", options=options_scale, selection_mode="single", key=keys[15])
    st.pills(label="Jak oceniasz zrozumiałość wpływu wyborów na zakończenie historii w narracji interaktywnej?", options=options_scale, selection_mode="single", key=keys[16])

    # Ogólne wrażenia z aplikacji 
    st.subheader("Ogólne wrażenia z aplikacji")
    st.pills(label="Jak ogólnie oceniasz aplikację?", options=options_scale, selection_mode="single", key=keys[17])
    st.text_area(label="Czy chcesz podzielić się dodatkowymi uwagami lub sugestiami dotyczącymi aplikacji? Jeśli tak, wpisz je poniżej:", key="q_opt", placeholder="Twoje uwagi...")

    # Metryczka
    st.subheader("Metryczka")
    st.radio(label="Które z poniższych określeń najlepiej opisuje Twoją obecną sytuację zawodową lub edukacyjną? Jestem:", options=job_options, key=keys[18], index=None)
    st.text_input(label="Wpisz proszę kierunek, na którym studiujesz / dyscyplinę badań / doprecyzuj odpowiedź (w zależności od wybranej wcześniej opcji)", key=keys[19], value=None)
    st.caption("Naciśnij Enter, aby zatwierdzić odpowiedź")
    
    st.space("xsmall")

    questions_answered_percentage_float = check_number_of_answered_questions() / len(keys)
    st.progress(questions_answered_percentage_float, f"{check_number_of_answered_questions()} / {len(keys)} pytań ocenionych")

    if st.button("Wyślij ocenę"):
        save_evaluation()

def check_number_of_answered_questions():
    counter = 0
    for key in keys:
        if st.session_state[key] != None:
            counter += 1
    return counter


def save_evaluation() -> None:
    """
    Collects user answers, appends timestamp and session ID,
    and saves the complete record as a new row in Google Sheets.
    """
    # Fetch answers based on predefined keys
    answers = [st.session_state.get(k) for k in keys]

    if any(a is None for a in answers):
        st.warning("Proszę odpowiedzieć na wszystkie pytania.")
        return

    # Handle the optional comment
    optional_answer = st.session_state.get("q_opt") or "Brak dodatkowych uwag"

    # Insert optional answer before the last two demographic questions
    answers = answers[0:-2] + [optional_answer] + answers[-2:]

    # Get session ID and current timestamp
    session_id = get_or_create_session_id()
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Authorize and connect to Google Sheets
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).sheet1

    # Construct the data row.
    # Index 0: Timestamp (Column A)
    # Index 1: Session ID (Column B)
    # Index 2+: Survey answers (Columns C onwards)
    row = [current_timestamp, session_id] + answers

    # Append to the Google Sheet
    sheet.append_row(row)

    st.success("Dziękujemy za wyrażenie swoich opinii!")


def get_or_create_session_id() -> str:
    """
    Retrieves the existing session ID from the Streamlit session state.
    If it does not exist, generates a new UUID4 and stores it.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    return st.session_state.session_id
