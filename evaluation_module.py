import streamlit as st
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

keys = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18"]

options_scale = ["1", "2", "3", "4", "5"]
scale_descriptions = '''
* 1 - bardzo źle,
* 2 - źle,
* 3 - neutralnie,
* 4 - dobrze,
* 5 - bardzo dobrze'''

def display_all_questions():
    st.markdown(f"Poniższe pytania dotyczą różnych apektów splikacji. Odpowiedz na każde wybierając numer od 1 do 5, gdzie {scale_descriptions}.")

    # Ogólne wrażenia z aplikacji
    st.pills(label="Komfort wyboru motywu aplikacji (jasny/ciemny)", options=options_scale, selection_mode="single", key=keys[0])
    st.pills(label="Czytelność motywu jasnego", options=options_scale, selection_mode="single", key=keys[1])
    st.pills(label="Kolorystyka motywu jasnego", options=options_scale, selection_mode="single", key=keys[2])

    # Wyszukiwanie i filtrowanie treści
    st.pills(label="Łatwość wyboru filtrów do opowieści", options=options_scale, selection_mode="single", key=keys[3])
    st.pills(label="Czytelność listy tematów", options=options_scale, selection_mode="single", key=keys[4])
    st.pills(label="Łatwość dodawania wielu tematów jednocześnie", options=options_scale, selection_mode="single", key=keys[5])
    st.pills(label="Jak oceniasz intuicyjność obecnego systemu wyszukiwania?", options=options_scale, selection_mode="single", key=keys[6])

    # Wybór narracji
    st.pills(label="Intuicyjność wyboru typu narracji (historyczna / interaktywna / oś czasu)", options=options_scale, selection_mode="single", key=keys[7])

    # Dodatkowe opcje filtrowania
    st.pills(label="Czy opcja „Uwzględnij dokumenty powiązane z tematami i/lub datami” jest dla Ciebie zrozumiała? ", options=options_scale, selection_mode="single", key=keys[8])

    # Oś czasu
    st.pills(label="Oceń na ile funkcje osi czasu są dla Ciebie zrozumiałe?", options=options_scale, selection_mode="single", key=keys[9])
    st.pills(label="Czy oś czasu jest czytelna wizualnie?", options=options_scale, selection_mode="single", key=keys[10])
    st.pills(label="Na ile łatwo było Ci odnaleźć interesujące dokumenty na osi czasu?", options=options_scale, selection_mode="single", key=keys[11])
    st.pills(label="Czy łatwo jest zlokalizować konkretne dokumenty i przejść do ich źródła w JBC (skany PDF)?", options=options_scale, selection_mode="single", key=keys[12])

    # Narracja historyczna
    st.pills(label="Jak oceniasz wygenerowane opowieści historyczne?", options=options_scale, selection_mode="single", key=keys[13])
    st.pills(label="Jak oceniasz jakość i wiarygodność podsumowań historycznych?", options=options_scale, selection_mode="single", key=keys[14])

    # Narracja interaktywna
    st.pills(label="Czy sposób wyboru ścieżki narracji interaktywnej jest intuicyjny?", options=options_scale, selection_mode="single", key=keys[15])
    st.pills(label="Wskaż na ile jasne jest dla Ciebie, że wybór opcji w narracji interaktywnej wpływa na zakończenie historii?", options=options_scale, selection_mode="single", key=keys[16])

    # Ostateczna ocena
    st.pills(label="Jak ogólnie oceniasz aplikację?", options=options_scale, selection_mode="single", key=keys[17])

    st.text_area(label="Czy chcesz podzielić się dodatkowymi uwagami lub sugestiami dotyczącymi aplikacji? Tutaj możesz je wpisać:", key="q_opt", placeholder="Twoje uwagi...")

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

def save_evaluation():
    answers = [st.session_state.get(k) for k in keys]

    if any(a is None for a in answers):
        st.warning("Proszę odpowiedzieć na wszystkie pytania.")
        return

    optional_answer = st.session_state.get("q_opt") or "Brak dodatkowych uwag"
    answers.append(optional_answer)

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["connections"]["gsheets"]["spreadsheet"]).sheet1

    # Wiersz danych: timestamp + odpowiedzi
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + answers
    sheet.append_row(row)

    st.success("Dziękujemy za wyrażenie swoich opini!")
