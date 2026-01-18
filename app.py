import streamlit as st

import utils

def top_part():
    st.title("Interaktywne Opowieści z danych JBC")
    st.write("Aplikacja do eksploracji danych z Jagiellońskiej Biblioteki Cyfrowej za pomocą modeli językowych Google GenAI.")


def main_interface(all_subject_names, all_centuries, dates__range, kg):
    st.write("Wybierz filtry do tematu, o którym chciałbyś usłyszeć opowieść:")

    selected_subject_names = st.multiselect("Wybierz tematy:", all_subject_names)
    selected_centuries = st.pills("Wybierz wiek(i):", all_centuries, selection_mode="multi")

    selected_related = st.checkbox("Uwzględnij dokumenty powiązane z tematami i/lub datami")

    selected_date_range = st.slider(
        "Wybierz zakres lat (opcjonalnie):",
        min_value=dates__range[0],
        max_value=dates__range[1],
        value=dates__range
    )

    if st.button("Generuj opowieść"):
        with st.spinner("Generuję opowieść... ⏳"):
            story = utils.handle_button_click(
                selected_subject_names,
                selected_centuries,
                selected_date_range,
                selected_related,
                kg
            )

        if story:
            st.divider()
            st.subheader("📖 Wygenerowana opowieść")
            st.markdown(story)
        else:
            st.warning("Nie znaleziono dokumentów pasujących do wybranych filtrów.")
