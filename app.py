import streamlit as st

def top_part():
    st.title("Interaktywne Opowieści z danych JBC")
    st.write("Aplikacja do eksploracji danych z Jagiellońskiej Biblioteki Cyfrowej za pomocą modeli językowych Google GenAI.")


def main_interface(all_subject_names, all_centuries, dates__range):
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

    button_clicked = st.button("Generuj opowieść", on_click=st.write("*Tu będzie się generować opowieść, ale jeszcze ta funkcja nie została dodana.*"))
    # TODO: podpiąć generowanie opowieści i zastanowić się, czy chcę filtry chować wtedy

    if button_clicked:
        st.space("medium")

        st.write("Wybrane tematy:")
        for subject in selected_subject_names:
            st.write(f"- {subject}")
        st.write("Wybrane wieki:")
        for century in selected_centuries:
            st.write(f"- {century} wiek")
        st.write(f"Wybrany zakres lat: {selected_date_range[0]} - {selected_date_range[1]}")

        if selected_related:
            st.write("Dokumenty powiązane zostaną uwzględnione w opowieści.")

