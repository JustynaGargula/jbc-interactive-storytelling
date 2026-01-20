import streamlit as st

import utils

def top_part():
    st.title("Interaktywne Opowieści z danych JBC")
    st.write("Aplikacja do eksploracji danych z Jagiellońskiej Biblioteki Cyfrowej za pomocą modeli językowych Google GenAI.")
    st.space("small")

def main_interface(all_subject_names, all_centuries, dates__range, kg):
    st.header("Wybierz filtry do tematu opowieści lub osi czasu:")

    selected_subject_names = st.multiselect("Wybierz tematy:", all_subject_names, placeholder="Wybierz jeden lub więcej tematów")
    selected_centuries = st.pills("Wybierz wiek(i):", all_centuries, selection_mode="multi")

    selected_related = st.checkbox("Uwzględnij dokumenty powiązane z tematami i/lub datami")

    selected_date_range = st.slider(
        "Wybierz zakres lat (opcjonalnie):",
        min_value=dates__range[0],
        max_value=dates__range[1],
        value=dates__range
    )

    output_type = st.segmented_control(
        "Wybierz typ opowieści:",
        ["Interaktywna opowieść", "Oś czasu"],
        selection_mode="single", default="Oś czasu")

    if st.button("Generuj"):
        with st.spinner("Znajduję odpowiednie dokumenty... ⏳"):
            data = utils.get_data_based_on_selected_filters(
                selected_subject_names,
                selected_centuries,
                selected_date_range,
                selected_related,
                kg
            )

        if output_type == "Interaktywna opowieść":
            with st.spinner("Generuję opowieść... ⏳"):
                story = utils.generate_story_from_data(data)

            if story:
                st.divider()
                st.subheader("📖 Wygenerowana opowieść")
                st.markdown(story)
            else:
                st.warning("Nie znaleziono dokumentów pasujących do wybranych filtrów.")

        elif output_type == "Oś czasu":
            with st.spinner("Generuję oś czasu... ⏳"):
                timeline, df = utils.generate_timeline(data)

            if timeline:
                st.divider()
                st.subheader("🕰️ Wygenerowana oś czasu")
                st.plotly_chart(timeline, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Liczba dokumentów", len(df))
                with col2:
                    st.metric("Zakres lat", f"{int(df['year'].min())} - {int(df['year'].max())}")
                with col3:
                    st.metric("Typy dokumentów", len(df['type'].unique()))

                with st.expander("📋 Zobacz wszystkie dokumenty w tabeli"):
                    for idx, row in df.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{row['title']}**")
                            st.caption(f"{row['subjects']}")
                        with col2:
                            st.text(row['date_display'])
                        with col3:
                            if row['url']:
                                st.link_button("Otwórz", row['url'], use_container_width=True)
                        st.divider()

            else:
                st.warning("Nie znaleziono dokumentów pasujących do wybranych filtrów.")