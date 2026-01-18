import utils, app

ris_file = ".\data\data_ris\dlibra.ris"
part = 1
rdfs_directory_path = f"./data/part{part}/*rdf"

app.top_part()

kg = utils.get_knowledge_graph_from_ris(ris_file, rdfs_directory_path, part, True, True)
all_subject_names = kg.get_all_subject_names()
available_centuries = kg.get_all_centuries()
dates__range = kg.get_dates_range()

app.main_interface(all_subject_names, available_centuries, dates__range)