import utils

# wyselekcjonowane cechy to: title, date, description, subject, identifier, creator, publisher, type

file = ".\data\data_ris\dlibra.ris"
ids = utils.get_ids(file)

part = 1
# rdfs = utils.get_rdfs(ids)
# utils.save_rdfs_to_file(rdfs, ids, part)

rdfs_directory_path = f"./data/part{part}/*rdf"
g = utils.create_graph(rdfs_directory_path)
# utils.save_data_to_one_file(g, "turtle", ".ttl")

kg = utils.build_kg_from_rdf(g)
print(f"Wczytano {len(kg.documents)} dokumentów do grafu wiedzy.")

# jsonld_graph = utils.export_kg_to_jsonld(kg)
# utils.save_jsonld_to_file(jsonld_graph, "data/jbc_knowledge_graph.jsonld")

# wymagania
century = 19
subject = "Teatr Krakowski 19 w."
subject2 = "Teatr Łódzki (Polska)"

# test funkcji pobierającej dokumenty na podstawie filtrów i powiązanych dokumentów
selected_docs = utils.get_documents_from_filters_and_related(kg, years=[], centuries=[century], subjects=[subject, subject2])
print(f"Znaleziono {len(selected_docs)} dokumentów spełniających kryteria.")
for doc in selected_docs:
    print(doc.identifier, doc.title)