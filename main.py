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

jsonld_graph = utils.export_kg_to_jsonld(kg)
# utils.save_jsonld_to_file(jsonld_graph, "data/jbc_knowledge_graph.jsonld")