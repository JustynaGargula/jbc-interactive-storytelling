import utils
from google import genai

part = 1
rdfs_directory_path = f"./data/part{part}/*rdf"
g = utils.create_graph(rdfs_directory_path)

kg = utils.build_kg_from_rdf(g)

ids = kg.by_subject.get("druki ulotne 19 w.")
data = []
for id in ids:
    doc = kg.documents.get(id)
    data.append(doc)
# print(data)

prompt = f"Kontekst: Skorzystaj przede wszystkim z tych danych: {data}. Zadanie: Co się działo w 19 wieku?"

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=prompt
)
print(response.text)