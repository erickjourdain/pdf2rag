import sys
import time
import os

from ollama import OllamaClient
from markdownPreprocessor import MarkdownPreprocessor
import ollama
from pdf2md import pdf2md
from qdrant import QdrantClient

if __name__ == "__main__":  
    url_ollama = "http://localhost:11434/api"
    url_qdrant = "http://192.168.1.70:6333/"
    collection_name = "cgrer"
    pdf_dir = "./doc/data"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("Aucun fichier PDF trouvé dans le répertoire ./doc/data")
        sys.exit(1)

    ollama = OllamaClient(url=url_ollama)
    qdrant = QdrantClient(url=url_qdrant)
    qdrant.delete_collection(collection_name)
    if not qdrant.collection_exists(collection_name):
        print(f"Création de la collection '{collection_name}' dans Qdrant...")
        qdrant.create_collection(collection_name, vector_size=768, distance="Cosine")
        print(f"Collection '{collection_name}' créée avec succès")
    else:
        print(f"La collection '{collection_name}' existe déjà dans Qdrant.")

    for pdf_file_path in pdf_files:
        print(f"--------------------\nTraitement du fichier: {pdf_file_path}")
        file_name = os.path.basename(pdf_file_path)
        # Convert PDF to Markdown
        print("Conversion du PDF en Markdown...")
        start_time = time.time()
        md_str = pdf2md(pdf_file_path)
        end_time = time.time() - start_time
        print(f"La conversion en Markdown a pris {end_time:.2f} seconds")

        print("Nettoyage et découpage du Markdown...")
        md_preprocessor = MarkdownPreprocessor(chunk_size=1000, overlap=100)
        chunks = md_preprocessor.process(md_str, source=file_name)
        print(f"Nombre de chunks générés: {len(chunks)}")

        print("Génération des embeddings avec Ollama...")
        points = []
        for c in chunks:
            vector = ollama.embeddings(c["text"], url=url_ollama)
            points.append({
                "id": c["meta"]["global_index"],
                "vector": vector,
                "payload": {
                    "content": c["text"],
                    "metadata": {
                        **c["meta"]
                    }
                }
            })
        print(f"Nombre d'embeddings générés: {len(points)}")

        print("Insertion des points dans Qdrant...")
        qdrant.upsert_points(collection_name, points)
        print("Insertion des données terminée")
        print("Traitement du fichier terminé\n--------------------\n")        




