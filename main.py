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

    files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf") or f.lower().endswith(".md")]

    if not files:
        print("Aucun fichier PDF ou Markdown trouvé dans le répertoire ./doc/data")
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

    for file_path in files:
        print(f"--------------------\nTraitement du fichier: {file_path}")
        file_name = os.path.basename(file_path)
        if file_path.lower().endswith(".pdf"):
            # Convert PDF to Markdown
            print("Conversion du PDF en Markdown...")
            start_time = time.time()
            md_str = pdf2md(file_path)
            end_time = time.time() - start_time
            print(f"La conversion en Markdown a pris {end_time:.2f} seconds")
        elif file_path.lower().endswith(".md"):
            print("Lecture du fichier Markdown...")
            with open(file_path, "r", encoding="utf-8") as f:
                md_str = f.read()
        else:
            print(f"Format de fichier non supporté pour {file_name}, saut du fichier.")
            continue
        
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




