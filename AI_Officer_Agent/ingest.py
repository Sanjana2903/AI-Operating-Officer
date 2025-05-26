import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingestion.pdf_loader import extract_text_from_pdf

embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

def ingest_persona_folder(folder_path, role, source_type):
    persona_docs = []
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if not fname.endswith((".pdf", ".txt")):
            print(f"⚠️ Skipping unsupported file: {fname}")
            continue

        if fname.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        # Optional: add a placeholder URL if you ever extract from web
        url = f"https://internal/{role}/{fname.replace(' ', '_')}"
        
        chunks = text_splitter.create_documents(
            [text],
            metadatas=[{
                "source": fname,
                "type": source_type,
                "persona": role,
                "url": url
            }]
        )

        print(f"📄 {fname} → {len(chunks)} chunks")
        persona_docs.extend(chunks)
    return persona_docs

def ingest_all_personas():
    all_docs = []
    all_docs += ingest_persona_folder("samples/ceo_satya", "CEO", "satya_content")
    all_docs += ingest_persona_folder("samples/cto_kevin", "CTO", "kevin_content")
    all_docs += ingest_persona_folder("samples/evp_pavan", "Product", "pavan_content")

    print(f"\n📥 Ingesting {len(all_docs)} persona chunks into ChromaDB...\n")
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_function,
        persist_directory="db"
    )
    db.persist()
    print("✅ Persona ingestion complete.")

if __name__ == "__main__":
    ingest_all_personas()
