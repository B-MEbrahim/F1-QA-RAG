import os
import argparse
import pymupdf4llm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config.config import EMBEDDING_MODEL, CHROMADB_DIR
from .chunk import chunk_fia_document, generate_chunk_id

# init embedder
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def run_ingestion(year_dir):
    # extract the year from the directory name 
    year = os.path.basename(year_dir.rstrip('/'))

    # init vecore db
    vector_store = Chroma(
                        collection_name=year,
                        embedding_function=embeddings,
                        persist_directory=CHROMADB_DIR,  
                        )
    
    all_chunks = []

    # Loop through all PDFs in the year directory
    for filename in os.listdir(year_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(year_dir, filename)
            print(f"--- Processing: {filename} ---")

            # extract PDF to Markdown
            md_text = pymupdf4llm.to_markdown(file_path)
            # chunk file
            chunks = chunk_fia_document(md_text, file_path)

            all_chunks.extend(chunks)

    # upsert to ChromaDB
    if all_chunks:
        ids = [
            generate_chunk_id(doc, i)
            for i, doc in enumerate(all_chunks)
        ]

        print(f"--- Adding {len(all_chunks)} total chunks to Vector DB ---")
        vector_store.add_documents(
            documents=all_chunks, 
            ids=ids
        )
        print(f"--- Successfully ingested rules for {year} ---")
    else:
        print("No PDF files found in the directory.")


def ingest_pdf_to_collection(file_path: str, collection_name: str) -> int:
    """
    Ingest a single PDF into the specified Chroma collection.
    Returns the number of chunks added.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMADB_DIR,
    )

    md_text = pymupdf4llm.to_markdown(file_path)
    chunks = chunk_fia_document(md_text, file_path)

    if not chunks:
        return 0

    ids = [
        generate_chunk_id(doc, i)
        for i, doc in enumerate(chunks)
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return len(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to the year directory (e.g., data/raw/2026)")
    args = parser.parse_args()
    
    run_ingestion(args.dir)