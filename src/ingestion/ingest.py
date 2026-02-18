import os
import argparse
import pymupdf4llm
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)
from langchain_core.documents import Document
from config.config import EMBEDDING_MODEL, TOKENIZER, CHUNK_SIZE, CHUNK_OVERLAP, CHROMADB_DIR


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

HEADER_TO_SPLIT_ON = [
    ("####", "clause"),
    ("###", "sub_article"),
    ("##", "article"),
    ("#", "section")
]

# init embedder
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# def token_len(text):
#     return len(tokenizer.encode(text, add_special_tokens=True))

# def extract_metadata_from_filename(file_path: str):

#     file_name = Path(file_path).stem

#     pattern = (
#         r"FIA\s+(?P<year>\d{4})\s+F1 Regulations\s+-\s+"
#         r"Section\s+(?P<section>[A-Z])\s+\[(?P<section_name>.*?)\]\s+-\s+"
#         r"Iss\s+(?P<issue>\d+)\s+-\s+(?P<date>\d{4}-\d{2}-\d{2})"
#     )
#     match = re.search(pattern, file_name)

#     if not match:
#         return {"source": file_name}
    
#     meta = match.groupdict()

#     return {
#         "source": file_name,
#         "regulation_year": meta["year"],
#         "section": meta["section"],
#         "section_name": meta["section_name"],
#         "issue": meta["issue"],
#         "publication_date": meta["date"]
#     }

# def normalize_file_markdown(md):
#     # remove empty lines
#     md = re.sub(r"\n{3,}", "\n\n", md)

#     # remove page footer
#     md = re.sub(
#         r"2026 Formula 1 Regulations.*?Issue \d+",
#         "",
#         md
#     )

#     # article to level 2 header
#     md = re.sub(
#         r"\*\*ARTICLE\s+(.*?)\*\*",
#         r"## ARTICLE \1",
#         md
#     )

#     # F3.1 to level 3
#     md = re.sub(
#         r"\*\*(F\d+\.\d+)\*\*\s+\*\*(.*?)\*\*",
#         r"### \1 \2",
#         md
#     )

#     # F3.1.1 to level 4
#     md = re.sub(
#         r"\*\*(F\d+\.\d+\.\d+)\*\*",
#         r"#### \1",
#         md
#     )

#     return md

# def extract_rule_id(text: str):

#     match = re.search(r"(F\d+\.\d+(\.\d+)?)", text)

#     if match:
#         return match.group(1)
    
#     return None

# def chunk_fia_document(md_text: str, file_path: str):

#     # extract file metadata
#     file_metadata = extract_metadata_from_filename(file_path)

#     # normalize markdown
#     md_text = normalize_file_markdown(md_text)

#     # header splitting
#     header_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=HEADER_TO_SPLIT_ON
#     )
#     docs = header_splitter.split_text(md_text)

#     # token splitting 
#     # Using actual E5 tokenizer - safe limit is 480 tokens (with buffer under 512 max)
#     token_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE, 
#         chunk_overlap=CHUNK_OVERLAP,
#         length_function=token_len
#     )
#     docs = token_splitter.split_documents(docs)

#     # Safety check: truncate any chunks still over limit
#     MAX_TOKENS = 480
    
#     for doc in docs:
#         current_len = token_len(doc.page_content)
#         if current_len > MAX_TOKENS:
#             # Truncate to max tokens using the correct tokenizer
#             tokens = tokenizer.encode(doc.page_content, add_special_tokens=True)[:MAX_TOKENS]
#             doc.page_content = tokenizer.decode(tokens, skip_special_tokens=True)
#             print(f"Truncated chunk from {current_len} to {MAX_TOKENS} tokens")

#     # enrich metadata
#     final_docs = []

#     for d in docs:
#         # add filename metadata
#         metadata = {**file_metadata, **d.metadata}

#         # extract rule id
#         rule_id = extract_rule_id(d.page_content)
#         if rule_id:
#             metadata["rule_id"] = rule_id

#         final_docs.append(
#             Document(
#                 page_content=d.page_content,
#                 metadata=metadata
#             )
#         )

#     return final_docs

# def generate_chunk_id(doc, index):
#     base = (
#         doc.metadata.get("source", "") + 
#         doc.metadata.get("rule_id", "") + 
#         str(index)
#     )
#     return hashlib.md5(base.encode()).hexdigest()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to the year directory (e.g., data/raw/2026)")
    args = parser.parse_args()
    
    run_ingestion(args.dir)