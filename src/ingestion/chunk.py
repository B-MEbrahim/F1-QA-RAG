import hashlib
from src.tools import extract_metadata_from_filename, normalize_file_markdown, extract_rule_id
from langchain_text_splitters import (MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)
from langchain_core.documents import Document
from transformers import AutoTokenizer
from config.config import TOKENIZER, CHUNK_SIZE, CHUNK_OVERLAP


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

HEADER_TO_SPLIT_ON = [
    ("####", "clause"),
    ("###", "sub_article"),
    ("##", "article"),
    ("#", "section")
]

def token_len(text):
    return len(tokenizer.encode(text, add_special_tokens=True))
    
def generate_chunk_id(doc, index):
    base = (
        doc.metadata.get("source", "") + 
        doc.metadata.get("rule_id", "") + 
        str(index)
    )
    return hashlib.md5(base.encode()).hexdigest()

def chunk_fia_document(md_text: str, file_path: str):

    # extract file metadata
    file_metadata = extract_metadata_from_filename(file_path)

    # normalize markdown
    md_text = normalize_file_markdown(md_text)

    # header splitting
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADER_TO_SPLIT_ON
    )
    docs = header_splitter.split_text(md_text)

    # token splitting 
    # Using actual E5 tokenizer - safe limit is 480 tokens (with buffer under 512 max)
    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len
    )
    docs = token_splitter.split_documents(docs)

    # Safety check: truncate any chunks still over limit
    MAX_TOKENS = 480
    
    for doc in docs:
        current_len = token_len(doc.page_content)
        if current_len > MAX_TOKENS:
            # Truncate to max tokens using the correct tokenizer
            tokens = tokenizer.encode(doc.page_content, add_special_tokens=True)[:MAX_TOKENS]
            doc.page_content = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"Truncated chunk from {current_len} to {MAX_TOKENS} tokens")

    # enrich metadata
    final_docs = []

    for d in docs:
        # add filename metadata
        metadata = {**file_metadata, **d.metadata}

        # extract rule id
        rule_id = extract_rule_id(d.page_content)
        if rule_id:
            metadata["rule_id"] = rule_id

        final_docs.append(
            Document(
                page_content=d.page_content,
                metadata=metadata
            )
        )

    return final_docs