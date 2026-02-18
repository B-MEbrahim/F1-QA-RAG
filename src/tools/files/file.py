import re
from pathlib import Path

def extract_metadata_from_filename(file_path: str):

    file_name = Path(file_path).stem

    pattern = (
        r"FIA\s+(?P<year>\d{4})\s+F1 Regulations\s+-\s+"
        r"Section\s+(?P<section>[A-Z])\s+\[(?P<section_name>.*?)\]\s+-\s+"
        r"Iss\s+(?P<issue>\d+)\s+-\s+(?P<date>\d{4}-\d{2}-\d{2})"
    )
    match = re.search(pattern, file_name)

    if not match:
        return {"source": file_name}
    
    meta = match.groupdict()

    return {
        "source": file_name,
        "regulation_year": meta["year"],
        "section": meta["section"],
        "section_name": meta["section_name"],
        "issue": meta["issue"],
        "publication_date": meta["date"]
    }

def normalize_file_markdown(md):
    # remove empty lines
    md = re.sub(r"\n{3,}", "\n\n", md)

    # remove page footer
    md = re.sub(
        r"2026 Formula 1 Regulations.*?Issue \d+",
        "",
        md
    )

    # article to level 2 header
    md = re.sub(
        r"\*\*ARTICLE\s+(.*?)\*\*",
        r"## ARTICLE \1",
        md
    )

    # F3.1 to level 3
    md = re.sub(
        r"\*\*(F\d+\.\d+)\*\*\s+\*\*(.*?)\*\*",
        r"### \1 \2",
        md
    )

    # F3.1.1 to level 4
    md = re.sub(
        r"\*\*(F\d+\.\d+\.\d+)\*\*",
        r"#### \1",
        md
    )

    return md

def extract_rule_id(text: str):

    match = re.search(r"(F\d+\.\d+(\.\d+)?)", text)

    if match:
        return match.group(1)
    
    return None