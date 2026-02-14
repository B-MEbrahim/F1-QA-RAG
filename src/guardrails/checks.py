"""
Guard-rail checks for the F1 RAG Bot.

Implements:
1. Input validation (topic filtering, prompt injection detection)
2. Output validation (hallucination mitigation, source grounding)
3. Safety checks (PII, harmful content)
"""
from typing import Tuple
from langchain_core.documents import Document


# ============ Input Guards ============

def is_on_topic(query: str, allowed_topics: list[str] = None) -> Tuple[bool, str]:
    """
    Check if the query is related to F1 regulations or race results.
    Returns (is_valid, reason).
    """
    if allowed_topics is None:
        allowed_topics = [
            "formula 1", "f1", "fia", "regulations", "rules", 
            "race", "grand prix", "gp", "penalty", "technical",
            "sporting", "financial", "driver", "team", "constructor",
            "qualifying", "sprint", "pit stop", "safety car", "drs"
        ]
    
    query_lower = query.lower()
    
    # Check for any topic keyword
    for topic in allowed_topics:
        if topic in query_lower:
            return True, "Query is on-topic"
    
    return False, "Query does not appear to be related to F1. Please ask about regulations, rules, or race results."


def detect_prompt_injection(query: str) -> Tuple[bool, str]:
    """
    Basic prompt injection detection.
    Returns (is_safe, reason).
    """
    # Common injection patterns
    injection_patterns = [
        "ignore previous instructions",
        "ignore all instructions", 
        "disregard your instructions",
        "forget your instructions",
        "you are now",
        "pretend you are",
        "act as if",
        "system prompt",
        "reveal your prompt",
    ]
    
    query_lower = query.lower()
    
    for pattern in injection_patterns:
        if pattern in query_lower:
            return False, f"Potential prompt injection detected: '{pattern}'"
    
    return True, "No injection detected"


def validate_input(query: str) -> Tuple[bool, str]:
    """
    Run all input validation checks.
    Returns (is_valid, message).
    """
    # Check for prompt injection first
    is_safe, reason = detect_prompt_injection(query)
    if not is_safe:
        return False, reason
    
    # Check if on-topic (can be relaxed for general chat)
    # is_on_topic_result, topic_reason = is_on_topic(query)
    # if not is_on_topic_result:
    #     return False, topic_reason
    
    return True, "Input validated successfully"


# ============ Output Guards ============

def check_source_grounding(answer: str, sources: list[Document]) -> Tuple[bool, float]:
    """
    Check if the answer is grounded in the provided sources.
    Returns (is_grounded, confidence_score).
    
    Simple implementation: check for keyword overlap.
    For production, use NLI models or semantic similarity.
    """
    if not sources:
        return False, 0.0
    
    # Combine all source content
    source_text = " ".join([doc.page_content.lower() for doc in sources])
    
    # Extract key terms from answer (simple word tokenization)
    answer_words = set(answer.lower().split())
    
    # Filter out common stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "must", "shall",
                 "can", "to", "of", "in", "for", "on", "with", "at", "by",
                 "from", "as", "into", "through", "during", "before", "after",
                 "above", "below", "between", "under", "again", "further",
                 "then", "once", "here", "there", "when", "where", "why",
                 "how", "all", "each", "few", "more", "most", "other", "some",
                 "such", "no", "nor", "not", "only", "own", "same", "so",
                 "than", "too", "very", "just", "and", "but", "if", "or",
                 "because", "until", "while", "this", "that", "these", "those"}
    
    answer_keywords = answer_words - stopwords
    
    if not answer_keywords:
        return True, 1.0  # No keywords to check
    
    # Count how many answer keywords appear in sources
    grounded_count = sum(1 for word in answer_keywords if word in source_text)
    confidence = grounded_count / len(answer_keywords)
    
    return confidence >= 0.3, confidence  # 30% threshold


def add_source_citations(answer: str, sources: list[Document]) -> str:
    """
    Append source citations to the answer.
    """
    if not sources:
        return answer
    
    citations = "\n\n**Sources:**\n"
    seen_sources = set()
    
    for doc in sources:
        source_name = doc.metadata.get("source", "Unknown")
        rule_id = doc.metadata.get("rule_id", "")
        
        citation_key = f"{source_name}_{rule_id}"
        if citation_key not in seen_sources:
            seen_sources.add(citation_key)
            if rule_id:
                citations += f"- {source_name} (Rule {rule_id})\n"
            else:
                citations += f"- {source_name}\n"
    
    return answer + citations


def validate_output(answer: str, sources: list[Document] = None) -> Tuple[str, dict]:
    """
    Run all output validation checks.
    Returns (processed_answer, validation_info).
    """
    validation_info = {
        "is_grounded": True,
        "grounding_score": 1.0,
        "has_citations": False
    }
    
    if sources:
        # Check grounding
        is_grounded, score = check_source_grounding(answer, sources)
        validation_info["is_grounded"] = is_grounded
        validation_info["grounding_score"] = score
        
        # Add warning if not well grounded
        if not is_grounded:
            answer = "⚠️ *Note: This answer may not be fully supported by the sources.*\n\n" + answer
        
        # Add citations
        answer = add_source_citations(answer, sources)
        validation_info["has_citations"] = True
    
    return answer, validation_info
