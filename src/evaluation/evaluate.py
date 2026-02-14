"""
Evaluation pipeline for the F1 RAG Bot.

Implements metrics for:
1. Retrieval quality (precision, recall, MRR)
2. Answer quality (faithfulness, relevance)
3. End-to-end evaluation

Can be extended with RAGAS library for more comprehensive evaluation.
"""
import json
from typing import Optional
from dataclasses import dataclass, asdict
from langchain_core.documents import Document


@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    expected_answer: Optional[str] = None
    expected_source: Optional[str] = None  # Expected rule_id or source filename
    actual_answer: Optional[str] = None
    retrieved_docs: Optional[list] = None
    

@dataclass  
class EvalMetrics:
    """Evaluation metrics for a single sample."""
    retrieval_hit: bool = False  # Did we retrieve the expected source?
    answer_relevance: float = 0.0  # How relevant is the answer to the question?
    faithfulness: float = 0.0  # Is the answer grounded in retrieved context?
    latency_ms: float = 0.0  # Response time
    

def calculate_retrieval_hit(
    expected_source: str, 
    retrieved_docs: list[Document]
) -> bool:
    """
    Check if the expected source was retrieved.
    """
    if not expected_source or not retrieved_docs:
        return False
    
    expected_lower = expected_source.lower()
    
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "").lower()
        rule_id = doc.metadata.get("rule_id", "").lower()
        
        if expected_lower in source or expected_lower in rule_id:
            return True
    
    return False


def calculate_keyword_overlap(text1: str, text2: str) -> float:
    """
    Simple keyword overlap score between two texts.
    Returns a score between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word tokenization
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Filter stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "have", 
                 "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "to", "of", "in", "for", "on", "with", "at", "by",
                 "and", "or", "but", "if", "this", "that", "it"}
    
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def evaluate_sample(sample: EvalSample) -> EvalMetrics:
    """
    Evaluate a single sample and return metrics.
    """
    metrics = EvalMetrics()
    
    # Retrieval hit
    if sample.expected_source and sample.retrieved_docs:
        metrics.retrieval_hit = calculate_retrieval_hit(
            sample.expected_source, 
            sample.retrieved_docs
        )
    
    # Answer relevance (keyword overlap with question - simple proxy)
    if sample.actual_answer and sample.question:
        metrics.answer_relevance = calculate_keyword_overlap(
            sample.actual_answer, 
            sample.question
        )
    
    # Faithfulness (overlap with retrieved context)
    if sample.actual_answer and sample.retrieved_docs:
        context = " ".join([doc.page_content for doc in sample.retrieved_docs])
        metrics.faithfulness = calculate_keyword_overlap(
            sample.actual_answer, 
            context
        )
    
    return metrics


def run_evaluation(samples: list[EvalSample]) -> dict:
    """
    Run evaluation on a list of samples and return aggregate metrics.
    """
    if not samples:
        return {"error": "No samples provided"}
    
    all_metrics = [evaluate_sample(s) for s in samples]
    
    # Aggregate metrics
    n = len(all_metrics)
    
    return {
        "num_samples": n,
        "retrieval_hit_rate": sum(m.retrieval_hit for m in all_metrics) / n,
        "avg_answer_relevance": sum(m.answer_relevance for m in all_metrics) / n,
        "avg_faithfulness": sum(m.faithfulness for m in all_metrics) / n,
        "avg_latency_ms": sum(m.latency_ms for m in all_metrics) / n,
    }


# ============ Test Data ============

EVAL_DATASET = [
    EvalSample(
        question="What is the minimum weight of an F1 car in 2026?",
        expected_source="Section C [Technical]",
        expected_answer=None  # Fill in expected answer if known
    ),
    EvalSample(
        question="What are the penalties for exceeding track limits?",
        expected_source="Section B [Sporting]",
    ),
    EvalSample(
        question="What is the budget cap for F1 teams in 2026?",
        expected_source="Section D [Financial Regulations",
    ),
    EvalSample(
        question="What are the DRS rules for 2026?",
        expected_source="Section C [Technical]",
    ),
    EvalSample(
        question="What is the parc ferme procedure?",
        expected_source="Section B [Sporting]",
    ),
]


def save_eval_results(results: dict, filepath: str = "eval_results.json"):
    """Save evaluation results to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Running evaluation with test dataset...")
    
    # In practice, you would run the RAG pipeline on each question 
    # and populate retrieved_docs and actual_answer
    
    # For now, just show the structure
    for sample in EVAL_DATASET:
        print(f"Q: {sample.question}")
        print(f"   Expected source: {sample.expected_source}")
        print()
