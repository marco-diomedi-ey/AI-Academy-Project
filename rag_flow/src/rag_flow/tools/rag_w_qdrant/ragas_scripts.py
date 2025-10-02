from typing import List

from ragas import EvaluationDataset, evaluate
from ragas.metrics import \
    answer_correctness  
from ragas.metrics import \
     AnswerRelevancy  
from ragas.metrics import \
    context_precision  
from ragas.metrics import context_recall  
from ragas.metrics import faithfulness  

from .rag_structure import get_contexts_for_question
from .utils import Settings

def format_contexts_for_chain(contexts_with_metadata: List[dict]) -> str:
    """
    Format a list of contexts with metadata for RAG chain processing.
    
    Transforms retrieved contexts with metadata into a standardized string format
    suitable for RAG chain input. Includes source attribution and trustability
    indicators for transparency and quality tracking.
    
    Parameters
    ----------
    contexts_with_metadata : List[dict]
        List of dictionaries containing content, source, and metadata information.
        Each dictionary should have keys: 'content', 'source', and optional metadata.
        
    Returns
    -------
    str
        Formatted text with real sources and trustability indicators,
        separated by double newlines for clear context boundaries
        
    Format Structure
    ---------------
    Each context block follows the pattern::
    
        [source:filename][trustability: trusted] content_text
        
    Where:
    - **source**: Original document filename or identifier
    - **trustability**: Trust level indicator (defaults to "trusted")
    - **content**: Extracted text content from the context chunk
    
    Examples
    --------
    >>> contexts = [
    ...     {'content': 'Aircraft design principles...', 'source': 'manual.pdf'},
    ...     {'content': 'Safety regulations...', 'source': 'regulations.pdf'}
    ... ]
    >>> formatted = format_contexts_for_chain(contexts)
    >>> print(formatted)
    [source:manual.pdf][trustability: trusted] Aircraft design principles...
    
    [source:regulations.pdf][trustability: trusted] Safety regulations...
    
    Notes
    -----
    - Gracefully handles missing source information with "unknown" default
    - All contexts are marked as "trusted" by default
    - Double newline separation ensures clear context boundaries for LLM processing
    """
    formatted_contexts = []
    for ctx in contexts_with_metadata:
        content = ctx.get('content', '')
        source = ctx.get('source', 'unknown')
        formatted_contexts.append(f"[source:{source}][trustability: trusted] {content}")
    
    return "\n\n".join(formatted_contexts)

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Build RAGAS evaluation dataset from RAG pipeline execution.
    
    Executes the complete RAG pipeline for each question to generate the
    evaluation dataset required by RAGAS framework. Each dataset entry contains
    question, retrieved contexts, generated answer, and optional ground truth.
    
    Parameters
    ----------
    questions : List[str]
        List of questions to evaluate through the RAG pipeline
    retriever : VectorStoreRetriever
        Configured retriever for context extraction
    chain : RunnableSequence
        RAG chain for answer generation
    k : int
        Number of context chunks to retrieve per question
    ground_truth : dict[str, str], optional
        Dictionary mapping questions to their ground truth answers
        
    Returns
    -------
    List[dict]
        List of evaluation entries, each containing:
        - user_input: Input question
        - retrieved_contexts: Retrieved context chunks
        - response: Generated RAG answer
        - reference: Reference answer (if ground truth provided)
        
    Dataset Structure
    -----------------
    Each entry follows RAGAS expected format::
    
        {
            'user_input': str,
            'retrieved_contexts': List[str], 
            'response': str,
            'reference': str (optional)
        }
    
    Notes
    -----
    - Reference answer is optional but enables answer_correctness evaluation
    - Context extraction uses the configured retrieval strategy
    - Answer generation follows the complete RAG chain with question-context format
    - Dataset format is compatible with RAGAS EvaluationDataset.from_list()
    - Includes fallback for FAISS-based chains (commented line for direct question input)
    """
    dataset = []
    for q in questions:
        contexts_with_metadata = get_contexts_for_question(retriever, q, k)
        # answer = chain.invoke(q) SCOMMENTA PER FAISS
        ctx = format_contexts_for_chain(contexts_with_metadata)  
        answer = chain.invoke({"question": q, "context": ctx})

        contexts_for_ragas = [ctx_meta['content'] for ctx_meta in contexts_with_metadata]

        row = {
            "user_input": q,
            "retrieved_contexts": contexts_for_ragas,  
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


def ragas_evaluation(
    questions: List[str], chain, llm, embeddings, retriever, settings: Settings, ground_truth = None
):
    """
    Execute comprehensive RAGAS evaluation of RAG system performance.
    
    Performs end-to-end evaluation of the RAG pipeline using the RAGAS framework,
    measuring multiple quality metrics including faithfulness, answer relevancy,
    context precision, context recall, and optionally answer correctness when
    ground truth is provided.
    
    Parameters
    ----------
    questions : List[str]
        List of input questions for RAG system evaluation
    chain : Any
        RAG chain instance for answer generation
    llm : Any
        Language model instance for RAGAS metric computation
    embeddings : Any
        Embedding model for semantic similarity calculations
    retriever : Any
        Document retriever for context extraction
    settings : Settings
        Configuration object containing retrieval parameters (k, final_k)
    ground_truth : dict, optional
        Dictionary mapping questions to reference answers for answer_correctness evaluation
        
    Returns
    -------
    pandas.DataFrame
        Evaluation results with columns:
        - user_input: Original questions
        - response: Generated RAG answers
        - faithfulness: Answer groundedness in context (0-1)
        - answer_correctness: Correctness vs ground truth (0-1, if provided)
        - answer_relevancy: Answer relevance to question (0-1)
        - context_precision: Precision of retrieved contexts (0-1)
        - context_recall: Recall of retrieved contexts (0-1)
        
    Evaluation Metrics
    ------------------
    - **Faithfulness**: Measures how grounded the answer is in the provided context
    - **Answer Relevancy**: Evaluates how well the answer addresses the question
    - **Context Precision**: Assesses the precision of the retrieval system
    - **Context Recall**: Measures the recall of the retrieval system
    - **Answer Correctness**: Compares generated answer with ground truth (optional)
    
    Error Handling
    --------------
    The function includes fallback logic:
    1. First attempts retrieval with `settings.k` parameter
    2. Falls back to `settings.final_k` if the first attempt fails
    3. Ensures robust evaluation even with configuration issues
    
    Notes
    -----
    - Answer correctness metric is only included when ground truth is provided
    - Results are rounded for readability
    - AnswerRelevancy uses strictness=1 for rigorous evaluation
    - Requires properly configured LLM and embeddings for metric computation
    """
    try:
        dataset = build_ragas_dataset(
            questions=questions, retriever=retriever, chain=chain, k=settings.k, ground_truth=ground_truth
        )
    except Exception as e:
        dataset = build_ragas_dataset(
            questions=questions, retriever=retriever, chain=chain, k=settings.final_k, ground_truth=ground_truth
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    ar = AnswerRelevancy(strictness=1)
    metrics = [
            context_precision,
           context_recall,
        faithfulness,
        ar,
    ]
    if all("reference" in row for row in dataset):
        metrics.append(answer_correctness)

    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,  
        embeddings=embeddings,
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "faithfulness", "answer_correctness", "answer_relevancy", "context_precision", "context_recall"]
    return df[cols].round(4)