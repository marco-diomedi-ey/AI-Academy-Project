from typing import List

from ragas import EvaluationDataset, evaluate
from ragas.metrics import \
    answer_correctness  # usa questa solo se hai ground_truth
from ragas.metrics import \
    answer_relevancy, AnswerRelevancy  # pertinenza della risposta vs domanda
from ragas.metrics import \
    context_precision  # "precision@k" sui chunk recuperati
from ragas.metrics import context_recall  # copertura dei chunk rilevanti
from ragas.metrics import faithfulness  # ancoraggio della risposta al contesto

from .rag_structure import get_contexts_for_question
from .utils import Settings
# from rag_structure import get_contexts_for_question
# from utils import Settings

def format_contexts_for_chain(contexts_with_metadata: List[dict]) -> str:
    """
    Formatta una lista di contexts con metadati per la chain.
    
    Parameters
    ----------
    contexts_with_metadata : List[dict]
        Lista di dizionari contenenti content, source e metadata
        
    Returns
    -------
    str
        Testo formattato con source reali e trustability
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
        - question: Input question
        - contexts: Retrieved context chunks
        - answer: Generated RAG answer
        - ground_truth: Reference answer (if provided)
        
    Dataset Structure
    -----------------
    Each entry follows RAGAS expected format::
    
        {
            'question': str,
            'contexts': List[str], 
            'answer': str,
            'ground_truth': str (optional)
        }
    
    Notes
    -----
    - Ground truth is optional but enables answer_correctness evaluation
    - Context extraction uses the configured retrieval strategy
    - Answer generation follows the complete RAG chain
    - Dataset format is compatible with RAGAS EvaluationDataset
    """
    dataset = []
    for q in questions:
        contexts_with_metadata = get_contexts_for_question(retriever, q, k)
        # answer = chain.invoke(q) SCOMMENTA PER FAISS
        ctx = format_contexts_for_chain(contexts_with_metadata)  # Formatta per la chain
        answer = chain.invoke({"question": q, "context": ctx})

        # Estrai solo il contenuto per RAGAS (che si aspetta List[str])
        contexts_for_ragas = [ctx_meta['content'] for ctx_meta in contexts_with_metadata]

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts_for_ragas,  # RAGAS vuole List[str]
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


def ragas_evaluation(
    questions: str, chain, llm, embeddings, retriever, settings: Settings, ground_truth = None
):
    # questions = [question]
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
    # 7) Scegli le metriche
    metrics = [
            context_precision,
           context_recall,
        faithfulness,
        ar,
    ]
    # Aggiungi correctness solo se tutte le righe hanno reference
    if all("reference" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,  # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=embeddings,
        # ground_truth = ground_truth  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "faithfulness", "answer_correctness", "answer_relevancy", "context_precision", "context_recall"]
    return df[cols].round()