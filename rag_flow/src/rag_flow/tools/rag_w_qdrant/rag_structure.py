import warnings
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .azure_connections import get_llm


warnings.filterwarnings("ignore", category=UserWarning)


def get_contexts_for_question(retriever, question: str, k: int) -> List[dict]:
    """
    Retrieve top-k document chunks with metadata as context for a given question.
    
    Uses the configured retriever to find the most relevant document chunks
    for the input question, returning both content and metadata for context formation.
    Handles source extraction from various metadata formats for proper attribution.
    
    Parameters
    ----------
    retriever : VectorStoreRetriever
        Configured retriever from vector store (FAISS, Qdrant, or other)
    question : str
        Input question to retrieve relevant context for
    k : int
        Maximum number of document chunks to retrieve
        
    Returns
    -------
    List[dict]
        List of dictionaries containing content and metadata from the top-k 
        most relevant document chunks. Each dict has:
        - 'content': str - The text content
        - 'source': str - Source document filename (extracted from path)
        - 'metadata': dict - Additional document metadata
        
    Examples
    --------
    >>> contexts = get_contexts_for_question(retriever, "How does the engine work?", k=3)
    >>> for ctx in contexts:
    ...     print(f"Source: {ctx['source']}")
    ...     print(f"Content: {ctx['content'][:100]}...")
    Source: engine_manual.pdf
    Content: The engine operates through a four-stroke cycle...
    
    Notes
    -----
    - The retriever uses the configured search strategy (MMR or similarity)
    - Source extraction handles both 'source' and 'file_path' metadata keys
    - File paths are processed to extract only the filename
    - Returns documents with full metadata for proper source attribution
    - Gracefully handles missing metadata with "unknown" default source
    """
    docs = retriever.invoke(question)[:k]
    contexts_with_metadata = []
    for doc in docs:
        source = "unknown"
        if hasattr(doc, 'metadata') and doc.metadata:
            if 'source' in doc.metadata:
                source = doc.metadata['source'].split('\\')[-1].split('/')[-1]  
            elif 'file_path' in doc.metadata:
                source = doc.metadata['file_path'].split('\\')[-1].split('/')[-1]
        
        contexts_with_metadata.append({
            'content': doc.page_content,
            'source': source,
            'metadata': getattr(doc, 'metadata', {})
        })
    
    return contexts_with_metadata




def build_rag_chain(llm):
    """
    Build a complete RAG chain for Italian technical assistance.
    
    Creates a LangChain pipeline that processes questions and contexts to generate
    accurate Italian responses based exclusively on provided content. Includes
    trust-based filtering to exclude untrusted sources and mandatory source citations.
    
    Parameters
    ----------
    llm : Any
        Language model instance for response generation (typically Azure OpenAI)
        
    Returns
    -------
    RunnableSequence
        Complete RAG chain ready for question-context processing with components:
        - Input mapping for question and context parameters
        - ChatPromptTemplate with Italian system instructions
        - LLM for response generation
        - StrOutputParser for clean text output
        
    Chain Behavior
    --------------
    The chain enforces strict content-only responses with:
    - **Language**: Responses exclusively in Italian
    - **Content Restriction**: Only information from provided context
    - **Source Attribution**: Mandatory [source:FILE] citations
    - **Trust Filtering**: Ignores content marked as 'untrusted'
    - **Fallback Response**: "Non è presente nel contesto fornito" for missing info
    
    Input Format
    -----------
    Expected input dictionary::
    
        {
            "question": str,
            "context": str  # Formatted context with source attribution
        }
    
    Examples
    --------
    >>> llm = get_llm()
    >>> chain = build_rag_chain(llm)
    >>> response = chain.invoke({
    ...     "question": "Come funziona il motore?",
    ...     "context": "[source:manual.pdf] Il motore funziona tramite..."
    ... })
    
    Notes
    -----
    - System prompt is in Italian to ensure consistent Italian responses
    - Built-in trustability filtering prevents use of untrusted sources
    - RunnablePassthrough allows flexible input parameter mapping
    - Chain follows LangChain Expression Language (LCEL) pattern
    """
    system_prompt = (
        "Sei un assistente tecnico. Rispondi in italiano, conciso e accurato. "
        "Usa ESCLUSIVAMENTE le informazioni presenti nel CONTENUTO. "
        "Se non è presente, dichiara: 'Non è presente nel contesto fornito.' "
        "Cita sempre le fonti nel formato [source:FILE]."
        "SE dentro i metadati del documento è presente alla chiave 'trustability' il valore 'untrusted' non prendere in considerazione il contenuto"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "CONTENUTO:\n{context}\n\n"
         "Istruzioni:\n"
         "1) Risposta basata solo sul contenuto.\n"
         "2) Includi citazioni [source:...].\n"
         "3) Niente invenzioni."
         "4) SE dentro i metadati del documento è presente il valore 'untrusted' non prendere in considerazione il contenuto")
    ])

    chain = (
        {
            "context": RunnablePassthrough(),  
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """
    Execute RAG chain to generate answer for a single question.
    
    Invokes the RAG chain with the provided question to generate an answer.
    This function is a simple wrapper for direct question-only processing,
    where context retrieval and formatting are handled externally.
    
    Parameters
    ----------
    question : str
        Input question to be answered using the RAG system
    chain : RunnableSequence
        Configured RAG chain from build_rag_chain()
        
    Returns
    -------
    str
        Generated answer with source citations in Italian
        
    Notes
    -----
    This function assumes:
    - The chain is configured to handle question-only input
    - Context retrieval and formatting are handled separately
    - The chain will process the question and generate an appropriate response
    - For question+context processing, use chain.invoke({"question": q, "context": ctx})
    
    Examples
    --------
    >>> chain = build_rag_chain(llm)
    >>> answer = rag_answer("Come funziona il sistema?", chain)
    >>> print(answer)
    
    See Also
    --------
    build_rag_chain : Create the RAG chain used by this function
    get_contexts_for_question : Retrieve contexts for question-context processing
    """
    return chain.invoke(question)


def keywords_generation(query: str) -> List[str]:
    """
    Generate web search keywords from a user query using LLM.
    
    Uses the configured language model to extract and expand relevant
    keywords from the input query for enhanced web search capabilities.
    Useful for enriching RAG context with web-sourced information.
    
    Parameters
    ----------
    query : str
        Input query to extract keywords from
        
    Returns
    -------
    List[str]
        List of generated keywords suitable for web search, split from
        comma-separated LLM response
        
    Examples
    --------
    >>> keywords = keywords_generation("aircraft engine design principles")
    >>> print(keywords)
    ['aircraft', 'engine', 'design', 'principles', 'aviation', 'turbine']
    
    Notes
    -----
    - Uses Azure OpenAI model via get_llm() for keyword generation
    - Keywords are comma-separated in the LLM response and split into list
    - Results are printed for debugging purposes
    - Intended for use with web search tools like DuckDuckGo or SerperDev
    - LLM prompt is in English to ensure consistent keyword format
    """
    llm = get_llm()

    prompt = f"""You are a helpful assistant. Generate keywords separated with commas for web search based on the user's query.
    
    Query: {query}
    
    Keywords:"""

    response = llm.invoke(prompt)
    keywords = response.content.strip().split(", ")
    print(keywords)
    return keywords
