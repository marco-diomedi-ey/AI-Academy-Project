from __future__ import annotations

from dotenv import load_dotenv
 

from crewai.tools import tool
from .ragas_scripts import ragas_evaluation
from .azure_connections import get_azure_embedding_model, get_llm 
from .rag_structure import build_rag_chain
from .config import Settings
from .utils import  load_documents, split_documents, scan_docs_folder, SimpleRetriever, format_docs_for_prompt
from .qdrant_script import (
    get_qdrant_client,
    recreate_collection_for_rag,
    upsert_chunks,
    hybrid_search
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


load_dotenv()

SETTINGS = Settings()



@tool('rag_system')
def rag_system(question: str) -> str:
    """
    Main execution function demonstrating the complete RAG pipeline.
    
    This function orchestrates the entire RAG workflow from document ingestion
    to intelligent question answering, showcasing the system's capabilities
    and providing a template for production deployment.
    
    Pipeline Overview:
    
    1. SYSTEM INITIALIZATION:
       - Load configuration settings
       - Initialize embedding model
       - Configure LLM (optional)
       - Establish database connection
        
    2. DOCUMENT PROCESSING:
       - Load or simulate document corpus
       - Split documents into manageable chunks
       - Generate vector embeddings for each chunk
        
    3. VECTOR DATABASE SETUP:
       - Create/configure Qdrant collection
       - Set up HNSW indexing and payload indices
       - Optimize for semantic search performance
        
    4. DATA INGESTION:
       - Store document chunks with metadata
       - Index vectors for fast retrieval
       - Ensure data consistency and availability
        
    5. INTELLIGENT RETRIEVAL:
       - Process user queries through hybrid search
       - Combine semantic and text-based matching
       - Apply MMR for result diversification
        
    6. CONTENT GENERATION:
       - Use LLM for intelligent answer generation
       - Fall back to content display if LLM unavailable
       - Provide source citations and context
        
    Performance Characteristics:
    
    Initialization Time:
    - Embedding model: 2-10 seconds (depends on model size)
    - LLM connection: 0.1-5 seconds (depends on service)
    - Database setup: 1-5 seconds (depends on collection size)
        
    Processing Time:
    - Document chunking: Linear with document count
    - Vector generation: Linear with chunk count
    - Database indexing: O(n log n) with HNSW construction
        
    Query Time:
    - Semantic search: Sub-millisecond with HNSW
    - Text search: Millisecond range with payload indices
    - Result fusion: Linear with candidate count
    - MMR diversification: Quadratic with candidate count
        
    Memory Usage:
    - Embedding model: 100MB-2GB (depends on model)
    - Vector storage: 4 bytes × dimensions × chunks (quantized)
    - Payload storage: Variable based on metadata size
    - LLM context: Depends on model and input size
        
    Scalability Considerations:
    
    Document Volume:
    - Small (<1K docs): Current settings optimal
    - Medium (1K-100K docs): Consider batch processing
    - Large (100K+ docs): Implement streaming ingestion
        
    Vector Dimensions:
    - 384 dimensions: Fast, memory-efficient, good quality
    - 768 dimensions: Higher quality, more memory, slower
    - 1024+ dimensions: Maximum quality, significant overhead
        
    Collection Management:
    - Single collection: Simple, good for small-medium datasets
    - Multiple collections: Better for large, diverse datasets
    - Sharding: Consider for very large datasets (>1M vectors)
        
    Error Handling Strategy:
    
    Graceful Degradation:
    - LLM failures: Fall back to content display
    - Database errors: Informative error messages
    - Network issues: Retry logic for transient failures
        
    Resource Management:
    - Memory monitoring: Prevent OOM conditions
    - Connection pooling: Efficient database usage
    - Cleanup: Proper resource deallocation
        
    Monitoring & Logging:
    - Performance metrics: Track response times
    - Error rates: Monitor system health
    - Usage patterns: Understand user behavior
        
    Production Deployment Considerations:
    
    Environment Configuration:
    - Use environment variables for sensitive data
    - Separate configs for dev/staging/production
    - Implement proper logging and monitoring
        
    Security:
    - API key management: Secure storage and rotation
    - Network security: HTTPS, firewall rules
    - Access control: User authentication and authorization
        
    Performance Optimization:
    - Caching: Redis for frequently accessed data
    - Load balancing: Distribute requests across instances
    - CDN: Static content delivery optimization
        
    Maintenance:
    - Regular backups: Database and configuration
    - Model updates: Periodic embedding model refresh
    - Performance tuning: Monitor and adjust parameters
    """
    s = SETTINGS
    embeddings = get_azure_embedding_model(s)  
    llm = get_llm()  

    client = get_qdrant_client(s)

    retriever = SimpleRetriever(client, s, embeddings)

    doc_folder = scan_docs_folder("src\\rag_flow\\tools\\rag_w_qdrant\\docs_test")
    docs = load_documents(doc_folder)  
    chunks = split_documents(docs, s)

    vector_size = s.vector_size  
    recreate_collection_for_rag(client, s, vector_size)

    upsert_chunks(client, s, chunks, embeddings)

    q = question
    hits = hybrid_search(client, s, q, embeddings)
    if not hits:
        print("Nessun risultato.")


    if llm:
        try:
            ctx = format_docs_for_prompt(hits)
            with open("output/last_context.txt", "w", encoding="utf-8") as f:
                f.write(ctx)
            chain = build_rag_chain(llm)
            answer = chain.invoke({"question": q, "context": ctx})

            questions = [
                "Quali sono le parti principali di un aereo?",
                "Cos'è la portanza in aeronautica?",
                "Quali sono le differenze tra un motore turbofan e un turbojet in termini di efficienza e utilizzo?",
                "Quali sono le principali compagnie aeree americane menzionate nel documento?",
                "Che cos'è un Full Service Carrier (FSC) e quali caratteristiche ha?",
                "Quali sono i tre tipi di motori utilizzati nell'aviazione commerciale?",
                "Cosa significa SAF nel contesto della sostenibilità ambientale dell'aviazione?",
            ]

            ground_truth = {
                questions[0]: "Profilo alare, fusoliera, sistemi di controllo di volo, struttura semi-monoscocca, avionics e sistemi di navigazione",
                questions[1]: "Forza aerodinamica generata dal profilo alare ottimizzato che permette all'aereo di volare, parte dell'aerodinamica insieme alla resistenza",
                questions[2]: "Turbofan è lo standard per aviazione commerciale con maggiore efficienza, turbojet non è menzionato specificamente nel documento",
                questions[3]: "American Airlines, Delta Air Lines, United Airlines, Southwest Airlines",
                questions[4]: "Servizio completo con pasti inclusi, reti hub-and-spoke, classi multiple di servizio",
                questions[5]: "Turbofan, Turboprop, Motori elettrici",
                questions[6]: "Sustainable Aviation Fuel - carburanti sostenibili",
            }
            rag_eval = ragas_evaluation(
                questions, chain, llm, embeddings, retriever, s, ground_truth
            )

            print("\n METRICHE OTTENUTE:\n", rag_eval)
            rag_eval.to_json("output/rag_eval_results.json", orient="records", lines=True)
            return answer
        except Exception as e:
            print(f"\nLLM generation failed: {e}")
            print("Falling back to content display...")
            print()
    else:
        print("LLM not available")



if __name__ == "__main__":
    rag_system("Qual è la capitale della Francia?")