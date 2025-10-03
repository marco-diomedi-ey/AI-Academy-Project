# RAG Flow - Technical Documentation

## 1. Obiettivo del Progetto

Il progetto implementa un sistema di Retrieval-Augmented Generation (RAG) specializzato per il dominio aeronautico, utilizzando l'architettura CrewAI Flow per orchestrare agenti specializzati e garantire risposte accurate e contestualizzate.

### Scopo Principale
- Fornire risposte tecniche accurate su argomenti aeronautici basate su documenti indicizzati
- Integrare ricerca locale (RAG) con ricerca web per copertura completa
- Garantire qualità e conformità etica attraverso validazione automatica
- Produrre documentazione strutturata in formato Markdown

### Dominio Target
Sistema specializzato per:
- Ingegneria aeronautica e progettazione aeromobili
- Operazioni e procedure di volo
- Industria aeronautica e modelli di business
- Normative e standard di certificazione
- Manutenzione e sicurezza aeronautica

## 2. Architettura del Sistema

### 2.1 Schema Architetturale

```
RAG Flow System
├── Flow Layer (CrewAI)
│   ├── Question Validation (Aeronautic + Ethical)
│   ├── RAG Processing (Local Knowledge)
│   ├── Web Research (External Sources)
│   ├── Document Synthesis (Content Integration)
│   ├── Bias Detection (Ethical Compliance)
│   └── Quality Evaluation (RAGAS)
│
├── Crew Layer (Specialized Agents)
│   ├── AeronauticRagCrew (RAG Expert)
│   ├── WebCrew (Web Analyst) 
│   ├── DocCrew (Document Generator)
│   └── BiasCrew (Bias Checker)
│
├── Tool Layer
│   ├── RAG System (Hybrid Search + MMR)
│   ├── Web Search (SerperDev API)
│   └── Quality Assessment (RAGAS)
│
├── Data Layer
│   ├── Vector Database (Qdrant)
│   ├── Document Processing (Multi-format)
│   └── Embedding Models (Sentence Transformers)
│
└── Evaluation Layer
    ├── RAGAS Metrics (Quality Assessment)
    ├── Bias Detection (Ethical Compliance)
    └── Performance Monitoring
```

### 2.2 Moduli Principali

#### CrewAI Flow Orchestration
- **Dual Router Validation**: Verifica dominio aeronautico + conformità etica
- **Sequential Processing**: Esecuzione ordinata con state management
- **Conditional Routing**: Logica di branching basata su validazione
- **Error Recovery**: Gestione robusta degli errori

#### RAG System
- **Hybrid Search**: Combinazione ricerca semantica + text matching
- **MMR Diversification**: Riduzione ridondanza tramite Maximal Marginal Relevance
- **Quality Filtering**: Pipeline multi-stage per validazione documenti
- **Trustability Assessment**: Sistema di classificazione affidabilità fonti

#### Evaluation Framework
- **RAGAS Integration**: Metriche automatiche (faithfulness, relevancy, precision, recall)
- **Bias Detection**: Analisi multi-dimensionale e redazione automatica
- **Performance Tracking**: Monitoraggio continuo qualità e latenza

### 2.3 Flusso Esecuzione

1. **Question Input**: Acquisizione input utente
2. **Dual Validation**: Controllo dominio aeronautico + etica
3. **RAG Analysis**: Recupero contesto da knowledge base locale
4. **Web Research**: Integrazione fonti esterne (opzionale)
5. **Content Synthesis**: Aggregazione e strutturazione informazioni
6. **Bias Check**: Rilevamento e mitigazione bias
7. **Quality Evaluation**: Valutazione RAGAS e metriche
8. **Output Generation**: Documento Markdown finale

## 3. Istruzioni di Installazione ed Esecuzione

### 3.1 Prerequisiti

**Requisiti Sistema:**
- Python 3.10-3.13
- 8GB RAM minimo (16GB raccomandato)
- 5GB spazio disco per modelli e indici

**Dipendenze Principali:**
- CrewAI Framework
- Qdrant Vector Database
- Azure OpenAI API
- Sentence Transformers
- RAGAS Evaluation

### 3.2 Installazione

```bash
# Clone repository
git clone <repository-url>
cd rag_flow

# Installazione con CrewAI (raccomandato)
pip install crewai
crewai install  # Crea venv e installa dipendenze

# Alternativa con UV
pip install uv
uv sync

# Alternativa con pip
pip install -r requirements.txt
```

### 3.3 Configurazione

**Environment Variables (.env):**
```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_API_VERSION=2024-12-01-preview
MODEL=azure/gpt-4o
SERPER_API_KEY=your-serper-key  # Opzionale per web search
```

**Struttura Configurazione:**
```
rag_flow/
├── .env                    # Variabili ambiente
├── pyproject.toml         # Dipendenze progetto
├── src/rag_flow/
│   ├── main.py           # Flow definition
│   └── crews/*/config/   # Configurazioni YAML agenti/task
└── docs_test/            # Documenti aeronautici per indicizzazione
```

### 3.4 Esecuzione

**Modalità Standard:**
```bash
# Attivazione environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Esecuzione interattiva
crewai run

# Esecuzione con Streamlit dalla root principale ./AI-Academy-Project
python -m streamlit run src/rag_flow/streamlit_main_app.py --theme.primaryColor="#0d47a1"

```

**Modalità Debug:**
```bash
crewai run --verbose    # Output dettagliato
crewai run --debug      # Trace completo
```

**Esecuzione Programmatica:**
```python
from rag_flow.main import AeronauticRagFlow

flow = AeronauticRagFlow()
result = flow.kickoff()
```

### 3.5 Output Files

```
output/
├── generated_document.md     # Documento aggregato
├── redacted_document.md      # Documento post bias-check
├── rag_eval_results.json    # Metriche RAGAS
├── last_context.txt         # Contesto RAG utilizzato
└── crewai_flow.html         # Visualizzazione flow
```

## 4. Scelte Progettuali e Trade-off

### 4.1 Architettura CrewAI vs Alternative

**Scelta: CrewAI Flow Framework**

**Vantaggi:**
- Orchestrazione dichiarativa con routing condizionale
- State management integrato tra agenti
- Configurazione esterna tramite YAML
- Monitoring e debugging avanzato

**Trade-off:**
- Overhead architetturale per use case semplici
- Curva di apprendimento per pattern Flow
- Dipendenza da framework specifico

**Alternative Considerate:**
- LangChain: Meno strutturato per workflow complessi
- Custom Pipeline: Maggior controllo ma più sviluppo

### 4.2 Hybrid Search vs Pure Semantic

**Scelta: Hybrid Search (Semantic + Text + MMR)**

**Configurazione:**
- Semantic weight: 0.7, Text weight: 0.3
- MMR lambda: 0.6 (bilanciamento relevance/diversity)
- Top semantic candidates: 30, Text candidates: 100

**Vantaggi:**
- Copertura terminologia tecnica esatta
- Diversificazione risultati tramite MMR
- Robustezza contro variazioni linguistiche

**Trade-off:**
- Complessità computazionale maggiore
- Tuning parametri richiesto
- Latency aumentata vs pure semantic

### 4.3 Qdrant vs FAISS

**Scelta: Qdrant Vector Database**

**Motivazioni:**
- Supporto hybrid search nativo
- Scalabilità e persistenza
- Metadata filtering avanzato
- Quantization integrata

**Trade-off:**
- Overhead deployment vs FAISS in-memory
- Configurazione più complessa
- Dipendenza servizio esterno

**Configurazione Ottimizzata:**
- HNSW index: m=32, ef_construct=256
- Scalar quantization INT8 per memory efficiency
- Cosine distance per embeddings normalizzati

### 4.4 Document Processing Strategy

**Scelta: Multi-format Processing con Quality Filtering**

**Pipeline:**
1. Format detection automatica
2. Quality assessment multi-stage
3. Trustability classification
4. Metadata enrichment

**Parametri Ottimizzati:**
- Chunk size: 700 caratteri (technical content)
- Overlap: 120 caratteri (context preservation)
- Quality threshold: 0.6 coherence, 0.3 relevance

**Trade-off:**
- Processing time aumentato
- Storage overhead per metadata
- Complessità configurazione vs accuratezza

### 4.5 Evaluation Framework

**Scelta: RAGAS + Bias Detection Integrati**

**Metriche RAGAS:**
- Faithfulness: >0.80 (anti-hallucination)
- Answer Relevancy: >0.75 
- Context Precision/Recall: >0.70/>0.65

**Quality Thresholds:**
- Bilanciamento precision vs recall
- Real-time evaluation vs batch processing
- Computational cost vs quality assurance

**Trade-off:**
- Latency aggiuntiva per evaluation
- Dipendenza ground truth per alcune metriche
- Costo computazionale vs qualità garantita

### 4.6 Embedding Model Selection

**Scelta: sentence-transformers/all-MiniLM-L6-v2**

**Caratteristiche:**
- 384 dimensioni (vs 768 alternative)
- Inference speed: ~3x faster di L12
- Quality: 90-95% di all-mpnet-base-v2

**Trade-off:**
- Speed vs absolute quality
- Memory efficiency vs semantic richness
- Deployment simplicity vs custom fine-tuning

### 4.7 LLM Integration

**Scelta: Azure OpenAI GPT-4o**

**Configurazione:**
- Temperature: 0 per validation (deterministic)
- Max retries: 2 per robustezza
- Context window optimization

**Motivazioni:**
- Accuracy superiore per technical content
- Azure integration enterprise-ready
- Consistent API vs multiple providers

**Trade-off:**
- Costo per token vs modelli open-source
- Vendor lock-in vs flexibility
- Latency cloud vs local inference

### 4.8 Error Handling Strategy

**Scelta: Graceful Degradation**

**Implementazione:**
- Retry automatici con exponential backoff
- Fallback web search se RAG fails
- Partial results vs complete failure
- User guidance per validation errors

**Trade-off:**
- User experience vs system complexity
- Recovery time vs failure transparency
- Resource usage vs reliability

### 4.9 Security e Compliance

**Scelta: Multi-layer Validation**

**Controlli Implementati:**
- Domain validation (aeronautic relevance)
- Ethical compliance checking
- Bias detection e mitigation
- Source trustability assessment

**Trade-off:**
- Security vs usability
- Processing overhead vs compliance
- False positive rate vs coverage

### 4.10 Guardrail e Content Safety

**Scelta: Multi-layer Guardrail System**

**Trustable Sources Control:**
```python
# Metadata trustability assessment in load_documents()
trustability = "trusted" 
doc.metadata["trustability"] = trustability

# RAG chain filtering untrusted content
"""SE dentro i metadati del documento è presente il valore 'untrusted' 
non prendere in considerazione il contenuto"
```

**Approved Domains Whitelist:**
```python
# TrustedWebSearch class con domini YAML-based
class TrustedWebSearch(BaseTool):
    def _load_trusted_domains(self):
        # Carica domini da crews/web_crew/config/domains.yaml
        with open('crews/web_crew/config/domains.yaml', 'r') as f:
            domains_config = yaml.safe_load(f)
            return domains_config.get('trusted_domains', [])
    
    # Filtering SerperDev results per domini approvati
    def _filter_by_trusted_domains(self, results):
        return [r for r in results if any(domain in r['link'] 
                for domain in self.trusted_domains)]
```

**Document Content Validation:**
```python
# PDF quality assessment in is_document_low_quality()
def is_pdf_low_quality(file_path: str, content: str) -> bool:
    doc = fitz.open(file_path)
    suspicious_chars = 0
    total_chars = 0
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        
        for char in text:
            total_chars += 1
            if char in '▯□■◆●◊':  # OCR artifacts detection
                suspicious_chars += 1
    
    # Quality threshold: <10% suspicious characters
    suspicion_ratio = suspicious_chars / max(total_chars, 1)
    return suspicion_ratio > 0.10
```

**Implementazione Guardrail:**
- **Source Layer**: `assess_source_trustability()` per pattern-based validation
- **Content Layer**: `is_document_low_quality()` per OCR artifact detection
- **Output Layer**: Prompt constraints nel RAG chain per untrusted filtering

**Trade-off:**
- Security robustezza vs false positive rate
- Processing latency vs content safety
- Whitelist maintenance vs coverage automatica

Questa architettura bilancia performance, accuracy e maintainability per un sistema RAG production-ready nel dominio aeronautico, con emphasis su quality assurance e ethical compliance.
