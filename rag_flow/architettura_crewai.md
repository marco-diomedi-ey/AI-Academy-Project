# Architettura CrewAI - RAG Flow

## Panoramica

Questo progetto implementa un'architettura **CrewAI Flow** avanzata che combina orchestrazione di flussi, agenti specializzati e strumenti personalizzati per creare un sistema di domande e risposte intelligente sull'aeronautica.

---

## CrewAI Flow Architecture

### Concetti Fondamentali

**CrewAI Flow** è un pattern di orchestrazione che permette di:
- Definire sequenze di operazioni complesse
- Gestire stati condivisi tra diverse fasi
- Implementare routing condizionale basato sui risultati
- Coordinare l'esecuzione di multiple Crew

### Pattern Flow

```python
class AeronauticRagFlow(Flow[AeronauticRagState]):
    @start()                    # Entry point del flow
    @listen(method)             # Listener per eventi
    @router(condition)          # Routing condizionale
```

---

## Struttura dell'Architettura

### Gerarchia Componenti

```
CrewAI Flow System
├── Flow Layer (Orchestrazione)
│   ├── State Management (Stato condiviso)
│   ├── Event Listeners (Reattività)
│   └── Conditional Routing (Logica di branching)
├── Crew Layer (Agenti specializzati)
│   ├── RAG Expert Crew
│   ├── Document Generation Crew
│   └── Web Research Crew
├── Tool Layer (Strumenti)
│   ├── RAG System Tool
│   ├── Web Search Tools
│   └── Content Validation Tools
└── Data Layer (Conoscenza)
    ├── Vector Stores (FAISS)
    ├── Document Collections
    └── Web Content Cache
```

---

## Flow Orchestration

### Definizione del Flow

```python
@dataclass
class AeronauticRagState(BaseModel):
    """Stato condiviso attraverso tutto il flow"""
    question_input: str = ""
    rag_result: str = ""
```

### Metodi del Flow

#### **1. Start Method**
```python
@start()
def starting_procedure(self):
    """
    Entry point - inizializza il sistema
    - Setup logging
    - Configurazione iniziale
    - State initialization
    """
    print("Starting the Aeronautic RAG Flow")
    return "ready"
```

#### **2. Listener Methods**
```python
@listen(starting_procedure)
def generate_question(self):
    """
    Ascolta il completion di starting_procedure
    - Input utente interattivo
    - Validazione formato domanda
    - State update
    """
    question = input("Enter your question about aeronautics:")
    self.state.question_input = question
    return question
```

#### **3. Router Methods**
```python
@router(generate_question)
def aeronautic_question_analysis(self):
    """
    Routing condizionale basato su analisi LLM
    - Validazione dominio aeronautico
    - Filtering domande off-topic
    - Routing verso pipeline specializzato
    """

@router("success")  
def ethic_question_analysis(self):
    """
    Ethical compliance validation
    - Verifica conformità standard etici
    - Controllo contenuto appropriato
    - Doppia validazione aeronautica + etica
    """
```

### Flow Execution Pattern

1. **Sequential Execution**: Ogni step attende il precedente
2. **State Persistence**: Lo stato è condiviso tra tutti i metodi
3. **Event-Driven**: Listeners reagiscono al completamento
4. **Conditional Branching**: Router permettono logica complessa

---

## Crew Architecture

### Crew Specialization Pattern

Il sistema implementa **3 crew specializzate**, ciascuna con agenti dedicati e responsabilità specifiche:

#### **1. AeronauticRagCrew** (`crews/rag_crew/`)
- **Agente**: `rag_expert`
- **Ruolo**: RAG Expert Agent specializzato in aeronautica
- **Responsabilità**: 
  - Utilizzare il sistema RAG per recuperare informazioni contestuali
  - Rispondere a domande aeronautiche basandosi su documenti indicizzati
  - Validare la presenza di informazioni nel contesto disponibile
- **Tool**: `rag_system` (FAISS + Azure OpenAI)
- **Output**: Risposte tecniche accurate basate su retrieval

#### **2. WebCrew** (`crews/web_crew/`)
- **Agente**: `web_analyst`
- **Ruolo**: Web Analyst Agent per ricerca e analisi web
- **Responsabilità**:
  - Eseguire ricerche web su topic aeronautici
  - Analizzare risultati di ricerca per rilevanza
  - Estrarre insights chiave da grandi volumi di dati web
  - Fornire summary concisi e strutturati
- **Tool**: `SerperDevTool` (Google Search API)
- **Output**: Summary analitici di contenuti web

#### **3. DocCrew** (`crews/doc_crew/`)
- **Agente**: `doc_redactor`
- **Ruolo**: Document Redactor Agent per generazione documenti
- **Responsabilità**:
  - Creare documenti Markdown strutturati
  - Integrare informazioni da RAG Expert e Web Analyst
  - Garantire chiarezza, coerenza e formattazione professionale
  - Produrre output finale user-ready
- **Tool**: Nessun tool esterno (processing interno)
- **Output**: Documenti Markdown formattati e completi

#### **4. BiasCrew** (`crews/bias_crew/`)
- **Agente**: `bias_checker`
- **Ruolo**: Bias Checker Agent per rilevamento e mitigazione bias
- **Responsabilità**:
  - Analizzare contenuti per potenziali bias di tono, accuratezza e rappresentazione
  - Identificare e categorizzare bias in dimensioni multiple
  - Eseguire redazione automatica di contenuti problematici
  - Garantire standard etici e bilanciamento nei documenti finali
- **Tool**: Nessun tool esterno (analisi LLM-based interna)
- **Output**: Documenti Markdown ripuliti da bias identificati

### Crew Structure

```python
# 1. AeronauticRagCrew - Specializzata in retrieval contestuale
@CrewBase
class AeronauticRagCrew:
    """Crew per elaborazione RAG aeronautica"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    @agent
    def rag_expert(self) -> Agent:
        """Agente esperto in sistemi RAG"""
        return Agent(
            config=self.agents_config["rag_expert"],
            tools=[rag_system],  # RAG tool integration
            verbose=True
        )
    
    @task
    def rag_response_task(self) -> Task:
        """Task di elaborazione RAG"""
        return Task(
            config=self.tasks_config["rag_response_task"],
            agent=self.rag_expert,
        )

# 2. WebCrew - Specializzata in ricerca web
@CrewBase  
class WebCrew:
    """Crew per ricerca e analisi web"""
    
    @agent
    def web_analyst(self) -> Agent:
        """Agente analista web"""
        return Agent(
            config=self.agents_config["web_analyst"],
            tools=[SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))],
            verbose=True
        )
    
    @task
    def web_analysis_task(self) -> Task:
        """Task di analisi web"""
        return Task(
            config=self.tasks_config["web_analysis_task"],
            agent=self.web_analyst,
        )

# 3. DocCrew - Specializzata in generazione documenti
@CrewBase
class DocCrew:
    """Crew per generazione documenti strutturati"""
    
    @agent
    def doc_redactor(self) -> Agent:
        """Agente redattore documenti"""
        return Agent(
            config=self.agents_config["doc_redactor"],
            verbose=True
        )
    
    @task
    def document_creation_task(self) -> Task:
        """Task di creazione documenti"""
        return Task(
            config=self.tasks_config["document_creation_task"],
            agent=self.doc_redactor,
            output_file="output/generated_document.md"
        )
    
    @crew
    def crew(self) -> Crew:
        """Assembly della crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
```

### Agent Configuration (YAML)

```yaml
# crews/rag_crew/config/agents.yaml
rag_expert:
  role: >
    RAG Expert Agent specializzato in aeronautica
  goal: >
    Use the RAG system to answer {question} based on provided context.
    MUST use RAG system to retrieve relevant information and generate accurate response.
  backstory: >
    Specialized agent that uses Retrieval-Augmented Generation (RAG) system
    to provide answers based on given context. Always uses RAG system instead of
    answering questions without context.
  llm: azure/gpt-4o

# crews/web_crew/config/agents.yaml  
web_analyst:
  role: >
    Web Analyst Agent per ricerca e analisi web
  goal: >
    Analyze web search results and extract relevant information about {question}.
    Summarize key points and insights from web search results.
    Check relevance to {question} before including in summary.
  backstory: >
    Specialized agent that analyzes web search results to extract relevant information.
    Keen eye for detail, quickly identifies important insights from large data volumes.
    Always provides clear and concise summaries.
  llm: azure/gpt-4o

# crews/doc_crew/config/agents.yaml
doc_redactor:
  role: >
    Document Redactor Agent
  goal: >
    Generate document in .md format about {paper} provided by rag_expert and web_analyst.
  backstory: >
    Specialized agent that creates well-structured markdown documents
    based on {paper} from rag_expert and web_analyst agents.
    Always formats in markdown ensuring clarity and coherence.
  llm: azure/gpt-4o
```

### Task Configuration (YAML)

```yaml
# crews/rag_crew/config/tasks.yaml
rag_response_task:
  description: >
    Use the RAG system to answer the {question} based on the provided context.
    You MUST use the RAG system to retrieve relevant information and generate
    accurate answers.
  expected_output: >
    A comprehensive and accurate answer to the question based on retrieved
    context from the RAG system, formatted in clear and professional language.
  agent: rag_expert

# crews/web_crew/config/tasks.yaml
web_analysis_task:
  description: >
    Search the web for information related to {question} and analyze the results.
    Extract the most relevant information and provide a structured summary.
  expected_output: >
    A structured summary of web search results with key insights and relevant
    information about {question}, properly formatted and organized.
  agent: web_analyst

# crews/doc_crew/config/tasks.yaml  
document_creation_task:
  description: >
    Create a comprehensive markdown document about {paper} combining information
    from RAG expert and web analyst findings.
  expected_output: >
    A well-structured markdown document that integrates information from both
    RAG system and web research, formatted professionally with clear sections.
  agent: doc_redactor
  output_file: "output/generated_document.md"
```

### Crew Integration nel Flow

```python
@listen("success")
def rag_analysis(self):
    """Integrazione Crew nel Flow"""
    result = (
        AeronauticRagCrew()                    # Istanziazione Crew
        .crew()                               # Creazione crew assembly
        .kickoff(inputs={                     # Esecuzione con input
            "question": self.state.question_input,
            "response": self.state.rag_result
        })
    )
    self.state.rag_result = result.raw        # State update
```

---

## Tool System Architecture

### Tool Definition Pattern

I **Tool** in CrewAI sono funzioni Python decorate che possono essere:
- Assegnate agli agenti
- Utilizzate per interagire con sistemi esterni  
- Composte per creare funzionalità complesse

### Tool Structure

```python
@tool('rag_system')
def rag_system(question: str) -> str:
    """
    Tool principale per il sistema RAG
    
    Capabilities:
    - Document loading and indexing
    - Vector similarity search
    - Web search integration  
    - Content validation
    - Response generation
    """
    # Tool implementation
    return processed_answer
```

### Tool Integration

#### **1. Agent-Tool Assignment**
```python
@agent  
def rag_expert_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["rag_expert_agent"],
        tools=[rag_system, web_search_tool],  # Multiple tools
        tool_sharing=True                      # Tool sharing tra agenti
    )
```

#### **2. Tool Execution Flow**
```
Agent riceve Task → Analizza requirements → Seleziona Tool → Esegue Tool → Processa risultato
```

#### **3. Tool Chaining**
I tool possono essere concatenati per operazioni complesse:
```python
# Esempio di tool chaining nel RAG system
def rag_system(question: str) -> str:
    docs = load_documents()           # Document loading
    embeddings = create_embeddings()  # Vectorization  
    results = search_similar()        # Similarity search
    web_content = web_search()        # Web enhancement
    answer = generate_response()      # LLM generation
    return answer
```

---

## Flow Execution Patterns

### Event-Driven Architecture

#### **1. Linear Flow**
```python
@start() → @listen(start) → @listen(step2) → @listen(step3)
```

#### **2. Conditional Flow** 
```python
@start() → @listen(start) → @router("condition") → [@listen("success"), @listen("failure")]
```

#### **3. Parallel Processing**
```python
@start() → [@listen(start), @listen(start)] → @listen([parallel_task1, parallel_task2])
```

### State Management

```python
class AeronauticRagState(BaseModel):
    """Stato globale del Flow"""
    question_input: str = ""          # Input utente
    rag_result: str = ""             # Risultato elaborazione
    
    # State è accessibile da tutti i metodi del Flow
    def any_flow_method(self):
        self.state.question_input = "new value"  # State update
        current_value = self.state.rag_result    # State read
```

### Flow Communication Patterns

#### **1. Method Return Values**
```python
@start()
def step1(self):
    return "success"  # Ritorna valore per routing

@router("success") 
def step2(self):     # Riceve solo se step1 ritorna "success"
    pass
```

#### **2. State Sharing**
```python
@start()
def step1(self):
    self.state.data = "shared_value"  # Salva nello state

@listen(step1)
def step2(self):
    value = self.state.data          # Legge dallo state
```

---

## Esecuzione e Deployment

### Setup Ambiente

#### **1. Installazione Dipendenze**
```bash
# Clona il repository
git clone <repository-url>
cd rag_flow

# Installa dipendenze con uv
uv sync

# Oppure con pip
pip install -r requirements.txt
```

#### **2. Configurazione Environment**
```bash
# Crea file .env
cp .env.example .env

# Configura le variabili:
AZURE_OPENAI_ENDPOINT=https://your-endpoint.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_API_VERSION=2024-12-01-preview
MODEL=azure/gpt-4o
SERPER_API_KEY=your-serper-key
```

#### **3. Struttura Configurazione**
```
rag_flow/
├── .env                    # Variabili ambiente
├── pyproject.toml         # Dipendenze progetto
└── src/rag_flow/
    ├── main.py           # Flow definition
    └── crews/            # Crew configurations
        └── */config/
            ├── agents.yaml
            └── tasks.yaml
```

### Comandi di Esecuzione

#### **1. Esecuzione Interattiva**
```bash
# Attiva environment virtuale
source .venv/bin/activate  # Linux/Mac
# oppure
.venv\Scripts\activate     # Windows

# Esegui il flow
crewai run

# Output interattivo:
# Starting the Aeronautic RAG Flow
# Enter your question about aeronautics: [USER INPUT]
# Analyzing question...
# [FLOW EXECUTION]
```

#### **2. Esecuzione con Debugging**
```bash
# Modalità verbose per debugging
crewai run --verbose

# Trace completo dell'esecuzione
crewai run --debug
```

#### **3. Esecuzione Programmatica**
```python
# In Python script
from rag_flow.main import AeronauticRagFlow

# Istanzia e esegui flow
flow = AeronauticRagFlow()
result = flow.kickoff()
print(result)
```

### Flow Monitoring

#### **1. Logging Structure**
```
Flow: AeronauticRagFlow
ID: [unique-flow-id]
├── Completed: starting_procedure
├── Completed: generate_question  
├── Completed: question_analysis
└── Running: rag_analysis
    └── Crew: AeronauticRagCrew
        └── Task: rag_response_task
            └── Agent: RAG Expert Agent
                └── Tool: rag_system
```

#### **2. Error Handling**
```python
# Flow con error handling
try:
    result = flow.kickoff()
except FlowExecutionError as e:
    print(f"Flow failed at step: {e.step}")
    print(f"Error: {e.message}")
except CrewExecutionError as e:
    print(f"Crew failed: {e.crew_name}")
    print(f"Agent: {e.agent_name}")
```

### Performance Optimization

#### **1. Parallel Execution**
```python
# Per crew indipendenti
@listen(start_step)
async def parallel_crew_1(self): ...

@listen(start_step)  
async def parallel_crew_2(self): ...
```

#### **2. Caching**
```python
# Cache per risultati costosi
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_rag_operation(query: str): ...
```

#### **3. Resource Management**
```python
# Configurazione limiti
crew = Crew(
    agents=agents,
    tasks=tasks,
    max_execution_time=300,    # 5 minuti timeout
    max_retry=3                # Retry automatici
)
```

---

## Best Practices

### Flow Design

1. **Single Responsibility**: Ogni step del flow ha una responsabilità specifica
2. **State Minimization**: Mantieni lo state essenziale e clean
3. **Error Boundaries**: Implementa graceful degradation
4. **Monitoring**: Logging strutturato per debugging

### Crew Organization

1. **Domain Separation**: Una crew per dominio/funzionalità
2. **Agent Specialization**: Agenti specializzati con tool specifici
3. **Configuration Externalization**: YAML per flessibilità
4. **Resource Isolation**: Evita condivisione non necessaria

### Tool Development

1. **Single Purpose**: Un tool = una funzionalità
2. **Type Safety**: Annotazioni di tipo per parametri/return
3. **Error Handling**: Gestione robusta degli errori
4. **Documentation**: Docstring dettagliate per gli agenti

---

## RAG System Integration

Il sistema RAG è integrato come **tool specializzato** che fornisce:

- **Document Retrieval**: FAISS vector search per documenti locali
- **Web Enhancement**: DuckDuckGo search per contenuti aggiornati  
- **Content Validation**: LLM-based quality assessment
- **Response Generation**: Context-aware answer synthesis

Questa integrazione permette agli agenti CrewAI di accedere a conoscenza estesa mantenendo l'architettura modulare e scalabile.

---

## Sistema di Valutazione e Qualità

### RAGAS Integration

Il sistema integra **RAGAS** (Retrieval-Augmented Generation Assessment) per la valutazione automatica della qualità delle risposte RAG.

#### **Architettura RAGAS**

```python
# Metrics Pipeline Integration
from ragas.metrics import (
    faithfulness,           # Fedeltà al contesto recuperato
    answer_correctness,     # Correttezza vs ground truth
    AnswerRelevancy,        # Rilevanza della risposta
    context_precision,      # Precisione del contesto
    context_recall          # Recall del contesto
)

def ragas_evaluation(questions, chain, llm, embeddings, retriever):
    """Valutazione automatica qualità RAG"""
    dataset = build_ragas_dataset(questions, retriever, chain, k)
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    
    metrics = [
        context_precision,      # Precisione contesto
        context_recall,         # Recall contesto  
        faithfulness,           # Fedeltà contenuto
        AnswerRelevancy(strictness=1), # Rilevanza risposta
    ]
    
    # Aggiunge answer_correctness se ground truth disponibile
    if ground_truth_available:
        metrics.append(answer_correctness)
    
    return evaluate(dataset=evaluation_dataset, metrics=metrics)
```

#### **Metriche di Qualità**

##### **1. Faithfulness (Fedeltà)**
- **Scopo**: Misura quanto la risposta è fedele al contesto recuperato
- **Range**: 0.0 - 1.0 (1.0 = completamente fedele)
- **Calcolo**: Verifica che ogni affermazione nella risposta sia supportata dal contesto
- **Anti-hallucination**: Rileva allucinazioni e contenuti inventati

##### **2. Answer Relevancy (Rilevanza)**
- **Scopo**: Valuta la rilevanza della risposta rispetto alla domanda
- **Range**: 0.0 - 1.0 (1.0 = completamente rilevante)  
- **Strictness**: Configurabile (1 = strict, richiede piena rilevanza)
- **Calcolo**: Analizza se la risposta risponde effettivamente alla domanda

##### **3. Context Precision (Precisione Contesto)**
- **Scopo**: Misura la precisione dei chunk di contesto recuperati
- **Range**: 0.0 - 1.0 (1.0 = tutti i chunk sono rilevanti)
- **Calcolo**: Frazione di chunk recuperati che sono effettivamente rilevanti
- **Impatto**: Ottimizza la strategia di retrieval

##### **4. Context Recall (Recall Contesto)**
- **Scopo**: Misura se tutto il contesto necessario è stato recuperato
- **Range**: 0.0 - 1.0 (1.0 = tutto il contesto necessario recuperato)
- **Calcolo**: Frazione del contesto ground-truth presente nei chunk recuperati
- **Dipendenza**: Richiede ground truth per essere calcolato

##### **5. Answer Correctness (Correttezza)**
- **Scopo**: Confronta la risposta con una risposta di riferimento (ground truth)
- **Range**: 0.0 - 1.0 (1.0 = risposta perfettamente corretta)
- **Componenti**: Combina aspetti semantici e fattuali
- **Opzionale**: Attivato solo se ground truth disponibile

#### **Pipeline di Valutazione**

```python
# Integrazione nel RAG System Tool
@tool('rag_system')
def rag_system(question: str) -> str:
    """RAG system con valutazione RAGAS integrata"""
    
    # 1. Esecuzione RAG standard
    context = retriever.get_relevant_documents(question)
    answer = chain.invoke({"question": question, "context": context})
    
    # 2. Valutazione RAGAS automatica
    try:
        rag_eval = ragas_evaluation(
            questions=[question], 
            chain=chain, 
            llm=llm, 
            embeddings=embeddings, 
            retriever=retriever
        )
        
        # 3. Logging metriche
        print("\nMETRICHE RAGAS:")
        print(f"Faithfulness: {rag_eval['faithfulness']:.3f}")
        print(f"Answer Relevancy: {rag_eval['answer_relevancy']:.3f}")
        print(f"Context Precision: {rag_eval['context_precision']:.3f}")
        print(f"Context Recall: {rag_eval['context_recall']:.3f}")
        
        # 4. Salvataggio risultati  
        rag_eval.to_json("output/rag_eval_results.json")
        
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
    
    return answer
```

#### **Dataset Construction**

```python
def build_ragas_dataset(questions, retriever, chain, k, ground_truth=None):
    """Costruisce dataset per valutazione RAGAS"""
    dataset = []
    
    for question in questions:
        # 1. Recupero contesto
        contexts_with_metadata = get_contexts_for_question(retriever, question, k)
        
        # 2. Generazione risposta
        formatted_context = format_contexts_for_chain(contexts_with_metadata)
        answer = chain.invoke({"question": question, "context": formatted_context})
        
        # 3. Preparazione entry dataset
        contexts_for_ragas = [ctx['content'] for ctx in contexts_with_metadata]
        
        entry = {
            "user_input": question,
            "retrieved_contexts": contexts_for_ragas,
            "response": answer,
        }
        
        # 4. Ground truth opzionale
        if ground_truth and question in ground_truth:
            entry["reference"] = ground_truth[question]
            
        dataset.append(entry)
    
    return dataset
```

#### **Output e Monitoraggio**

```json
// output/rag_eval_results.json
{
  "user_input": "Quali sono i fattori chiave per la progettazione aerodinamica?",
  "response": "I fattori chiave includono profilo alare, resistenza...",
  "faithfulness": 0.95,
  "answer_relevancy": 0.92,
  "context_precision": 0.88,
  "context_recall": 0.90,
  "answer_correctness": 0.85
}
```

#### **Quality Thresholds**

```python
# Soglie di qualità raccomandate
QUALITY_THRESHOLDS = {
    "faithfulness": 0.80,      # 80%+ fedeltà al contesto
    "answer_relevancy": 0.75,   # 75%+ rilevanza
    "context_precision": 0.70,  # 70%+ precisione contesto
    "context_recall": 0.65,     # 65%+ recall contesto
    "answer_correctness": 0.70   # 70%+ correttezza (se GT disponibile)
}

def validate_quality(metrics):
    """Validazione automatica qualità risposta"""
    warnings = []
    for metric, threshold in QUALITY_THRESHOLDS.items():
        if metric in metrics and metrics[metric] < threshold:
            warnings.append(f"WARNING {metric}: {metrics[metric]:.3f} < {threshold}")
    return warnings
```

### Continuous Quality Monitoring

#### **1. Real-time Evaluation**
- Valutazione automatica ad ogni esecuzione RAG
- Logging strutturato delle metriche
- Alert automatici per qualità sotto soglia

#### **2. Historical Analysis**
- Tracking trend qualità nel tempo
- Identificazione pattern di degradazione
- Analisi performance per tipologia domande

#### **3. Quality-based Routing**
- Routing condizionale basato su metriche qualità
- Fallback a web search per basse metriche RAG
- Retry automatici per risposte sotto soglia

---

## Bias Detection e Ethical AI Architecture

### BiasCrew Architecture

Il sistema integra una **crew specializzata** per il rilevamento e la mitigazione automatica di bias nei contenuti generati, garantendo standard etici e accuratezza nelle documentazioni aeronautiche.

#### **Architettura BiasCrew**

```python
@CrewBase
class BiasCrew():
    """
    CrewAI crew per controllo bias e redazione contenuti.
    
    Specializzata nell'identificare e mitigare potenziali bias in documenti
    generati, garantendo standard etici e accuratezza dei contenuti attraverso
    processi automatizzati di analisi e redazione.
    """
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    @agent
    def bias_checker(self) -> Agent:
        """Agente specializzato nel rilevamento bias"""
        return Agent(
            config=self.agents_config['bias_checker'],
            verbose=True
        )
    
    @task
    def bias_check_task(self) -> Task:
        """Task di controllo bias con output markdown"""
        return Task(
            config=self.tasks_config['bias_check_task'],
            output_file="output/redacted_document.md",
            verbose=True
        )
```

#### **Agent Configuration**

```yaml
# crews/bias_crew/config/agents.yaml
bias_checker:
  role: >
    Bias Detection and Content Redaction Specialist
  goal: >
    Analyze {document} for potential biases and create a clean, balanced version.
    Identify biases in tone, accuracy, representation and ethical considerations.
  backstory: >
    Specialized agent with expertise in identifying various forms of bias
    including confirmation bias, selection bias, representation bias, and
    cultural biases. Skilled in content redaction while preserving information
    integrity and maintaining professional document structure.
  llm: azure/gpt-4o
```

```yaml
# crews/bias_crew/config/tasks.yaml  
bias_check_task:
  description: >
    Analyze the provided {document} for potential biases and create a redacted version.
    
    Bias Analysis Scope:
    - Confirmation bias: One-sided presentation of information
    - Selection bias: Cherry-picking data or examples
    - Representation bias: Unfair representation of groups or concepts
    - Cultural bias: Cultural assumptions or stereotypes
    - Temporal bias: Outdated information presented as current
    - Source bias: Over-reliance on specific sources
    
    Redaction Process:
    - Identify and flag problematic content
    - Provide balanced alternatives
    - Maintain technical accuracy
    - Preserve document structure and formatting
    - Ensure professional tone throughout
    
  expected_output: >
    A clean, bias-free markdown document that maintains all technical content
    while addressing identified biases. The output should be professional,
    balanced, and ethically sound with clear structure and formatting.
  agent: bias_checker
  output_file: "output/redacted_document.md"
```

### Bias Detection Pipeline

#### **1. Multi-dimensional Bias Analysis**

```python
# Categorie di Bias Analizzate
BIAS_CATEGORIES = {
    "confirmation_bias": {
        "description": "Presentazione unilaterale delle informazioni",
        "detection": "Analisi di bilanciamento prospettive",
        "mitigation": "Aggiunta di viewpoint alternativi"
    },
    "selection_bias": {
        "description": "Cherry-picking di dati o esempi",
        "detection": "Verifica rappresentatività campioni",
        "mitigation": "Inclusione di dati bilanciati"
    },
    "representation_bias": {
        "description": "Rappresentazione ingiusta di gruppi o concetti",
        "detection": "Analisi linguaggio e terminologia",
        "mitigation": "Linguaggio neutro e inclusivo"
    },
    "cultural_bias": {
        "description": "Assunzioni culturali o stereotipi",
        "detection": "Verifica sensibilità culturale",
        "mitigation": "Terminologia culturalmente neutra"
    },
    "temporal_bias": {
        "description": "Informazioni obsolete presentate come attuali",
        "detection": "Controllo datazione informazioni",
        "mitigation": "Aggiornamento o qualificazione temporale"
    },
    "source_bias": {
        "description": "Eccessiva dipendenza da fonti specifiche",
        "detection": "Analisi diversità fonti",
        "mitigation": "Diversificazione referenze"
    }
}
```

#### **2. Processo di Redazione**

```python
# Flow Integration - Bias Check Stage
@listen(aggregate_results)
def bias_check(self, payload: Dict[str, Any]):
    """
    Esegue controllo bias sul documento generato.
    
    Pipeline di Bias Detection:
    1. Document Analysis: Analisi multi-dimensionale del contenuto
    2. Bias Identification: Rilevamento bias per categoria
    3. Content Redaction: Mitigazione automatica bias identificati
    4. Quality Validation: Verifica integrità contenuto post-redazione
    5. Markdown Output: Generazione documento ripulito
    """
    bias_crew = BiasCrew().crew()
    
    result = bias_crew.kickoff(inputs={
        "document": self.state.document,
    })
    
    # State update con documento bias-free
    self.state.final_doc = result.raw
    
    # Payload enrichment per monitoring
    payload.update({
        'bias_context': self.state.document,      # Documento originale
        'bias_result': result.raw,                # Documento redatto
        'bias_crew': bias_crew                    # Crew execution metadata
    })
    
    return payload
```

#### **3. Redaction Features**

##### **Content Preservation**
- **Technical Accuracy**: Mantenimento precisione tecnica
- **Information Integrity**: Preservazione contenuto informativo essenziale
- **Professional Structure**: Conservazione struttura e formattazione
- **Citation Maintenance**: Preservazione riferimenti e citazioni

##### **Bias Mitigation Strategies**
- **Balanced Perspectives**: Aggiunta di viewpoint alternativi
- **Neutral Language**: Sostituzione terminologia biased
- **Source Diversification**: Bilanciamento referenze
- **Cultural Sensitivity**: Adattamento per sensibilità culturale

##### **Quality Assurance**
- **Consistency Checks**: Verifica coerenza contenuto
- **Completeness Validation**: Controllo completezza informazioni
- **Tone Analysis**: Analisi tono professionale
- **Format Compliance**: Conformità standard markdown

### Ethical Standards Framework

#### **1. Ethical Guidelines**

```yaml
# Ethical Standards per Aeronautic Documentation
ethical_framework:
  accuracy:
    - "Informazioni tecniche verificabili"
    - "Citazioni accurate e complete"
    - "Distinzione tra fatti e opinioni"
  
  balance:
    - "Presentazione multi-prospettica"
    - "Riconoscimento limitazioni"
    - "Inclusione viewpoint alternativi"
  
  inclusivity:
    - "Linguaggio gender-neutral"
    - "Terminologia culturalmente sensibile" 
    - "Rappresentazione equa"
  
  transparency:
    - "Fonti chiaramente identificate"
    - "Metodologie esplicitate"
    - "Limitazioni riconosciute"
```

#### **2. Quality Metrics per Bias Detection**

```python
# Metriche di Valutazione Bias Detection
BIAS_METRICS = {
    "bias_detection_rate": "Percentuale bias identificati correttamente",
    "false_positive_rate": "Bias segnalati erroneamente",
    "content_preservation": "Percentuale contenuto preservato post-redazione",
    "readability_score": "Leggibilità documento finale",
    "professional_tone": "Mantenimento tono professionale",
    "technical_accuracy": "Accuratezza tecnica post-redazione"
}

def evaluate_bias_check_quality(original_doc, redacted_doc):
    """Valutazione qualità processo bias checking"""
    return {
        "content_preservation": calculate_content_overlap(original_doc, redacted_doc),
        "bias_reduction": measure_bias_indicators(original_doc, redacted_doc),
        "readability": assess_readability_score(redacted_doc),
        "professional_tone": analyze_tone_consistency(redacted_doc)
    }
```

#### **3. Integration nel Flow**

```python
# Sequenza Completa con Bias Check
Flow Sequence:
├── 1. Question Analysis (Aeronautic + Ethical validation)
├── 2. RAG Analysis (Context retrieval + RAGAS evaluation)  
├── 3. Web Analysis (External research + validation)
├── 4. Document Aggregation (Content synthesis)
├── 5. Bias Check (Bias detection + redaction)      # ← NEW
└── 6. Final Output (Clean, bias-free documentation)

# Output Files
output/
├── generated_document.md        # Documento aggregato originale
├── redacted_document.md         # Documento ripulito da bias
├── rag_eval_results.json       # Metriche RAGAS
└── bias_check_report.md         # Report bias detection
```

### Monitoring e Compliance

#### **1. Bias Detection Reporting**
- **Bias Categories Found**: Categorizzazione bias identificati
- **Redaction Summary**: Sommario modifiche effettuate
- **Quality Metrics**: Metriche qualità processo redazione
- **Compliance Check**: Verifica aderenza standard etici

#### **2. Continuous Improvement**
- **Bias Pattern Analysis**: Analisi pattern bias ricorrenti
- **Detection Model Tuning**: Ottimizzazione rilevamento
- **Redaction Quality**: Miglioramento qualità redazione
- **Ethical Guidelines Update**: Aggiornamento linee guida

---

## Advanced Document Processing Architecture

### Multi-modal Document Processing Pipeline

Il sistema implementa un **pipeline avanzato** per il processing intelligente di documenti multi-formato con quality filtering, trustability assessment e metadata enrichment.

#### **Architettura Document Processing**

```python
# Document Processing Pipeline
from langchain_community.document_loaders import (
    CSVLoader,                    # CSV data files
    PyMuPDFLoader,               # PDF documents  
    UnstructuredImageLoader,      # 🖼️ Image files
    UnstructuredMarkdownLoader,   # Markdown files
    TextLoader                   # 📃 Plain text files
)

def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Multi-format document loading con quality filtering integrato.
    
    Supported Formats:
    - PDF: Technical documents, reports, manuals
    - Markdown: Documentation, guides, specifications  
    - CSV: Data tables, metrics, structured info
    - Images: Diagrams, charts, technical drawings
    - Text: Legacy documents, notes, transcripts
    """
    all_documents = []
    
    for file_path in file_paths:
        try:
            # 1. Format Detection & Loader Selection
            docs = select_loader_by_format(file_path)
            
            # 2. Quality Assessment Pipeline
            valid_docs = quality_filter_pipeline(file_path, docs)
            
            # 3. Metadata Enrichment
            enriched_docs = enrich_document_metadata(file_path, valid_docs)
            
            all_documents.extend(enriched_docs)
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue
    
    return all_documents
```

#### **1. Format-Specific Processing**

##### **PDF Processing**
```python
def process_pdf_document(file_path: str) -> List[Document]:
    """
    Advanced PDF processing con OCR quality assessment.
    
    Quality Checks:
    - Text extraction success rate
    - Character recognition accuracy  
    - Suspicious character ratio analysis
    - Page content distribution
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # PDF-specific quality assessment
    for doc in docs:
        if is_pdf_low_quality(file_path, doc.page_content):
            print(f"Low quality PDF detected: {file_path}")
            continue
    
    return docs

def is_pdf_low_quality(file_path: str, content: str) -> bool:
    """
    PDF quality assessment basato su analisi contenuto.
    
    Quality Indicators:
    - Character count thresholds
    - Suspicious character ratio (OCR artifacts)
    - Content distribution analysis
    - Text coherence verification
    """
    # 1. Content Length Check
    if len(content.strip()) < 50:
        return True
        
    # 2. OCR Quality Analysis
    try:
        doc = fitz.open(file_path)
        suspicious_chars = 0
        total_chars = 0
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            # Character analysis per OCR quality
            for char in text:
                total_chars += 1
                if char in '▯□■◆●◊':  # OCR artifacts
                    suspicious_chars += 1
        
        # Quality threshold: <10% suspicious characters
        suspicion_ratio = suspicious_chars / max(total_chars, 1)
        return suspicion_ratio > 0.10
        
    except Exception:
        return True
```

##### **Markdown Processing**
```python
def process_markdown_document(file_path: str) -> List[Document]:
    """
    Markdown processing con structure validation.
    
    Features:
    - Header hierarchy validation
    - Link integrity checking
    - Code block preservation
    - Table structure validation
    """
    loader = UnstructuredMarkdownLoader(file_path)
    docs = loader.load()
    
    # Markdown-specific enrichment
    for doc in docs:
        doc.metadata.update({
            "headers_count": count_markdown_headers(doc.page_content),
            "links_count": count_markdown_links(doc.page_content),
            "code_blocks": extract_code_blocks(doc.page_content)
        })
    
    return docs
```

##### **Image Processing**
```python
def process_image_document(file_path: str) -> List[Document]:
    """
    Image processing con OCR e content extraction.
    
    Capabilities:
    - OCR text extraction
    - Technical diagram recognition
    - Chart/graph data extraction
    - Image metadata preservation
    """
    loader = UnstructuredImageLoader(file_path)
    docs = loader.load()
    
    # Image-specific metadata
    for doc in docs:
        doc.metadata.update({
            "image_type": get_image_format(file_path),
            "ocr_confidence": assess_ocr_quality(doc.page_content),
            "content_type": classify_image_content(file_path)
        })
    
    return docs
```

#### **2. Quality Filtering System**

```python
def quality_filter_pipeline(file_path: str, docs: List[Document]) -> List[Document]:
    """
    Multi-stage quality filtering pipeline.
    
    Filtering Stages:
    1. Content Length Validation
    2. Format-Specific Quality Checks  
    3. Content Coherence Analysis
    4. Technical Relevance Assessment
    """
    valid_docs = []
    
    for doc in docs:
        # Stage 1: Basic content validation
        if len(doc.page_content.strip()) < 50:
            print(f"🚫 Content too short: {file_path}")
            continue
            
        # Stage 2: Format-specific checks
        if not format_specific_quality_check(file_path, doc.page_content):
            print(f"🚫 Format quality failed: {file_path}")
            continue
            
        # Stage 3: Content coherence
        coherence_score = assess_content_coherence(doc.page_content)
        if coherence_score < 0.6:
            print(f"🚫 Low coherence ({coherence_score:.2f}): {file_path}")
            continue
            
        # Stage 4: Domain relevance  
        relevance_score = assess_aeronautic_relevance(doc.page_content)
        if relevance_score < 0.3:
            print(f"🚫 Low relevance ({relevance_score:.2f}): {file_path}")
            continue
            
        print(f"Quality passed: {file_path}")
        valid_docs.append(doc)
    
    return valid_docs

# Quality Assessment Functions
def assess_content_coherence(content: str) -> float:
    """Analizza coerenza contenuto usando NLP metrics"""
    # Implementazione analisi coerenza testuale
    return calculate_text_coherence_score(content)

def assess_aeronautic_relevance(content: str) -> float:
    """Valuta rilevanza aeronautica del contenuto"""
    aeronautic_keywords = {
        'aircraft', 'aviation', 'flight', 'aerodynamics', 'engine', 
        'wing', 'runway', 'pilot', 'airline', 'aerospace', 'turbine'
    }
    
    content_lower = content.lower()
    keyword_matches = sum(1 for keyword in aeronautic_keywords 
                         if keyword in content_lower)
    
    return min(keyword_matches / len(aeronautic_keywords), 1.0)
```

#### **3. Trustability Metadata System**

```python
def enrich_document_metadata(file_path: str, docs: List[Document]) -> List[Document]:
    """
    Document metadata enrichment con trustability assessment.
    
    Metadata Categories:
    - Source Trustability (trusted/untrusted)
    - Content Type Classification
    - Processing Quality Metrics
    - Technical Indicators
    """
    filename = Path(file_path).name
    
    for doc in docs:
        # Core Metadata
        doc.metadata.update({
            # Identification
            "filename": filename,
            "file_path": file_path,
            "processing_timestamp": datetime.now().isoformat(),
            
            # Trustability Assessment  
            "trustability": assess_source_trustability(file_path),
            
            # Content Classification
            "content_type": classify_document_type(doc.page_content),
            "technical_level": assess_technical_complexity(doc.page_content),
            
            # Quality Metrics
            "content_length": len(doc.page_content),
            "readability_score": calculate_readability_score(doc.page_content),
            "quality_score": calculate_overall_quality_score(doc),
            
            # Processing Info
            "extraction_method": get_extraction_method(file_path),
            "processing_status": "success"
        })
    
    return docs

def assess_source_trustability(file_path: str) -> str:
    """
    Source trustability assessment basato su euristiche.
    
    Trustability Factors:
    - Source reputation (known authoritative sources)
    - File naming patterns (official document patterns)
    - Content structure indicators
    - Metadata consistency
    """
    filename = Path(file_path).name.lower()
    
    # High Trust Indicators
    trusted_patterns = [
        'official', 'specification', 'standard', 'regulation',
        'manual', 'handbook', 'guide', 'documentation'
    ]
    
    # Low Trust Indicators  
    untrusted_patterns = [
        'draft', 'temp', 'test', 'sample', 'copy'
    ]
    
    # Pattern-based assessment
    if any(pattern in filename for pattern in trusted_patterns):
        return "trusted"
    elif any(pattern in filename for pattern in untrusted_patterns):
        return "untrusted"
    else:
        return "trusted"  # Default to trusted

def classify_document_type(content: str) -> str:
    """
    Document type classification basato su content analysis.
    
    Types:
    - technical_manual: Manuali tecnici
    - specification: Specifiche tecniche  
    - report: Report e analisi
    - guide: Guide e tutorial
    - reference: Materiale di riferimento
    - general: Contenuto generale
    """
    content_lower = content.lower()
    
    type_indicators = {
        "technical_manual": ["manual", "handbook", "procedure", "operation"],
        "specification": ["specification", "standard", "requirement", "criteria"],
        "report": ["analysis", "report", "study", "findings", "results"],
        "guide": ["guide", "tutorial", "how-to", "instructions"],
        "reference": ["reference", "glossary", "dictionary", "catalog"]
    }
    
    for doc_type, keywords in type_indicators.items():
        if any(keyword in content_lower for keyword in keywords):
            return doc_type
    
    return "general"
```

#### **4. Document Chunking Strategy**

```python
def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Advanced document chunking con context preservation.
    
    Chunking Strategy:
    - Chunk size: 700 characters (optimal for technical content)
    - Overlap: 120 characters (context preservation)
    - Separators: Hierarchical (paragraphs → sentences → words)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,         # 700 chars
        chunk_overlap=settings.chunk_overlap,   # 120 chars overlap
        separators=[
            "\n\n",    # Paragraphs (highest priority)
            "\n",      # Lines
            ". ",      # Sentences  
            " ",       # Words (lowest priority)
        ],
        keep_separator=True,  # Preserve structural indicators
        length_function=len,
        is_separator_regex=False
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    
    # Post-processing: chunk metadata enrichment
    for i, chunk in enumerate(chunked_docs):
        chunk.metadata.update({
            "chunk_id": f"{chunk.metadata['filename']}_{i}",
            "chunk_index": i,
            "chunk_size": len(chunk.page_content),
            "parent_document": chunk.metadata['filename']
        })
    
    return chunked_docs
```

#### **5. Processing Statistics & Monitoring**

```python
def generate_processing_statistics(processed_docs: List[Document]) -> Dict:
    """
    Generazione statistiche processing per monitoring.
    
    Statistics:
    - Document counts per format
    - Quality scores distribution  
    - Trustability breakdown
    - Processing success rates
    """
    stats = {
        "total_documents": len(processed_docs),
        "by_format": defaultdict(int),
        "by_trustability": defaultdict(int),
        "by_content_type": defaultdict(int),
        "quality_distribution": {
            "high": 0, "medium": 0, "low": 0
        },
        "average_quality_score": 0.0,
        "processing_errors": 0
    }
    
    quality_scores = []
    for doc in processed_docs:
        # Format statistics
        file_ext = Path(doc.metadata['filename']).suffix.lower()
        stats["by_format"][file_ext] += 1
        
        # Trustability statistics
        stats["by_trustability"][doc.metadata['trustability']] += 1
        
        # Content type statistics
        stats["by_content_type"][doc.metadata['content_type']] += 1
        
        # Quality statistics
        quality_score = doc.metadata.get('quality_score', 0.0)
        quality_scores.append(quality_score)
        
        if quality_score >= 0.8:
            stats["quality_distribution"]["high"] += 1
        elif quality_score >= 0.6:
            stats["quality_distribution"]["medium"] += 1
        else:
            stats["quality_distribution"]["low"] += 1
    
    stats["average_quality_score"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    return stats
```

### Integration con RAG System

```python
# Integration nel RAG Tool
@tool('rag_system')
def rag_system(question: str) -> str:
    """RAG system con advanced document processing"""
    
    # 1. Document Discovery & Loading
    document_paths = scan_docs_folder("docs_test")
    raw_documents = load_documents(document_paths)  # ← Advanced processing
    
    # 2. Processing Statistics
    stats = generate_processing_statistics(raw_documents)
    print(f"Processed {stats['total_documents']} documents")
    print(f"   Trusted: {stats['by_trustability']['trusted']}")
    print(f"   Average Quality: {stats['average_quality_score']:.2f}")
    
    # 3. Chunking & Indexing
    split_docs = split_documents(raw_documents, settings)
    
    # 4. RAG Execution con quality-aware retrieval
    # ... resto del RAG pipeline
```

---

## Advanced Retrieval Strategy Architecture

### Hybrid Search & MMR Integration

Il sistema implementa una **strategia di retrieval avanzata** che combina ricerca semantica, text matching e diversificazione tramite MMR (Maximal Marginal Relevance) per ottimizzare qualità e diversità dei risultati.

#### **Architettura Hybrid Retrieval**

```python
# Hybrid Search Pipeline
def hybrid_search(
    client: QdrantClient,
    collection_name: str, 
    query: str,
    embeddings,
    settings: Settings
) -> List[Dict]:
    """
    Advanced hybrid search combining semantic + text retrieval.
    
    Pipeline Stages:
    1. Query Embedding: Vectorization della query
    2. Semantic Search: Vector similarity search (top_n_semantic)
    3. Text Search: BM25-like text matching (top_n_text)  
    4. Hybrid Fusion: Reciprocal Rank Fusion (RRF)
    5. MMR Diversification: Maximal Marginal Relevance selection
    6. Final Ranking: Score normalization e ranking finale
    """
    
    # Stage 1: Query Preprocessing & Embedding
    query_vector = embeddings.embed_query(query)
    
    # Stage 2: Semantic Vector Search
    semantic_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=settings.top_n_semantic,    # 30 candidates
        score_threshold=settings.semantic_threshold  # 0.15 threshold
    )
    
    # Stage 3: Text-based Search  
    text_results = client.search(
        collection_name=collection_name,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="content",
                    match=models.MatchText(text=query)
                )
            ]
        ),
        limit=settings.top_n_text,        # 100 candidates
        with_vectors=True
    )
    
    # Stage 4: Hybrid Fusion (RRF)
    fused_results = reciprocal_rank_fusion(
        semantic_results, 
        text_results, 
        semantic_weight=0.7,    # Semantic preference
        text_weight=0.3         # Text complement
    )
    
    # Stage 5: MMR Diversification
    if settings.use_mmr and len(fused_results) > settings.final_k:
        diverse_results = apply_mmr_diversification(
            query_vector=query_vector,
            candidates=fused_results,
            lambda_mult=settings.mmr_lambda,  # 0.6 balance
            final_k=settings.final_k          # 6 results
        )
    else:
        diverse_results = fused_results[:settings.final_k]
    
    return diverse_results
```

#### **1. Semantic Search Strategy**

```python
# Vector Similarity Search Configuration
SEMANTIC_SEARCH_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    "similarity_metric": "cosine",                                # Normalized vectors
    "top_candidates": 30,                                         # Initial candidates
    "score_threshold": 0.15,                                      # Minimum relevance
    "vector_size": 384                                            # Model dimensions
}

def configure_semantic_search(settings: Settings):
    """
    Semantic search optimization per technical content.
    
    Model Trade-offs:
    - all-MiniLM-L6-v2: Fast inference, good quality, 384d
    - all-MiniLM-L12-v2: Better quality, slower, 768d  
    - all-mpnet-base-v2: Best quality, slowest, 768d
    
    Threshold Settings:
    - 0.10-0.20: Strict relevance (technical queries)
    - 0.05-0.15: Moderate relevance (general queries)
    - 0.00-0.10: Broad relevance (exploratory queries)
    """
    return {
        "model": SentenceTransformer(settings.hf_model_name),
        "threshold": settings.semantic_threshold,
        "candidates": settings.top_n_semantic
    }
```

#### **2. Text Matching Strategy**

```python
# BM25-style Text Search Integration
TEXT_SEARCH_CONFIG = {
    "algorithm": "BM25-inspired",      # Text relevance scoring
    "max_candidates": 100,             # Text search scope
    "field_weights": {
        "content": 1.0,                # Main content weight
        "title": 1.5,                  # Title boost
        "metadata": 0.8                # Metadata relevance
    },
    "term_boosting": True              # Query term importance
}

def text_search_strategy(query: str, settings: Settings):
    """
    Text-based retrieval per exact matches e terminology.
    
    Advantages:
    - Exact terminology matching
    - Acronym and technical term recognition
    - Phrase and quote preservation
    - Language-specific patterns
    
    Use Cases:
    - Technical specifications lookup
    - Exact procedure retrieval  
    - Acronym expansion
    - Regulatory reference finding
    """
    return {
        "query_filter": construct_text_filter(query),
        "max_results": settings.top_n_text,
        "boosting_enabled": True
    }
```

#### **3. Reciprocal Rank Fusion (RRF)**

```python
def reciprocal_rank_fusion(
    semantic_results: List[ScoredPoint],
    text_results: List[ScoredPoint], 
    semantic_weight: float = 0.7,
    text_weight: float = 0.3,
    k: int = 60
) -> List[Dict]:
    """
    Reciprocal Rank Fusion per combinazione risultati eterogenei.
    
    RRF Formula:
    RRF_score = semantic_weight * (1/(k + semantic_rank)) + 
                text_weight * (1/(k + text_rank))
    
    Parameters:
    - k=60: Smoothing parameter (standard value)
    - semantic_weight=0.7: Preference for semantic relevance
    - text_weight=0.3: Complement with text matching
    
    Benefits:
    - Handles different scoring scales
    - Combines heterogeneous retrieval methods
    - Robust against outlier scores
    - Position-based rather than score-based fusion
    """
    
    # Create unified candidate pool
    all_candidates = {}
    
    # Process semantic results
    for rank, result in enumerate(semantic_results, 1):
        doc_id = result.id
        semantic_score = 1 / (k + rank)
        
        all_candidates[doc_id] = {
            "point": result,
            "semantic_rank": rank,
            "semantic_score": semantic_score,
            "text_rank": float('inf'),  # Default if not in text results
            "text_score": 0.0
        }
    
    # Process text results
    for rank, result in enumerate(text_results, 1):
        doc_id = result.id
        text_score = 1 / (k + rank)
        
        if doc_id in all_candidates:
            all_candidates[doc_id]["text_rank"] = rank
            all_candidates[doc_id]["text_score"] = text_score
        else:
            all_candidates[doc_id] = {
                "point": result,
                "semantic_rank": float('inf'),
                "semantic_score": 0.0,
                "text_rank": rank,
                "text_score": text_score
            }
    
    # Calculate RRF scores
    for candidate in all_candidates.values():
        candidate["rrf_score"] = (
            semantic_weight * candidate["semantic_score"] +
            text_weight * candidate["text_score"]
        )
    
    # Sort by RRF score
    ranked_candidates = sorted(
        all_candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    return ranked_candidates
```

#### **4. MMR (Maximal Marginal Relevance) Diversification**

```python
def apply_mmr_diversification(
    query_vector: List[float],
    candidates: List[Dict],
    lambda_mult: float = 0.6,
    final_k: int = 6
) -> List[Dict]:
    """
    MMR diversification per riduzione ridondanza risultati.
    
    MMR Formula:
    MMR = λ * Sim(query, doc) - (1-λ) * max[Sim(doc, selected)]
    
    Parameters:
    - λ (lambda_mult): Relevance vs Diversity balance
      - λ=1.0: Pure relevance (no diversification)
      - λ=0.6: Balanced approach (recommended)  
      - λ=0.0: Pure diversity (may sacrifice relevance)
    
    Algorithm:
    1. Select highest relevance document first
    2. For each subsequent selection:
       - Calculate relevance to query
       - Calculate similarity to already selected docs
       - Choose doc maximizing MMR score
    3. Repeat until final_k documents selected
    """
    
    if len(candidates) <= final_k:
        return candidates[:final_k]
    
    # Extract vectors from candidates
    candidate_vectors = []
    for candidate in candidates:
        vector = candidate["point"].vector
        candidate_vectors.append(vector)
    
    # Apply MMR selection
    selected_indices = mmr_select(
        query_vec=query_vector,
        candidates_vecs=candidate_vectors,
        k=final_k,
        lambda_mult=lambda_mult
    )
    
    # Return diversified selection
    return [candidates[i] for i in selected_indices]

def mmr_select(
    query_vec: List[float],
    candidates_vecs: List[List[float]], 
    k: int,
    lambda_mult: float
) -> List[int]:
    """
    Core MMR selection algorithm.
    
    Complexity: O(k × n) where k=final results, n=candidates
    
    Trade-offs:
    - λ=1.0: Equivalent to top-K selection (no diversity benefit)
    - λ=0.8-0.9: High relevance, minimal diversification  
    - λ=0.6-0.7: Balanced relevance and diversity (recommended)
    - λ=0.3-0.5: High diversity, may sacrifice relevance
    - λ=0.0: Maximum diversity (may be irrelevant)
    """
    selected = []
    remaining = list(range(len(candidates_vecs)))
    
    # Step 1: Select most relevant document first
    query_similarities = [
        cosine_similarity(query_vec, candidates_vecs[i]) 
        for i in remaining
    ]
    best_idx = remaining[np.argmax(query_similarities)]
    selected.append(best_idx)
    remaining.remove(best_idx)
    
    # Step 2: Iteratively select diverse relevant documents
    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_candidate = None
        
        for candidate_idx in remaining:
            # Relevance component
            relevance = cosine_similarity(query_vec, candidates_vecs[candidate_idx])
            
            # Diversity component (max similarity to selected)
            if selected:
                max_similarity = max(
                    cosine_similarity(candidates_vecs[candidate_idx], candidates_vecs[sel_idx])
                    for sel_idx in selected
                )
            else:
                max_similarity = 0
            
            # MMR score calculation
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate_idx
        
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    
    return selected
```

### Qdrant Vector Database Integration

#### **1. Collection Configuration**

```python
def recreate_collection_for_rag(
    client: QdrantClient, 
    settings: Settings, 
    vector_size: int = 384
):
    """
    Qdrant collection optimization per RAG workloads.
    
    Configuration Choices:
    
    Distance Metric:
    - Cosine: Optimal per normalized embeddings (sentence-transformers)
    - Dot: Faster ma richiede normalizzazione manuale
    - Euclidean: Meno adatto per embeddings semantici
    
    HNSW Index:
    - m=32: Connections per node (quality vs memory trade-off)
    - ef_construct=256: Construction depth (quality vs speed)
    
    Quantization:
    - Scalar quantization: 4x memory reduction, minimal accuracy loss
    - always_ram=False: Disk storage con RAM caching
    """
    
    client.recreate_collection(
        collection_name=settings.collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,           # 384 for MiniLM-L6-v2
            distance=models.Distance.COSINE,  # Optimal for normalized vectors
            hnsw_config=models.HnswConfigDiff(
                m=32,                   # Avg connections (higher = better quality)
                ef_construct=256,       # Search depth during build
                full_scan_threshold=10000,  # Fallback to exact search
                max_indexing_threads=0      # Use all available threads
            )
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2,   # Parallel processing segments
            max_segment_size=None,      # Auto-sizing
            memmap_threshold=None,      # Auto-threshold  
            indexing_threshold=20000,   # Start indexing after 20K vectors
            flush_interval_sec=5,       # Periodic flush interval
            max_optimization_threads=None  # Use available threads
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,    # 8-bit quantization
                quantile=0.99,                  # Outlier handling
                always_ram=False                # Disk storage with RAM cache
            )
        )
    )
```

#### **2. Indexing Strategy**

```python
def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Document], 
    embeddings,
    batch_size: int = 100
):
    """
    Efficient batch indexing con metadata preservation.
    
    Optimization Features:
    - Batch processing: Reduced API calls
    - Parallel embedding: Concurrent vector generation
    - Metadata enrichment: Full document context preservation
    - Error recovery: Robust error handling
    """
    
    total_chunks = len(chunks)
    print(f"Indexing {total_chunks} chunks in batches of {batch_size}")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        
        try:
            # Parallel embedding generation
            texts = [chunk.page_content for chunk in batch]
            vectors = embeddings.embed_documents(texts)
            
            # Prepare points for upsert
            points = []
            for j, (chunk, vector) in enumerate(zip(batch, vectors)):
                point_id = i + j
                
                # Comprehensive metadata preservation
                payload = {
                    "content": chunk.page_content,
                    "filename": chunk.metadata.get("filename", "unknown"),
                    "trustability": chunk.metadata.get("trustability", "trusted"),
                    "content_type": chunk.metadata.get("content_type", "general"),
                    "chunk_id": chunk.metadata.get("chunk_id", f"chunk_{point_id}"),
                    "chunk_index": chunk.metadata.get("chunk_index", j),
                    "quality_score": chunk.metadata.get("quality_score", 0.8),
                    "processing_timestamp": chunk.metadata.get("processing_timestamp", ""),
                    "technical_level": chunk.metadata.get("technical_level", "medium")
                }
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            
            # Batch upsert
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # Ensure consistency
            )
            
            print(f"Indexed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            continue
    
    print(f"Indexing completed: {total_chunks} chunks processed")
```

### Configurable Retrieval Parameters

```python
@dataclass
class RetrievalSettings:
    """
    Comprehensive retrieval configuration con trade-off documentation.
    """
    
    # Semantic Search Configuration
    top_n_semantic: int = 30
    """Initial semantic candidates (quality vs speed trade-off)"""
    
    semantic_threshold: float = 0.15  
    """Minimum semantic similarity (precision vs recall trade-off)"""
    
    # Text Search Configuration
    top_n_text: int = 100
    """Text search candidates (coverage vs performance trade-off)"""
    
    # Hybrid Fusion Configuration
    semantic_weight: float = 0.7
    """Semantic vs text balance in RRF (domain-specific optimization)"""
    
    text_weight: float = 0.3
    """Text matching importance (terminology vs semantics trade-off)"""
    
    # MMR Diversification
    use_mmr: bool = True
    """Enable diversification (diversity vs pure relevance trade-off)"""
    
    mmr_lambda: float = 0.6
    """Relevance vs diversity balance (0.0=pure diversity, 1.0=pure relevance)"""
    
    # Final Results
    final_k: int = 6
    """Final result count (context window vs comprehensiveness trade-off)"""
    
    # Performance Configuration
    batch_size: int = 100
    """Indexing batch size (memory vs speed trade-off)"""
    
    parallel_embedding: bool = True
    """Parallel embedding generation (speed vs resource usage)"""
```

### Performance Monitoring

```python
def track_retrieval_performance(query: str, results: List[Dict]) -> Dict:
    """
    Retrieval performance tracking per optimization.
    
    Metrics:
    - Latency: Total retrieval time
    - Candidate Distribution: Semantic vs text results  
    - Diversity Score: MMR effectiveness
    - Quality Metrics: Average relevance scores
    """
    return {
        "query": query,
        "total_results": len(results),
        "semantic_candidates": count_semantic_results(results),
        "text_candidates": count_text_results(results),
        "average_relevance": calculate_average_relevance(results),
        "diversity_score": calculate_diversity_score(results),
        "retrieval_latency": measure_retrieval_time(),
        "mmr_effectiveness": assess_mmr_impact(results)
    }
```

---

## Ethical Question Analysis & Validation

### Dual Validation Router System

Il sistema implementa un **doppio router di validazione** che garantisce conformità sia al dominio aeronautico che agli standard etici, prevenendo processing di contenuti inappropriati o off-topic.

#### **Architettura Dual Router**

```python
# Flow Router Sequence
@router(generate_question)
def aeronautic_question_analysis(self):
    """
    Primo router: Validazione dominio aeronautico.
    
    Validation Scope:
    - Rilevanza aeronautica della domanda
    - Filtering domande off-topic
    - Technical domain compliance
    - Subject matter appropriateness
    
    Returns:
    - "success": Domanda aeronauticamente rilevante
    - "retry": Domanda off-topic, richiedi input diverso
    """
    llm = AzureChatOpenAI(
        azure_deployment=model_name,
        api_version=api_version,
        temperature=0,           # Deterministic validation
        max_retries=2
    )
    
    validation_prompt = [
        {"role": "system", "content": "You are an expert in aeronautics."},
        {"role": "user", "content": f"""
        Analyze this question for aeronautic relevance: "{self.state.question_input}"
        
        Aeronautic domains include:
        - Aircraft design and engineering
        - Flight operations and procedures  
        - Aviation industry and business
        - Aerospace technology
        - Air traffic management
        - Aircraft maintenance and safety
        - Pilot training and certification
        - Airport operations
        - Aviation regulations and standards
        
        Respond with exactly "True" if aeronautic-relevant, "False" if not.
        """}
    ]
    
    try:
        response = llm.invoke(validation_prompt)
        is_aeronautic = "true" in response.content.lower()
        
        if is_aeronautic:
            print(f"Aeronautic validation: PASSED")
            return "success"
        else:
            print(f"Aeronautic validation: FAILED - Off-topic question")
            return "retry"
            
    except Exception as e:
        print(f"Aeronautic validation error: {e}")
        return "retry"

@router("success")
def ethic_question_analysis(self):
    """
    Secondo router: Validazione etica e compliance.
    
    Ethical Validation Scope:
    - Content appropriateness
    - Safety and security compliance
    - Professional ethics adherence
    - Harmful content prevention
    - Privacy and confidentiality respect
    
    Returns:
    - "success-ethical": Passa sia validazione aeronautica che etica
    - "retry": Fallisce validazione etica, richiedi input diverso
    """
    llm = AzureChatOpenAI(
        azure_deployment=model_name,
        api_version=api_version,
        temperature=0,
        max_retries=2
    )
    
    ethical_validation_prompt = [
        {"role": "system", "content": """
        You are an AI ethics compliance specialist for aviation industry.
        Ensure all questions comply with professional, safety, and ethical standards.
        """},
        {"role": "user", "content": f"""
        Evaluate this aeronautic question for ethical compliance: "{self.state.question_input}"
        
        REJECT if question involves:
        Harmful or dangerous activities
        Security vulnerabilities or exploits
        Illegal aviation activities
        Confidential or classified information
        Personal data or privacy violations
        Discriminatory or biased content
        Misinformation or false claims
        
        ACCEPT if question is:
        Educational and informational
        Professional development focused
        Technical knowledge seeking
        Industry best practices inquiry
        Safety and compliance oriented
        Academic research purposes
        
        Respond with exactly "APPROVED" if ethically compliant, "REJECTED" if not.
        """}
    ]
    
    try:
        response = llm.invoke(ethical_validation_prompt)
        is_ethical = "approved" in response.content.lower()
        
        if is_ethical:
            print(f"Ethical validation: PASSED") 
            print(f"Question approved for processing: '{self.state.question_input}'")
            return "success-ethical"
        else:
            print(f"Ethical validation: FAILED - Inappropriate content")
            print(f"Please rephrase your question to comply with ethical guidelines")
            return "retry"
            
    except Exception as e:
        print(f"Ethical validation error: {e}")
        return "retry"
```

#### **Flow Integration con Dual Validation**

```python
# Complete Flow Sequence con Ethical Gating
class AeronauticRagFlow(Flow[AeronauticRagState]):
    """
    Multi-stage RAG flow con dual validation gating.
    
    Flow Sequence:
    1. starting_procedure() → Initialize
    2. generate_question() → User Input  
    3. aeronautic_question_analysis() → Domain Validation
    4. ethic_question_analysis() → Ethical Validation  
    5. rag_analysis() → RAG Processing (only if both validations pass)
    6. web_analysis() → Web Research
    7. aggregate_results() → Document Synthesis
    8. bias_check() → Bias Detection & Redaction
    9. plot_generation() → Flow Visualization
    """
    
    @listen("success-ethical")  # ← Only executes after BOTH validations pass
    def rag_analysis(self):
        """
        RAG processing - executes solo dopo doppia validazione.
        
        Prerequisites:
        aeronautic_question_analysis() returned "success"
        ethic_question_analysis() returned "success-ethical"
        
        This ensures:
        - Only aeronautically relevant questions are processed
        - Only ethically compliant questions are processed
        - No computational resources wasted on invalid queries
        - Compliance with professional and ethical standards
        """
        print(f"Initiating RAG analysis for validated question...")
        
        aero_crew = AeronauticRagCrew().crew()
        result = aero_crew.kickoff(inputs={
            "question": self.state.question_input,
        })
        
        # Context preservation for evaluation
        with open("output/last_context.txt", "r", encoding="utf-8") as f:
            context = f.read()
        
        self.state.rag_result = result.raw
        
        return {
            "aero_crew": aero_crew,
            "rag_context": context,
            "rag_result": result.raw,
            "validated_question": self.state.question_input
        }
```

### Ethical Compliance Framework

#### **1. Ethical Guidelines Taxonomy**

```python
ETHICAL_COMPLIANCE_FRAMEWORK = {
    "safety_compliance": {
        "description": "Avoid safety-compromising information",
        "examples": [
            "No dangerous flight procedures",
            "No security vulnerability details", 
            "No emergency override instructions"
        ],
        "validation": "Safety impact assessment"
    },
    
    "professional_ethics": {
        "description": "Maintain professional aviation standards",
        "examples": [
            "Evidence-based information only",
            "Industry best practices focus",
            "Professional development orientation"
        ],
        "validation": "Professional standards compliance"
    },
    
    "privacy_protection": {
        "description": "Respect confidential and personal information",
        "examples": [
            "No personal flight records",
            "No confidential airline data",
            "No individual identification"
        ],
        "validation": "Privacy impact assessment"
    },
    
    "accuracy_responsibility": {
        "description": "Prevent misinformation dissemination",
        "examples": [
            "Fact-based responses only",
            "Source attribution required",
            "Uncertainty acknowledgment"
        ],
        "validation": "Information accuracy verification"
    },
    
    "inclusivity_standards": {
        "description": "Maintain inclusive and non-discriminatory content",
        "examples": [
            "Gender-neutral language",
            "Cultural sensitivity",
            "Equal representation"
        ],
        "validation": "Inclusivity compliance check"
    }
}
```

#### **2. Validation Error Handling**

```python
def handle_validation_failure(validation_type: str, question: str) -> str:
    """
    Structured error handling per validation failures.
    
    Provides user guidance for reformulating questions to meet
    both aeronautic relevance and ethical compliance standards.
    """
    
    error_guidance = {
        "aeronautic_relevance": {
            "message": "Question is not related to aeronautics domain.",
            "suggestions": [
                "Focus on aircraft design, operations, or technology",
                "Ask about aviation industry, regulations, or procedures", 
                "Inquire about flight training, safety, or maintenance",
                "Consider aerospace engineering or air traffic topics"
            ],
            "examples": [
                "How does wing design affect aircraft efficiency?",
                "What are the latest trends in aviation safety?",
                "Explain the certification process for new aircraft."
            ]
        },
        
        "ethical_compliance": {
            "message": "Question does not meet ethical compliance standards.",
            "suggestions": [
                "Ensure question seeks educational information",
                "Focus on professional development topics",
                "Avoid sensitive security or safety details",
                "Request general industry knowledge"
            ],
            "examples": [
                "What are best practices in aircraft maintenance?",
                "How do airlines optimize fuel efficiency?",
                "What training is required for commercial pilots?"
            ]
        }
    }
    
    guidance = error_guidance.get(validation_type, {})
    
    response = f"""
    Validation Failed: {guidance.get('message', 'Unknown validation error')}
    
    Suggestions:
    {chr(10).join(f"   • {suggestion}" for suggestion in guidance.get('suggestions', []))}
    
    Example Questions:
    {chr(10).join(f"   • {example}" for example in guidance.get('examples', []))}
    
    Please rephrase your question and try again.
    """
    
    return response
```

#### **3. Compliance Monitoring & Reporting**

```python
def track_validation_metrics():
    """
    Monitoring delle metriche di validazione per continuous improvement.
    
    Tracked Metrics:
    - Aeronautic validation success rate
    - Ethical compliance success rate  
    - Common failure patterns
    - User retry behavior
    - Question reformulation success
    """
    return {
        "validation_stats": {
            "aeronautic_pass_rate": "% domande aeronautiche valide",
            "ethical_pass_rate": "% domande eticamente conformi", 
            "combined_pass_rate": "% domande passanti entrambe validazioni",
            "retry_rate": "% utenti che riprovano dopo fallimento",
            "reformulation_success": "% successo dopo reformulazione"
        },
        
        "failure_analysis": {
            "common_off_topic_patterns": "Pattern domande off-topic frequenti",
            "ethical_violation_categories": "Categorie violazioni etiche",
            "user_guidance_effectiveness": "Efficacia guidance utente"
        },
        
        "optimization_insights": {
            "prompt_refinement_opportunities": "Aree miglioramento prompt",
            "threshold_adjustment_needs": "Necessità aggiustamento soglie",
            "user_experience_enhancements": "Miglioramenti UX possibili"
        }
    }

class ValidationAuditLog:
    """
    Audit logging per compliance e continuous improvement.
    """
    
    def log_validation_attempt(
        self, 
        question: str,
        aeronautic_result: str,
        ethical_result: str,
        final_decision: str,
        timestamp: str
    ):
        """Log structured validation data for analysis"""
        
        audit_entry = {
            "timestamp": timestamp,
            "question_hash": hash(question),  # Privacy-preserving
            "aeronautic_validation": aeronautic_result,
            "ethical_validation": ethical_result, 
            "final_decision": final_decision,
            "processing_allowed": final_decision == "success-ethical"
        }
        
        # Append to audit log for analysis
        self.append_to_audit_log(audit_entry)
    
    def generate_compliance_report(self, period: str) -> Dict:
        """Generate periodic compliance reporting"""
        
        return {
            "period": period,
            "total_questions_processed": self.count_total_questions(period),
            "compliance_rate": self.calculate_compliance_rate(period),
            "common_violations": self.identify_common_violations(period),
            "improvement_recommendations": self.generate_recommendations(period)
        }
```

### Integration con Flow State Management

```python
# Enhanced State Model con Validation Tracking
@dataclass
class AeronauticRagState(BaseModel):
    """Enhanced state model con validation tracking"""
    
    # Core State
    question_input: str = ""
    rag_result: str = ""
    web_result: str = ""
    all_results: str = ""
    document: str = ""
    final_doc: str = ""
    
    # Validation State (NEW)
    aeronautic_validation_passed: bool = False
    ethical_validation_passed: bool = False
    validation_timestamp: str = ""
    validation_attempts: int = 0
    
    # Quality State  
    rag_context: str = ""
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    bias_check_completed: bool = False
    
    # Flow Metadata
    flow_execution_id: str = ""
    processing_stage: str = "initialized"
```

---

## Scalability & Extensions

### Horizontal Scaling
- Multiple crew instances
- Distributed tool execution
- Load balancing per web requests

### Vertical Scaling  
- Enhanced LLM models
- Larger vector stores
- More sophisticated routing logic

### Extensibility Points
- Nuove crew specializzate
- Tool aggiuntivi per domini specifici
- Integration con sistemi enterprise

---

*Documentazione CrewAI Architecture*  
*Versione: 2.0*  
*Focus: Flow, Crews & Tools*