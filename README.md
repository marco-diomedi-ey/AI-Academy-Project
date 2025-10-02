# AI-Academy-Project
Aeronautic RAG System with evaluation

**Versione:** 0.1

## Autori
- Fabio Rizzi
- Giulia Pisano  
- Marco Diomedi
- Riccardo Zuanetto
- Roberto Gennaro Sciarrino

## Descrizione del Progetto

RAG Flow è un sistema avanzato di Retrieval-Augmented Generation (RAG) progettato per l'analisi e il processing di documenti con un focus particolare sulla conformità al AI Act europeo. Il sistema utilizza un'architettura multi-agente basata su CrewAI Flow per orchestrare diversi processi specializzati: processing documentale, bias detection, ricerca web trusted e generazione di contenuti poetici.

## Caratteristiche Principali

- **Architettura Multi-Agente**: Implementazione di specialized crews per diverse tipologie di analisi
- **RAG Avanzato**: Sistema di retrieval ibrido con vector database (Qdrant/FAISS) e diversificazione MMR
- **Bias Detection**: Analisi automatica dei bias nei contenuti con crew dedicato
- **Web Search Trusted**: Ricerca web filtrata su domini approvati per garantire attendibilità delle fonti
- **Evaluation Framework**: Integrazione RAGAS per valutazione qualitativa delle risposte
- **Document Processing**: Pipeline multi-formato con quality filtering e assessment di trustability
- **Ethical Compliance**: Validazione etica delle domande con dual-agent approach

## Architettura del Sistema

Il sistema è organizzato in 4 crews principali:
- **DocCrew**: Processing e analisi documentale multi-formato
- **BiasCrew**: Detection e mitigazione bias nei contenuti
- **WebCrew**: Ricerca web con filtering su domini trusted
- **RagCrew**: Crew per utilizzo di RAG con documenti locali per individuazione di informazioni aeronautiche

## Installazione

```bash
cd rag_flow

# Installazione con CrewAI (raccomandato)
pip install crewai
crewai install  # Crea venv e installa dipendenze

# Alternativa con UV
pip install uv
uv sync

# Alternativa con pip
pip install -r requirements.txt

# Creazione file env
cp .env.full_example .env

# Da compilare con le proprie chiavi per variabili d'ambiente
```

## Struttura del Repository

```
rag_flow/
├── src/rag_flow/          # Codice sorgente principale
│   ├── crews/             # Implementazione crews specializzati
│   └── tools/             # Tools custom e integrazione RAG
├── docs/                  # Documentazione tecnica
├── eu_ai_act_docs/        # Documentazione compliance AI Act
└── output/                # Output delle elaborazioni
```

## Documentazione

Per maggiori dettagli sull'architettura e l'implementazione, consultare:
- `architettura_crewai.md` - Documentazione tecnica completa
- `technical_documentation.md` - Sintesi tecnica del progetto
