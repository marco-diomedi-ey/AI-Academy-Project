#!/usr/bin/env python
import streamlit as st
from pydantic import BaseModel
from crewai.flow import Flow, listen, start, router
from rag_flow.crews.bias_crew.bias_crew import BiasCrew
from rag_flow.crews.rag_crew.rag_crew import AeronauticRagCrew
from rag_flow.crews.web_crew.web_crew import WebCrew
from rag_flow.crews.doc_crew.doc_crew import DocCrew
import os
import pandas as pd
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import Dict, Any
from opik import configure 
from opik.integrations.crewai import track_crewai 
import json


# Carica le variabili d'ambiente
load_dotenv()

def create_download_link(data, filename, label):
    """Crea un link di download personalizzato"""
    import base64
    
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data
        
    b64 = base64.b64encode(data_bytes).decode()
    
    return f"""
    <a href="data:application/octet-stream;base64,{b64}" 
       download="{filename}" 
       style="text-decoration: none;">
        <button style="
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            width: 100%;
            font-size: 14px;
            font-weight: 500;
        " onmouseover="this.style.backgroundColor='#0052a3'" 
           onmouseout="this.style.backgroundColor='#0066cc'">
            {label}
        </button>
    </a>
    """

# Configurazioni iniziali
configure(use_local=True)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OTEL_SDK_DISABLED"] = "true"

track_crewai(project_name="final-project")

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="Aeronautic RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #0d47a1;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    
    /* Forza lo stile blu per tutti i download button */
    .stDownloadButton > button {
        background-color: #0066cc !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #0052a3 !important;
        color: white !important;
    }
    
    .stDownloadButton > button:focus {
        background-color: #0052a3 !important;
        color: white !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 102, 204, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

#load_dotenv()  # Carica le variabili d'ambiente dal file .env 
endpoint = os.getenv("AZURE_API_BASE")
key = os.getenv("AZURE_API_KEY")
deployment_name = os.getenv("MODEL")  # nome deployment modello completions
api_version=os.getenv("AZURE_API_VERSION", "2024-06-01")

class AeronauticRagState(BaseModel):
    """
    State model for Aeronautic RAG Flow execution.
    
    This Pydantic model manages the state data throughout the RAG flow execution,
    storing user inputs, intermediate results, and final aggregated outputs from
    different processing stages.
    
    Attributes
    ----------
    question_input : str
        User's input question about aeronautics (default: "")
    rag_result : str
        Result from RAG system analysis using local document knowledge base (default: "")
    web_result : str
        Result from web search and analysis using external sources (default: "")
    all_results : str
        Aggregated results combining RAG and web analysis outputs (default: "")
        
    Notes
    -----
    State persistence enables tracking of data flow between different crew executions
    and allows for comprehensive result aggregation and document generation.
    """
    question_input: str = ""
    rag_result: str = ""
    web_result: str = ""
    all_results: str = ""
    document: str = ""
    final_doc: str = ""
    rag_context: str = ""  # Nuovo campo per ctx
    validation_error: str = ""  # Tipo di errore di validazione
    error_message: str = ""  # Messaggio di errore per il frontend

class AeronauticRagFlow(Flow[AeronauticRagState]):
    """
    Multi-stage RAG flow for comprehensive aeronautic question answering with Streamlit UI integration.
    
    This CrewAI Flow orchestrates a sophisticated question-answering pipeline that combines
    local knowledge base retrieval, web search capabilities, and document generation,
    with real-time UI updates through Streamlit.
    """

    def __init__(self):
        super().__init__()
        # Streamlit UI components per aggiornamenti real-time
        self.status_placeholder = None
        self.progress_placeholder = None
        
    def set_ui_components(self, status_placeholder, progress_placeholder):
        """Imposta i componenti UI per aggiornamenti real-time"""
        self.status_placeholder = status_placeholder
        self.progress_placeholder = progress_placeholder
    
    def update_ui(self, message: str, progress: float):
        """Aggiorna l'interfaccia Streamlit"""
        if self.status_placeholder:
            self.status_placeholder.info(message)
        if self.progress_placeholder:
            self.progress_placeholder.progress(progress, text="Processing...")

    @start('retry')
    def starting_procedure(self):
        """
        Initialize the Aeronautic RAG Flow execution.
        
        Entry point for the flow that sets up the initial state and begins
        the question-answering pipeline. Configured to retry on validation failure.
        
        Notes
        -----
        The 'retry' parameter enables automatic restart when question validation
        determines that the input is not relevant to aeronautics.
        """
        self.update_ui(":material/rocket_launch: Initializing pipeline...", 0.05)

    @listen(starting_procedure)
    def generate_question(self):
        """
        Capture user input for aeronautic question processing.
        
        Interactive step that prompts the user to enter their aeronautic-related
        question and stores it in the flow state for subsequent processing stages.
        
        State Updates
        -------------
        Updates self.state.question_input with the user's entered question.
        
        Notes
        -----
        This method uses interactive input() which requires console interaction.
        The captured question will be validated for aeronautic relevance in
        the next flow stage.
        """
        # La domanda è già impostata da Streamlit, non serve input()
        pass

    @router(generate_question)
    def aeronautic_question_analysis(self):
        """
        Validate question relevance to aeronautics using Azure OpenAI.
        """
        self.update_ui(":material/search: Step 1/6: Validating aeronautic relevance...", 0.15)
        
        try:
            llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",  # or your deployment
            api_version=api_version,  # or your api version
            temperature=0,
            max_retries=2,
            ) 
            messages=[
                    {"role": "system", "content": "You are an expert in aeronautics."},
                    {"role": "user", "content": f"Is the following question relevant to aeronautics? Question: {self.state.question_input}. Answer only with 'True' or 'False'"}
                ]
            
            res = llm.invoke(messages)
            res = res.content.strip().lower()

            if 'true' in res:
                return "success"
            else:
                # Imposta errore per il frontend
                self.state.validation_error = "aeronautic"
                self.state.error_message = "La domanda non è rilevante per l'aeronautica. Inserisci una domanda pertinente al settore aeronautico (aerei, elicotteri, droni, motori aeronautici, aerodinamica, etc.)."
                return "validation_failed"
        except Exception as e:
            # Errore tecnico durante la validazione
            self.state.validation_error = "technical"
            self.state.error_message = f"Errore durante la validazione aeronautica: {str(e)}"
            return "validation_failed"

    @router("success")
    def ethic_question_analysis(self):
        """
        Validate question ethics using Azure OpenAI.
        """
        self.update_ui(":material/balance: Step 2/6: Ethics validation...", 0.25)
        
        try:
            llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",  # or your deployment
            api_version=api_version,  # or your api version
            temperature=0,
            max_retries=2,
            ) 
            messages=[{"role": "system", "content":
                    """You are an ethical AI expert specialized in content moderation. 
                    Your role is to evaluate if questions are appropriate and ethical.
                    Consider a question ETHICAL if it:
                    - Seeks legitimate information
                    - Has educational or professional purpose
                    - Does not promote harm, violence, or illegal activities
                    - Does not involve personal attacks or hate speech
                    Consider a question UNETHICAL if it:
                    - Requests harmful, dangerous, or illegal information
                    - Contains hate speech, discrimination, or personal attacks
                    - Aims to manipulate, deceive, or cause harm
                    - Violates privacy or confidentiality
                    Be permissive with legitimate academic, technical, or professional questions."""
                    },
                    {"role": "user", "content": f"Is the following question ethical or harmful? Question: {self.state.question_input}. Answer only with 'True' or 'False'"}
                ]
            
            res = llm.invoke(messages)
            res = res.content.strip().lower()

            if 'true' in res:
                return "success-ethical"
            else:
                # Imposta errore per il frontend
                self.state.validation_error = "ethical"
                self.state.error_message = "La domanda presenta problematiche etiche. Inserisci una domanda appropriata che non contenga contenuti dannosi, discriminatori o inappropriati."
                return "validation_failed"
        except Exception as e:
            # Errore tecnico durante la validazione etica
            self.state.validation_error = "technical"
            self.state.error_message = f"Errore durante la validazione etica: {str(e)}"
            return "validation_failed"

    @listen("validation_failed")
    def handle_validation_error(self):
        """
        Gestisce gli errori di validazione fermando il Flow.
        """
        # Il Flow si ferma qui, l'errore è già impostato nello state
        self.update_ui(":material/error: Validation failed", 0.0)
        return None

    @listen("success-ethical")
    def rag_analysis(self):
        """
        Execute RAG-based analysis using local aeronautic knowledge base.
        """
        self.update_ui(":material/note_stack: Step 3/6: RAG analysis (local knowledge base)...", 0.45)
        
        aero_crew = AeronauticRagCrew().crew()
        result = (
            aero_crew.kickoff(inputs={"question": self.state.question_input,
                             })
        )
        
        # Leggi il context se disponibile
        try:
            with open("output/last_context.txt", "r", encoding="utf-8") as f:
                CONTEXT = f.read()
        except FileNotFoundError:
            CONTEXT = "Context file not available"
        except Exception as e:
            CONTEXT = f"Error reading context: {str(e)}"
            
        self.state.rag_result = result.raw
        return {
            "aero_crew": aero_crew,
            "rag_context": CONTEXT,
            "rag_result": result.raw,
            "question": self.state.question_input
        }
    
    @listen(rag_analysis)
    def web_analysis(self, payload:Dict[str, Any]):
        """
        Execute web-based analysis to complement local knowledge base.
        """
        self.update_ui(":material/captive_portal: Step 4/6: Web analysis (external sources)...", 0.65)
        
        web_crew = WebCrew().crew()
        result = (
            web_crew.kickoff(inputs={"question": self.state.question_input,
                             })
        )
        self.state.web_result = result.raw
        payload['web_result'] = result.raw
        payload['web_crew'] = web_crew
        return payload
    
    @listen(web_analysis)
    def aggregate_results(self, payload:Dict[str, Any]):
        """
        Aggregate and synthesize results from RAG and web analysis.
        """
        self.update_ui(":material/contract_edit: Step 5/6: Generating comprehensive document...", 0.85)
        
        aggregated = f"RAG Result: {self.state.rag_result}\n\nWeb Result: {self.state.web_result}"
        self.state.all_results = aggregated
        doc_crew = DocCrew().crew()
        result = (
            doc_crew
            .kickoff(inputs={"paper": aggregated,
                             })
        )
        self.state.document = result.raw
        payload['doc_context'] = aggregated
        payload['doc_result'] = result.raw
        payload['doc_crew'] = doc_crew
        return payload
    
    @listen(aggregate_results)
    def bias_check(self, payload:Dict[str, Any]):
        """
        Execute bias checking on the generated document.
        """
        self.update_ui(":material/fact_check: Step 6/6: Bias checking...", 0.95)
        
        bias_crew = BiasCrew().crew()
        result = (
            bias_crew
            .kickoff(inputs={"document": self.state.document,
                             })
        )
        self.state.final_doc = result.raw
        payload['bias_context'] = self.state.document
        payload['bias_result'] = result.raw
        payload['bias_crew'] = bias_crew
        return payload
    
    @listen(bias_check)
    def plot_generation(self, payload:Dict[str, Any]):
        """
        Generate and display flow execution visualization.
        """
        self.update_ui(":material/check_circle: Pipeline completed successfully!", 1.0)
        return payload

# Sidebar con informazioni sistema
with st.sidebar:
    # Logo EY centrato
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("src/rag_flow/EY.png", width=75)
        except:
            try:
                st.image("EY.png", width=100)
            except:
                st.markdown("<div style='text-align: center;'><strong>EY Logo</strong></div>", unsafe_allow_html=True)
    
    st.header("Aeronautic RAG System")
    st.markdown("---")
   
       
    st.subheader("Architettura Pipeline")
    st.markdown("""
    **Flow Stages:**
    1. Question Validation :material/search:
    2. Ethics Check :material/balance:
    3. RAG Analysis (Local KB) :material/note_stack:
    4. Web Analysis :material/captive_portal:
    5. Document Generation :material/contract_edit:
    6. Bias Check :material/fact_check:
    """)
    
    st.markdown("---")
    st.subheader("Components")
    st.markdown("""
    - **RAG Crew**: GPT-4o + text-embedding-ada-002 + Qdrant
    - **Web Crew**: SerperDev API
    - **Doc Crew**: Markdown Generator
    - **Bias Crew**: Content Moderator
    - **Monitoring**: Opik + RAGAS
    """)
    
    st.markdown("---")
    # st.subheader("Session Stats")
    # if 'total_queries' not in st.session_state:
    #     st.session_state.total_queries = 0
    # st.metric("Total Queries", st.session_state.total_queries)
    
    if st.button(":material/refresh: Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content
st.title("Aeronautic RAG Question Answering System")
st.markdown("""Sistema avanzato di risposta a domande aeronautiche con validazione etica e controllo bias.""")
st.markdown("""La risposta è stata generata dall'IA, sfruttando una knowledge base locale e una ricerca online in tempo reale,  \
potrebbe mostrare informazioni imprecise. Verifica sempre le risposte sulla documentazione originale.""")

# Tabs per organizzare l'interfaccia
tab1, tab2, tab3 = st.tabs([
    ":material/question_exchange: Query", 
    ":material/table_convert: Results History", 
    ":material/search: Flow Visualization"
])

with tab1:
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_area(
            "Inserisci la tua domanda aeronautica:",
            height=120,
            placeholder="Es: Quali sono i principi di funzionamento di un motore a reazione?",
            key="question_input"
        )
    
    with col2:
        st.markdown("### Options")
        show_intermediate = st.toggle("Show intermediate steps", value=True)
        show_metrics = st.toggle("Show RAGAS metrics", value=True)
        show_context = st.toggle("Show retrieved context", value=False)
    
    # Execute button sotto la text area con la stessa larghezza
    with col1:
        if st.button("Execute Pipeline", type="primary", use_container_width=True):
            if not question:
                st.warning(":material/warning: Inserisci una domanda prima di procedere")
            else:
                # Inizializza session state
                if 'execution_history' not in st.session_state:
                    st.session_state.execution_history = []
                
                try:
                    # Progress tracking
                    progress_bar = st.progress(0, text="Initializing pipeline...")
                    status_text = st.empty()
                    
                    # Container per risultati
                    results_container = st.container()
                    
                    # Crea il Flow con UI integration
                    aeronautic_rag_flow = AeronauticRagFlow()
                    aeronautic_rag_flow.state.question_input = question
                    aeronautic_rag_flow.set_ui_components(status_text, progress_bar)
                    
                    # Esegui il Flow - ora con step reali visibili
                    aeronautic_rag_flow.kickoff()
                    
                    # Verifica se ci sono errori di validazione
                    if aeronautic_rag_flow.state.validation_error:
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Mostra errore specifico basato sul tipo
                        if aeronautic_rag_flow.state.validation_error == "aeronautic":
                            st.error(f":material/search: **Validazione Aeronautica Fallita**\n\n{aeronautic_rag_flow.state.error_message}")
                            st.info(":material/lightbulb: **Esempi di domande valide:**\n- Come funziona un motore a reazione?\n- Quali sono i principi dell'aerodinamica?\n- Come si pilota un elicottero?\n- Che cos'è la portanza?")
                        elif aeronautic_rag_flow.state.validation_error == "ethical":
                            st.error(f":material/balance: **Validazione Etica Fallita**\n\n{aeronautic_rag_flow.state.error_message}")
                            st.info(":material/info: **Suggerimento:** Assicurati che la domanda sia appropriata e non contenga contenuti offensivi o dannosi.")
                        else:
                            st.error(f":material/error: **Errore Tecnico**\n\n{aeronautic_rag_flow.state.error_message}")
                        
                        # Pulsante per riprovare
                        st.markdown("---")
                        if st.button(":material/refresh: Inserisci una Nuova Domanda", type="primary", use_container_width=True):
                            st.rerun()
                            
                    elif not aeronautic_rag_flow.state.final_doc:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(":material/close: Errore durante l'esecuzione del pipeline. Riprova.")
                    else:
                        # Successo!
                        progress_bar.progress(1.0, text="Completed!")
                        status_text.success(":material/check_circle: Pipeline completed successfully!")
                        
                        # RESULTS DISPLAY
                        # st.markdown("---")
                        # st.markdown("## :material/description: Final Document")
                        
                        # # Final document in a nice box
                        # st.markdown(f"""
                        # <div class="success-box">{aeronautic_rag_flow.state.final_doc}</div>
                        # """, 
                        # unsafe_allow_html=True
                        # )
                        
                        # Intermediate results
                        if show_intermediate:
                            st.markdown("---")
                            st.markdown("### :material/layers: Results")
                            
                            with st.expander(":material/note_stack: RAG Analysis", expanded=False):
                                st.markdown(aeronautic_rag_flow.state.rag_result)
                            
                            with st.expander(":material/captive_portal: Web Analysis", expanded=False):
                                st.markdown(aeronautic_rag_flow.state.web_result)
                            
                            with st.expander(":material/contract_edit: Document Generation", expanded=False):
                                st.markdown(aeronautic_rag_flow.state.document)
                        
                        # Metrics section
                        if show_metrics:
                            st.markdown("---")
                            st.markdown("## :material/finance: Quality Metrics")
                            
                            try:
                                try:
                                    # Prova formato JSON standard
                                    with open("output/rag_eval_results.json", "r") as f:
                                        metrics_data = json.load(f)
                                    
                                    cols = st.columns(5)
                                    metric_names = {
                                        'answer_relevancy': ('Relevancy', ':material/target:'),
                                        'faithfulness': ('Faithfulness', ':material/check_box:'),
                                        'context_precision': ('Precision', ':material/search:'),
                                        'context_recall': ('Recall', ':material/finance:'),
                                        'answer_correctness': ('Answer Correctness',':material/finance:')
                                    }
                                    
                                    for idx, (key, (label, icon)) in enumerate(metric_names.items()):
                                        if key in metrics_data:
                                            cols[idx].metric(
                                                f"{icon} {label}",
                                                f"{metrics_data[key]:.2%}"
                                            )
                                except json.JSONDecodeError:
                                    df = pd.read_json("output/rag_eval_results.json", lines=True)
                                    metrics_data = df.iloc[-1].to_dict()  

                                    display_cols = ['user_input', 
                                                    'response',
                                                    'faithfulness', 
                                                    'answer_correctness', 
                                                    'answer_relevancy', 
                                                    'context_precision', 
                                                    'context_recall']
                                    
                                    st.dataframe(
                                        df[display_cols], 
                                        use_container_width=True,
                                        hide_index=True
                                    )
                            except FileNotFoundError:
                                st.info("RAGAS metrics not available for this query.")
                            except Exception as e:
                                st.warning(f"Could not load metrics: {str(e)}")
                        
                        # Context display
                        if show_context:
                            st.markdown("---")
                            st.markdown("## :material/note_stack: Retrieved Context")
                            try:
                                with open("output/last_context.txt", "r", encoding="utf-8") as f:
                                    rag_context = f.read()
                                with st.expander("View RAG Context"):
                                    st.text(rag_context)
                            except FileNotFoundError:
                                st.info("Context file not available.")
                            except Exception as e:
                                st.warning(f"Could not load context: {str(e)}")
                        
                        # Download options
                        st.markdown("---")
                        st.markdown("### :material/download: Downloads")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            download_link = create_download_link(
                                aeronautic_rag_flow.state.final_doc,
                                f"aeronautic_answer_{len(st.session_state.get('execution_history', []))}.md",
                                ":material/description: Final Document"
                            )
                            st.markdown(download_link, unsafe_allow_html=True)
                        
                        with col2:
                            try:
                                with open("output/last_context.txt", "r", encoding="utf-8") as f:
                                    context = f.read()
                                download_link = create_download_link(
                                    context,
                                    f"rag_context_{len(st.session_state.get('execution_history', []))}.txt",
                                    ":material/note_stack: RAG Context"
                                )
                                st.markdown(download_link, unsafe_allow_html=True)
                            except:
                                st.button(
                                    label=":material/note_stack: RAG Context",
                                    disabled=True,
                                    use_container_width=True
                                )
                        
                        with col3:
                            try:
                                with open("output/rag_eval_results.json", "r") as f:
                                    metrics_json = f.read()
                                download_link = create_download_link(
                                    metrics_json,
                                    f"metrics_{len(st.session_state.get('execution_history', []))}.json",
                                    ":material/analytics: RAGAS Metrics"
                                )
                                st.markdown(download_link, unsafe_allow_html=True)
                            except:
                                st.button(
                                    label=":material/analytics: RAGAS Metrics",
                                    disabled=True,
                                    use_container_width=True
                                )
                        
                        # Save to history
                        st.session_state.execution_history.append({
                            'question': question,
                            'rag_result': aeronautic_rag_flow.state.rag_result,
                            'web_result': aeronautic_rag_flow.state.web_result,
                            'document': aeronautic_rag_flow.state.document,
                            'final_document': aeronautic_rag_flow.state.final_doc
                            # 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        # Query count now tracked by execution_history length
                        
                except Exception as e:
                    st.error(f":material/close: Error during execution: {str(e)}")
                    with st.expander("Error Details"):
                        st.exception(e)

with tab2:
    st.header(":material/history: Execution History")
    
    if 'execution_history' in st.session_state and st.session_state.execution_history:
        for idx, item in enumerate(reversed(st.session_state.execution_history)):
            query_num = len(st.session_state.execution_history) - idx
            
            with st.expander(f"Query {query_num}: {item['question'][:80]}"):
                st.markdown(f"**:material/question_mark: Question:** {item['question']}")
                st.markdown("---")
                
                # Tabs per i risultati - usa .get() per gestire history vecchie
                result_tabs = st.tabs([
                    ":material/note_stack: RAG", 
                    ":material/captive_portal: Web", 
                    ":material/contract_edit: Document",
                    ":material/description: Final"
                ])
                
                with result_tabs[0]:
                    st.markdown(item.get('rag_result', 'Not available'))
                
                with result_tabs[1]:
                    st.markdown(item.get('web_result', 'Not available'))
                
                with result_tabs[2]:
                    st.markdown(item.get('document', 'Not available'))
                
                with result_tabs[3]:
                    st.markdown(item.get('final_document', 'Not available'))
                
                # Download per questo item
                st.download_button(
                    label=":material/download: Download Final Document",
                    data=item.get('final_document', ''),
                    file_name=f"aeronautic_answer_{query_num}.md",
                    mime="text/markdown",
                    key=f"download_{idx}"
                )
    else:
        st.info("No execution history yet. Run a query to see results here.")

with tab3:
    st.header(":material/account_tree: Flow Architecture Visualization")
    st.info("Click the button below to generate and display the flow diagram")
    
    if st.button(":material/device_hub: Generate Flow Diagram", use_container_width=False):
        with st.spinner("Generating flow visualization..."):
            try:
                flow = AeronauticRagFlow()
                flow.plot()
                
                # Il file è nella root del progetto
                if os.path.exists("crewai_flow.html"):
                    with open("crewai_flow.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    st.components.v1.html(html_content, height=600, scrolling=True)
                    st.success("✅ Flow diagram generated and displayed above!")
                else:
                    st.info("The flow diagram file was not created. Check the console for errors.")
                    
            except Exception as e:
                st.error(f"Error generating diagram: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>Aeronautic RAG System</strong></p>
    <p>Powered by CrewAI + Azure OpenAI + Qdrant | Monitored by Opik and RAGAS</p>

</div>
""", unsafe_allow_html=True)

def kickoff():
    """
    Initialize and execute the Aeronautic RAG Flow.
    
    Entry point function that creates an instance of AeronauticRagFlow
    and starts the complete question-answering pipeline execution.
    """
    aeronautic_rag_flow = AeronauticRagFlow()
    question = input("Inserisci la tua domanda aeronautica: ")
    aeronautic_rag_flow.state.question_input = question
    aeronautic_rag_flow.kickoff()

def plot():
    """
    Generate and display the flow architecture visualization.
    """
    aeronautic_rag_flow = AeronauticRagFlow()
    aeronautic_rag_flow.plot()

if __name__ == "__main__":
    # Se eseguito direttamente (non da Streamlit), usa la modalità console
    if not hasattr(st, 'session_state'):
        kickoff()
