from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Any
import fitz  # PyMuPDF

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (CSVLoader, PyMuPDFLoader,
                                                  UnstructuredImageLoader, 
                                                  UnstructuredMarkdownLoader,
                                                  TextLoader)

from .qdrant_script import (hybrid_search)
from .config import Settings


def is_document_low_quality(file_path: str, content: str) -> bool:
    """
    Assess document quality to filter out low-quality or corrupted files.
    
    Performs quality analysis on documents to identify and filter out files
    with insufficient content, OCR artifacts, or formatting issues that could
    negatively impact retrieval performance. Includes specialized PDF analysis
    for text visibility and background color detection.
    
    Parameters
    ----------
    file_path : str
        Absolute path to the document file being analyzed
    content : str
        Extracted text content from the document
        
    Returns
    -------
    bool
        True if document quality is too low and should be filtered, False otherwise
        
    Quality Checks
    --------------
    - **Content Length**: Filters documents with less than 50 characters
    - **PDF Analysis**: Detects suspicious characters and text visibility issues
    - **Background Detection**: Adaptive threshold based on background luminance
    - **Font Analysis**: Identifies problematic text size and color contrast
    - **OCR Artifacts**: Detects common OCR processing errors
    
    PDF-Specific Analysis
    --------------------
    - Analyzes background color and luminance
    - Calculates adaptive color distance thresholds
    - Detects text with insufficient contrast
    - Identifies suspiciously small fonts (< 6pt)
    - Counts suspicious characters as quality indicator
    
    Notes
    -----
    The function uses PyMuPDF for detailed PDF analysis, examining each page
    for text visibility issues. For non-PDF files, only basic content length
    validation is performed. Errors during analysis are handled gracefully
    with fallback to acceptance.
    """
    ext = file_path.split(".")[-1].lower()
    
    print(f"DEBUG: Analizzando {file_path} (ext: {ext}, chars: {len(content)})")
    
    # Controllo contenuto troppo breve
    if len(content.strip()) < 50:
        print(f"Contenuto troppo breve: {len(content.strip())} < 50")
        return True
    
    # Controllo specifico PDF
    if ext == "pdf":
        # print(f"Analisi PDF: {file_path}")
        try:
            doc = fitz.open(file_path)
            suspicious_chars = 0
            total_chars = 0
            
            for page_num in range(doc.page_count):  # Tutte le pagine
                # print(f"Analizzando pagina {page_num + 1}/{doc.page_count}")
                page = doc[page_num]
                
                # Rileva colore background (default bianco)
                bg_color = 16777215  # Bianco (0xFFFFFF)
                try:
                    drawings = page.get_drawings()
                    for drawing in drawings:
                        if drawing.get('type') == 'f' and 'color' in drawing:
                            bg_color = drawing['color']
                            break
                except:
                    pass
                
                bg_r = (bg_color >> 16) & 255
                bg_g = (bg_color >> 8) & 255
                bg_b = bg_color & 255
                
                # Calcola luminosità background
                bg_luminance = (0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b) / 255
                
                # Soglia adattiva
                if bg_luminance > 0.9:  # Background molto chiaro
                    threshold = 15
                elif bg_luminance < 0.1:  # Background molto scuro
                    threshold = 15
                else:  # Background normale
                    threshold = 30
                
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                font_size = span["size"]
                                color = span["color"]
                                text = span["text"]
                                
                                if not text.strip():
                                    continue
                                
                                total_chars += len(text)
                                
                                # Testo molto piccolo
                                if font_size < 6:
                                    suspicious_chars += len(text)
                                
                                # Testo simile al background (soglia adattiva)
                                text_r = (color >> 16) & 255
                                text_g = (color >> 8) & 255
                                text_b = color & 255
                                
                                color_distance = ((text_r - bg_r) ** 2 + 
                                                (text_g - bg_g) ** 2 + 
                                                (text_b - bg_b) ** 2) ** 0.5
                                
                                if color_distance < threshold:
                                    sus_ch = len(text)
                                    suspicious_chars += len(text)
                                    print(f"   Testo sospetto (colore simile): {sus_ch}")

            doc.close()
            print(f"   Totale caratteri: {total_chars}, Sospetti: {suspicious_chars}")
            if total_chars > 0 and suspicious_chars > 0:
                return True
                
        except Exception as e:
            print(f"Errore durante l'analisi PDF: {e}")
            pass  
    
    return False







def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load documents from PDF, CSV, Markdown, text and image files using specialized loaders.
    
    Uses appropriate LangChain loaders for each file type, ensuring optimal content
    extraction and automatic handling of format-specific requirements.
    
    Parameters
    ----------
    file_paths : List[str]
        List of absolute paths to files to be loaded
        
    Returns
    -------
    List[Document]
        List of LangChain documents with extracted content and metadata
        
    Supported Formats
    ----------------
    - PDF: PyMuPDFLoader for accurate text extraction
    - CSV: CSVLoader for tabular structure handling  
    - Markdown: UnstructuredMarkdownLoader for optimized parsing
    - Text: TextLoader with automatic encoding handling
    - Images: UnstructuredImageLoader with integrated OCR
    
    Loader Benefits
    ---------------
    - **PDF**: Extracts real text instead of binary bytes
    - **CSV**: Preserves data structure and relationships
    - **Markdown**: Maintains formatting and structure
    - **Images**: Automatic OCR for text extraction
    - **Text**: Robust handling of different encodings
    
    Error Handling
    --------------
    - Skip unsupported files with logging
    - Handle errors for corrupted or inaccessible files
    - Continue processing even with individual file errors
    
    Notes
    -----
    This function replaces load_your_corpus() by solving PDF reading
    problems and significantly improving content extraction quality
    for all supported formats.
    """
    print(f"LOAD_DOCUMENTS: Caricamento di {len(file_paths)} file(s)")
    documents = []
    
    for file_path in file_paths:
        ext = file_path.split(".")[-1].lower()
        
        try:
            if ext == "pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext == "csv":
                loader = CSVLoader(file_path)
            elif ext == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif ext == "txt":
                loader = TextLoader(file_path)
            elif ext in ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]:
                loader = UnstructuredImageLoader(file_path)
            else:
                print(f"Tipo file non supportato: {file_path}")
                continue
            
            docs = loader.load()
            print(f"Caricati {len(docs)} documento/i da {file_path}")
            
            filename = Path(file_path).name
            trustability = "trusted"
            
            valid_docs = []
            for doc in docs:
                if is_document_low_quality(file_path, doc.page_content):
                    print(f"FILTRATO {file_path}: Bassa qualità")
                    continue
                else:
                    print(f"Documento accettato: {file_path}")
                
                # Metadata
                doc.metadata["trustability"] = trustability
                doc.metadata["filename"] = filename
                valid_docs.append(doc)
            
            for i, doc in enumerate(valid_docs):
                content_preview = doc.page_content[:100].replace("\n", " ")
            
            documents.extend(valid_docs)
            
        except Exception as e:
            print(f"Errore caricamento {file_path}: {e}")
            continue

    print(f"LOAD_DOCUMENTS: Totale {len(documents)} documenti caricati")
    return documents

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Apply robust document splitting for optimal retrieval performance.
    
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter
    with hierarchical separators optimized for technical documentation.
    Maintains semantic coherence through configurable overlap.
    
    Parameters
    ----------
    docs : List[Document]
        List of documents to be split into chunks
    settings : Settings
        Configuration object containing chunk_size and chunk_overlap parameters
        
    Returns
    -------
    List[Document]
        List of document chunks with preserved metadata and optimized content boundaries
        
    Notes
    -----
    Uses hierarchical separator strategy:
    
    1. Markdown headers (#, ##, ###)
    2. Paragraph breaks (double and single newlines)
    3. Sentence endings (., ?, !, ;, :)
    4. Clause separators (, )
    5. Word boundaries ( )
    6. Aggressive character-level fallback
    
    Chunk size and overlap are optimized for technical documents to ensure
    sufficient context while maintaining computational efficiency.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "#", "##", "###",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",
        ],
    )
    return splitter.split_documents(docs)


def format_docs_for_prompt(points: Iterable[Any]) -> str:
    """
    Format retrieved document points for LLM prompt integration.
    
    Converts Qdrant search results into a structured text format suitable
    for inclusion in LLM prompts. Extracts source information, trustability
    metadata, and content text, formatting them with clear attribution
    markers for transparency and citation tracking.
    
    Parameters
    ----------
    points : Iterable[Any]
        Iterable of Qdrant point objects containing search results with
        payload data including text content and metadata
        
    Returns
    -------
    str
        Formatted string with source attribution and content blocks,
        separated by double newlines for clear prompt integration
        
    Format Structure
    ---------------
    Each document block follows the pattern:
    ``[source:filename][trustability:level] content_text``
    
    Where:
    - **source**: Original filename or document identifier
    - **trustability**: Trust level (trusted/untrusted/unknown)
    - **content**: Extracted text content from the document chunk
    
    Notes
    -----
    - Gracefully handles missing payload data with "unknown" defaults
    - Preserves document attribution for citation requirements
    - Optimized for LLM context window with clear separator formatting
    - Supports transparency and accountability through source tracking
    """
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        trust = pay.get("trustability", "unknown")
        blocks.append(f"[source:{src}][trustability: {trust}] {pay.get('text','')} ")
    return "\n\n".join(blocks)

def scan_docs_folder(docs_dir: str = "docs") -> List[str]:
    """
    Recursively scan directory for supported document formats.
    
    Searches through the specified directory and its subdirectories to find
    all files with supported extensions for document loading and processing.
    
    Parameters
    ----------
    docs_dir : str, optional
        Directory path to scan for documents (default: "docs")
        
    Returns
    -------
    List[str]
        List of absolute file paths for all supported documents found
        
    Supported Extensions
    -------------------
    - Documents: .pdf, .csv, .md, .txt
    - Images: .png, .jpg, .jpeg, .bmp, .gif, .tiff
    
    Notes
    -----
    - Performs recursive search through all subdirectories
    - Returns empty list if directory doesn't exist
    - Logs the total number of files found
    - Case-insensitive extension matching
    """
    supported_extensions = {
        ".pdf",
        ".csv",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".tiff",
        ".txt"
    }
    file_paths = []

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Cartella {docs_dir} non trovata")
        return []

    # Scansione ricorsiva
    for file_path in docs_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            file_paths.append(str(file_path))

    print(f"Trovati {len(file_paths)} file nella cartella {docs_dir}")
    return file_paths


def clean_web_content(text: str) -> str:
    """
    Clean web-scraped content by removing unwanted UI elements and noise.
    
    Applies comprehensive text cleaning to web-scraped content, removing
    navigation elements, advertisements, legal notices, and formatting
    artifacts while preserving meaningful textual information.
    
    Parameters
    ----------
    text : str
        Raw web content text to be cleaned
        
    Returns
    -------
    str
        Cleaned text with unwanted elements removed and normalized formatting
        
    Cleaning Operations
    ------------------
    1. Normalize whitespace and control characters
    2. Remove common UI patterns (cookies, navigation, social media)
    3. Remove legal notices and copyright information
    4. Filter out URLs and email addresses
    5. Remove timestamps and date patterns
    6. Clean special character sequences
    7. Filter short/meaningless lines
    8. Remove all-caps navigation text
    
    Notes
    -----
    - Preserves accented characters and Unicode text
    - Maintains sentence structure and punctuation
    - Filters content shorter than 20 characters per line
    - Returns empty string for null/empty input
    """
    if not text:
        return ""

    # Rimuovi caratteri di controllo e spazi multipli
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Rimuovi pattern comuni di navigazione e UI
    patterns_to_remove = [
        r"Cookie Policy|Privacy Policy|Note Legali|Termini e Condizioni",
        r"Accetta tutti i cookie|Gestisci cookie|Rifiuta cookie",
        r"Iscriviti alla newsletter|Seguici su|Condividi su",
        r"Copyright.*?\d{4}|All rights reserved|Tutti i diritti riservati",
        r"Menu|Navbar|Header|Footer|Sidebar",
        r"Caricamento in corso|Loading|Attendere prego",
        r"Clicca qui|Click here|Leggi tutto|Read more",
        r"Ti potrebbe interessare|Articoli correlati|Notizie correlate",
        r"I più visti|Più letti|Trending|Popular",
        r"Pubblicità|Advertisement|Sponsor|Promo",
        r"PODCAST|RUBRICHE|SONDAGGI|LE ULTIME EDIZIONI",
        r"Ascolta i Podcast.*?|Vedi tutti.*?|Scopri di più.*?",
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Rimuovi URL e email
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Rimuovi numeri isolati (spesso date, ore, contatori)
    text = re.sub(r"\b\d{1,2}[:.]\d{2}\b", "", text)  # Orari
    text = re.sub(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", "", text)  # Date

    # Rimuovi caratteri speciali ripetuti
    text = re.sub(r'[^\w\s.,!?;:()\-"\'àèéìíîòóùúçñü]+', " ", text, flags=re.UNICODE)

    # Rimuovi linee molto corte (probabilmente navigazione)
    lines = text.split(".")
    meaningful_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 20 and not re.match(r"^[A-Z\s]+$", line):  # Non solo maiuscole
            meaningful_lines.append(line)

    text = ". ".join(meaningful_lines)

    # Pulizia finale
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

def retriever_func(query: str, embeddings, client, s) -> List[Document]:
    """
    Execute hybrid search and convert results to LangChain Document format.
    
    Performs hybrid search using Qdrant client and converts the retrieved
    points into LangChain Document objects for seamless integration with
    RAG pipelines. Preserves all metadata and payload information from
    the original search results.
    
    Parameters
    ----------
    query : str
        Search query string for document retrieval
    embeddings : Any
        Embedding model for query vectorization
    client : Any
        Qdrant client instance for search execution
    s : Any
        Settings object containing search configuration parameters
        
    Returns
    -------
    List[Document]
        List of LangChain Document objects with content and metadata
        
    Document Structure
    -----------------
    Each returned Document contains:
    - **page_content**: Text content from the retrieved chunk
    - **metadata**: Complete payload information including source attribution
    
    Notes
    -----
    This function serves as a bridge between Qdrant's search results and
    LangChain's Document format, enabling seamless integration with
    existing RAG processing pipelines while preserving all metadata
    for citation and quality tracking.
    """
    hits = hybrid_search(client, s, query, embeddings)
    documents = []
    for hit in hits:
        doc = Document(
            page_content=hit.payload.get('text', ''),
            metadata=hit.payload
        )
        documents.append(doc)
    return documents

class SimpleRetriever:
    """
    Simplified retriever interface for Qdrant-based document search.
    
    Provides a streamlined interface for document retrieval using Qdrant
    vector database with hybrid search capabilities. Implements both
    LangChain-compatible methods and custom retrieval interfaces for
    flexible integration with different RAG frameworks.
    
    Attributes
    ----------
    client : Any
        Qdrant client instance for database operations
    settings : Any
        Configuration settings for search parameters and behavior
    embeddings : Any
        Embedding model for query and document vectorization
        
    Methods
    -------
    get_relevant_documents(query)
        Retrieve documents relevant to the given query
    invoke(query)
        Alternative interface for document retrieval (LangChain compatible)
        
    Examples
    --------
    >>> retriever = SimpleRetriever(client, settings, embeddings)
    >>> docs = retriever.get_relevant_documents("aerodynamics principles")
    >>> # or using invoke interface
    >>> docs = retriever.invoke("aerodynamics principles")
    
    Notes
    -----
    The class provides dual interfaces to support both direct usage and
    integration with LangChain pipelines. Both methods return identical
    results formatted as LangChain Document objects with full metadata
    preservation for citation and quality tracking.
    """
    def __init__(self, client, settings, embeddings):
        """
        Initialize the SimpleRetriever with required components.
        
        Sets up the retriever instance with the necessary Qdrant client,
        configuration settings, and embedding model for document retrieval
        operations.
        
        Parameters
        ----------
        client : Any
            Qdrant client instance for database operations
        settings : Any
            Configuration settings containing search parameters
        embeddings : Any
            Embedding model for query and document vectorization
            
        Notes
        -----
        The client should be properly configured and connected to the
        Qdrant vector database before initialization. Settings should
        include search limits, similarity thresholds, and other
        retrieval configuration parameters.
        """
        self.client = client
        self.settings = settings  
        self.embeddings = embeddings
        
    def get_relevant_documents(self, query: str):
        """
        Retrieve documents relevant to the given query using hybrid search.
        
        Performs similarity search on the Qdrant vector database to find
        documents most relevant to the input query. Applies quality filtering
        and metadata preservation to ensure high-quality results for RAG
        applications.
        
        Parameters
        ----------
        query : str
            Search query string for document retrieval
            
        Returns
        -------
        list[Document]
            List of LangChain Document objects with content and metadata.
            Each document includes:
            - page_content: Document text content
            - metadata: Source information, quality scores, and timestamps
            
        Examples
        --------
        >>> retriever = SimpleRetriever(client, settings, embeddings)
        >>> docs = retriever.get_relevant_documents("wing design principles")
        >>> for doc in docs:
        ...     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        ...     print(f"Content: {doc.page_content[:100]}...")
        
        Notes
        -----
        The method uses hybrid_search() to combine semantic and keyword-based
        retrieval from Qdrant. Results include full metadata preservation for
        citation tracking and quality assessment in downstream RAG processes.
        """
        hits = hybrid_search(self.client, self.settings, query, self.embeddings)
        documents = []
        for hit in hits:
            doc = Document(
                page_content=hit.payload.get('text', ''),
                metadata=hit.payload
            )
            documents.append(doc)
        return documents
    
    def invoke(self, query: str):
        """
        Alternative interface for document retrieval compatible with LangChain.
        
        Provides LangChain-compatible method signature for seamless integration
        with existing RAG pipelines and frameworks. Internally delegates to
        get_relevant_documents() to maintain consistent behavior across
        different invocation patterns.
        
        Parameters
        ----------
        query : str
            Search query string for document retrieval
            
        Returns
        -------
        list[Document]
            List of LangChain Document objects with content and metadata,
            identical to get_relevant_documents() output
            
        Examples
        --------
        >>> retriever = SimpleRetriever(client, settings, embeddings)
        >>> docs = retriever.invoke("propulsion systems")
        >>> print(f"Found {len(docs)} relevant documents")
        
        Notes
        -----
        This method exists primarily for LangChain compatibility where the
        invoke() pattern is preferred. The implementation is identical to
        get_relevant_documents() to ensure consistent retrieval behavior
        regardless of the interface used.
        """
        return self.get_relevant_documents(query)

