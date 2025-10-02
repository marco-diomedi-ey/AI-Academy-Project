from typing import Type
import os
import yaml
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool


class TrustedWebSearchInput(BaseModel):
    """
    Input model for TrustedWebSearch tool defining required search parameters.
    
    This class defines the schema for input parameters expected by the
    TrustedWebSearch tool, specifically the search query string that will
    be used to search trusted web sources.
    
    Attributes
    ----------
    search_query : str
        The search query string to be used for searching trusted web sources
    """
    search_query: str = Field(..., description="Search query for trusted web sources")

class TrustedWebSearch(BaseTool):
    """
    A custom web search tool that filters results to only include trusted domains.
    
    This tool extends the CrewAI BaseTool to provide web search functionality
    that only returns results from a predefined list of trusted domains. It uses
    the SerperDev API for web searches and applies domain filtering to ensure
    only reliable sources are included in the results.
    
    Attributes
    ----------
    name : str
        The name identifier for this tool
    description : str
        Brief description of the tool's functionality
    args_schema : Type[BaseModel]
        The Pydantic model defining input schema for this tool
    serper_tool : SerperDevTool
        The underlying SerperDev tool instance for web searching
    trusted_domains : list
        List of trusted domain names to filter search results
        
    Methods
    -------
    _load_trusted_domains() -> list
        Load trusted domains from YAML configuration file
    _get_default_domains() -> list
        Get default trusted domains as fallback
    _is_trusted_domain(url: str) -> bool
        Check if a URL belongs to a trusted domain
    _process_organic_results(organic_results: list) -> list
        Filter organic search results by trusted domains
    _process_people_also_ask(paa_results: list) -> list
        Filter People Also Ask results by trusted domains
    _process_knowledge_graph(kg: dict) -> dict
        Filter Knowledge Graph data by trusted domains
    _process_related_searches(related_results: list) -> list
        Process related search suggestions
    _format_output(trusted_data: dict, search_params: dict) -> str
        Format filtered results into readable output
    _run(search_query: str) -> str
        Execute the search and return filtered results
    """
    name: str = "Trusted Web Search"
    description: str = "Search web using only trusted domains"
    args_schema: Type[BaseModel] = TrustedWebSearchInput
    serper_tool: SerperDevTool = None
    trusted_domains: list = None
    
    def _load_trusted_domains(self) -> list:
        """
        Load trusted domains from YAML configuration file.
        
        This method loads a list of trusted domain names from a YAML configuration
        file located in the crews/web_crew/config/domains.yaml path. If the file
        cannot be loaded or parsed, it falls back to default domains.
        
        Returns
        -------
        list
            List of trusted domain names as strings. Returns default domains
            if YAML file cannot be loaded or parsed
            
        Raises
        ------
        FileNotFoundError
            If the domains.yaml file is not found (handled gracefully)
        yaml.YAMLError
            If the YAML file cannot be parsed (handled gracefully)
        """
        # Percorso relativo al file YAML dalla posizione corrente
        yaml_path = Path(__file__).parent.parent / "crews" / "web_crew" / "config" / "domains.yaml"
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                domains = config.get('trusted_domains', [])
                # print(f"Caricati {len(domains)} domini trusted da {yaml_path}")
                return domains
        except FileNotFoundError:
            print(f"File {yaml_path} non trovato, uso domini di default")
            return self._get_default_domains()
        except yaml.YAMLError as e:
            print(f"Errore parsing YAML {yaml_path}: {e}, uso domini di default")
            return self._get_default_domains()
        except Exception as e:
            print(f"Errore caricamento domini: {e}, uso domini di default")
            return self._get_default_domains()
    
    def _get_default_domains(self) -> list:
        """
        Get default trusted domains as fallback when YAML loading fails.
        
        This method provides a hardcoded list of reliable, well-known domains
        that serve as a fallback when the YAML configuration file cannot be
        loaded or parsed. These domains are selected for their reliability
        and trustworthiness.
        
        Returns
        -------
        list
            List of default trusted domain names including Wikipedia, GitHub,
            Stack Overflow, Python.org, Nature.com, and Italian government sites
        """
        return [
            'wikipedia.org', 'github.com', 'stackoverflow.com', 
            'python.org', 'nature.com', 'gov.it'
        ]
    
    def __init__(self, api_key: str, n_results: int = 10):
        """
        Initialize the TrustedWebSearch tool with API configuration.
        
        This constructor sets up the TrustedWebSearch tool by configuring
        the SerperDev API key, initializing the underlying SerperDevTool,
        and loading the trusted domains list from configuration.
        
        Parameters
        ----------
        api_key : str
            The API key for SerperDev service used for web searches
        n_results : int, default=10
            Maximum number of search results to retrieve from SerperDev
            
        Notes
        -----
        The API key is automatically set as an environment variable
        that SerperDevTool expects. Trusted domains are loaded from
        YAML configuration or default fallback list.
        """
        super().__init__()
        # Imposta la variabile d'ambiente che SerperDevTool si aspetta
        import os
        os.environ["SERPER_API_KEY"] = api_key
        
        # Inizializza SerperDevTool con parametri completi
        self.serper_tool = SerperDevTool(n_results=n_results)
        # Carica domini trusted dal file YAML
        self.trusted_domains = self._load_trusted_domains()
    
    def _is_trusted_domain(self, url: str) -> bool:
        """
        Check if a URL belongs to a trusted domain.
        
        This method verifies whether the given URL contains any of the
        trusted domain names loaded from the configuration. It performs
        a substring match to determine if the URL is from a trusted source.
        
        Parameters
        ----------
        url : str
            The URL to check against the trusted domains list
            
        Returns
        -------
        bool
            True if the URL contains any trusted domain name, False otherwise
            
        Notes
        -----
        Uses substring matching, so 'example.com' will match URLs like
        'https://subdomain.example.com/path'
        """
        return any(domain in url for domain in self.trusted_domains)
    
    def _process_organic_results(self, organic_results: list) -> list:
        """
        Process organic search results filtering by trusted domains.
        
        This method filters the organic search results from SerperDev API
        to include only those from trusted domains. It preserves the structure
        of each result while filtering out untrusted sources and their sitelinks.
        
        Parameters
        ----------
        organic_results : list
            List of organic search result dictionaries from SerperDev API
            
        Returns
        -------
        list
            Filtered list containing only results from trusted domains,
            each with title, link, snippet, position, and optional sitelinks
            
        Notes
        -----
        Sitelinks within each result are also filtered to include only
        those from trusted domains
        """
        trusted_organic = []
        for result in organic_results:
            link = result.get("link", "")
            if self._is_trusted_domain(link):
                trusted_result = {
                    "title": result.get("title", ""),
                    "link": link,
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", ""),
                }
                
                # Aggiungi sitelinks se presenti e trusted
                if "sitelinks" in result:
                    trusted_sitelinks = []
                    for sitelink in result["sitelinks"]:
                        if self._is_trusted_domain(sitelink.get("link", "")):
                            trusted_sitelinks.append({
                                "title": sitelink.get("title", ""),
                                "link": sitelink.get("link", "")
                            })
                    if trusted_sitelinks:
                        trusted_result["sitelinks"] = trusted_sitelinks
                
                trusted_organic.append(trusted_result)
        return trusted_organic
    
    def _process_people_also_ask(self, paa_results: list) -> list:
        """
        Process People Also Ask results filtering by trusted domains.
        
        This method filters the "People Also Ask" section results from
        SerperDev API to include only those from trusted domains or those
        without links (general knowledge questions).
        
        Parameters
        ----------
        paa_results : list
            List of People Also Ask result dictionaries from SerperDev API
            
        Returns
        -------
        list
            Filtered list containing only PAA results from trusted domains
            or without links, each with question, snippet, title, and link
            
        Notes
        -----
        Results without links are included as they typically represent
        general knowledge that doesn't require source verification
        """
        trusted_paa = []
        for result in paa_results:
            link = result.get("link", "")
            if not link or self._is_trusted_domain(link):
                trusted_paa.append({
                    "question": result.get("question", ""),
                    "snippet": result.get("snippet", ""),
                    "title": result.get("title", ""),
                    "link": link
                })
        return trusted_paa
    
    def _process_knowledge_graph(self, kg: dict) -> dict:
        """
        Process Knowledge Graph data if from trusted source.
        
        This method processes the Knowledge Graph information from SerperDev API,
        checking if the website source is from a trusted domain before including
        the data in the filtered results.
        
        Parameters
        ----------
        kg : dict
            Knowledge Graph data dictionary from SerperDev API containing
            title, type, website, description, and attributes
            
        Returns
        -------
        dict or None
            Filtered Knowledge Graph data with title, type, website, description,
            descriptionSource, descriptionLink, and attributes if from trusted
            domain, None otherwise
            
        Notes
        -----
        Only Knowledge Graph entries with websites from trusted domains
        are included in the output
        """
        kg_website = kg.get("website", "")
        if kg_website and self._is_trusted_domain(kg_website):
            return {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "website": kg_website,
                "description": kg.get("description", ""),
                "descriptionSource": kg.get("descriptionSource", ""),
                "descriptionLink": kg.get("descriptionLink", ""),
                "attributes": kg.get("attributes", {})
            }
        return None
    
    def _process_related_searches(self, related_results: list) -> list:
        """
        Process related search suggestions (always included).
        
        This method processes the related searches section from SerperDev API.
        Unlike other sections, related searches are always included as they
        represent search query suggestions rather than content from specific domains.
        
        Parameters
        ----------
        related_results : list
            List of related search result dictionaries from SerperDev API
            
        Returns
        -------
        list
            List of processed related search queries, each containing
            a query string extracted from the original results
            
        Notes
        -----
        Related searches are not filtered by trusted domains as they
        represent search suggestions rather than source content
        """
        return [{"query": result.get("query", "")} for result in related_results]
    
    def _format_output(self, trusted_data: dict, search_params: dict) -> str:
        """
        Format the final output with all trusted information.
        
        This method takes the filtered trusted data and formats it into a
        human-readable string output with proper sections for different types
        of search results including Knowledge Graph, organic results, People
        Also Ask, and related searches.
        
        Parameters
        ----------
        trusted_data : dict
            Dictionary containing filtered trusted data with keys like 'organic',
            'knowledgeGraph', 'peopleAlsoAsk', and 'relatedSearches'
        search_params : dict
            Search parameters dictionary from SerperDev API containing query
            and other search metadata
            
        Returns
        -------
        str
            Formatted string output with structured sections for different
            types of search results, including headers, result counts, and
            properly formatted content
            
        Notes
        -----
        Knowledge Graph results are prioritized and displayed first,
        followed by organic results, People Also Ask, and related searches
        """
        output_lines = []
        
        # Header con parametri di ricerca
        total_trusted = (
            len(trusted_data.get("organic", [])) +
            (1 if trusted_data.get("knowledgeGraph") else 0)
        )
        
        output_lines.append(f"TRUSTED SEARCH RESULTS")
        output_lines.append(f"Query: {search_params.get('q', '')}")
        output_lines.append(f"Total trusted sources found: {total_trusted}")
        output_lines.append("=" * 60)
        output_lines.append("")
        
        # Knowledge Graph (prioritario)
        if "knowledgeGraph" in trusted_data:
            kg = trusted_data["knowledgeGraph"]
            output_lines.append("KNOWLEDGE GRAPH")
            output_lines.append(f"**{kg.get('title', '')}** ({kg.get('type', '')})")
            output_lines.append(f"Source: {kg.get('website', '')}")
            output_lines.append(f"{kg.get('description', '')}")
            if kg.get('attributes'):
                output_lines.append("Key Attributes:")
                for key, value in kg.get('attributes', {}).items():
                    output_lines.append(f"   • {key}: {value}")
            output_lines.append("")
        
        # Risultati organici
        if "organic" in trusted_data:
            output_lines.append("ORGANIC RESULTS")
            for i, result in enumerate(trusted_data["organic"], 1):
                output_lines.append(f"{i}. **{result.get('title', '')}**")
                output_lines.append(f" {result.get('link', '')}")
                output_lines.append(f"{result.get('snippet', '')}")
                output_lines.append(f"Position: {result.get('position', 'N/A')}")
                
                # Sitelinks se presenti
                if result.get('sitelinks'):
                    output_lines.append("   🔗 Related links:")
                    for sitelink in result['sitelinks']:
                        output_lines.append(f"      • {sitelink.get('title', '')}: {sitelink.get('link', '')}")
                output_lines.append("")
        
        # People Also Ask
        if "peopleAlsoAsk" in trusted_data:
            output_lines.append("PEOPLE ALSO ASK")
            for i, paa in enumerate(trusted_data["peopleAlsoAsk"], 1):
                output_lines.append(f"{i}. Q: {paa.get('question', '')}")
                if paa.get('snippet'):
                    output_lines.append(f"   A: {paa.get('snippet', '')}")
                if paa.get('link'):
                    output_lines.append(f"  {paa.get('link', '')}")
                output_lines.append("")
        
        # Ricerche correlate
        if "relatedSearches" in trusted_data:
            output_lines.append("RELATED SEARCHES")
            for search in trusted_data["relatedSearches"]:
                output_lines.append(f"   • {search.get('query', '')}")
            output_lines.append("")
        
        return "\n".join(output_lines)
    
    def _run(self, search_query: str) -> str:
        """
        Execute the trusted web search and return filtered results.
        
        This method performs the main search operation by executing a search
        using SerperDevTool, filtering all results through trusted domain checks,
        and formatting the output. It processes all sections of search results
        including organic results, Knowledge Graph, People Also Ask, and related
        searches.
        
        Parameters
        ----------
        search_query : str
            The search query string to be executed
            
        Returns
        -------
        str
            Formatted string containing only results from trusted domains,
            or a message indicating no trusted sources were found
            
        Notes
        -----
        If no trusted sources are found, returns a message with the total
        number of available results and suggests expanding the trusted domains list
        """
        # Ricerca completa con SerperDevTool
        results = self.serper_tool._run(search_query=search_query)
        
        # Estrai dati trusted da tutte le sezioni
        trusted_data = {}
        
        # Processa risultati organici
        if "organic" in results:
            trusted_organic = self._process_organic_results(results["organic"])
            if trusted_organic:
                trusted_data["organic"] = trusted_organic
        
        # Processa Knowledge Graph
        if "knowledgeGraph" in results:
            trusted_kg = self._process_knowledge_graph(results["knowledgeGraph"])
            if trusted_kg:
                trusted_data["knowledgeGraph"] = trusted_kg
        
        # Processa People Also Ask
        if "peopleAlsoAsk" in results:
            trusted_paa = self._process_people_also_ask(results["peopleAlsoAsk"])
            if trusted_paa:
                trusted_data["peopleAlsoAsk"] = trusted_paa
        
        # Processa ricerche correlate (sempre incluse se presenti)
        if "relatedSearches" in results:
            trusted_data["relatedSearches"] = self._process_related_searches(results["relatedSearches"])
        
        # Formatta output finale
        if trusted_data:
            return self._format_output(trusted_data, results.get("searchParameters", {}))
        else:
            total_results = len(results.get("organic", []))
            return f"NO TRUSTED SOURCES FOUND\nTotal results available: {total_results}\nConsider expanding trusted domains list."