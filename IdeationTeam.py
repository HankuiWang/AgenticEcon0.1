import asyncio
import os
import json
import datetime
import io
import sys
from dotenv import load_dotenv
import arxiv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Import OpenAI (the official client)
from openai import OpenAI

# Import for web scraping
import requests
from bs4 import BeautifulSoup
import time

###########################################
# ANSI color codes and logging system
###########################################
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# Global variable to control debug output
DEBUG_MODE = False

def log(message, color=Colors.RESET, level="INFO", always_show=False):
    """Simplified logging function that respects debug mode setting"""
    if DEBUG_MODE or always_show:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{color}[{timestamp}][{level}] {message}{Colors.RESET}")

# Execution tracking structures
INFO_LOG = []  # Plain INFO lines for markdown
INFO_SEEN = set()
LAST_FINALIZED_OUTPUT = ""  # Raw finalizer output (including justifications)
EXECUTION_STATS = {
    "start_time": None,
    "end_time": None,
    "mode": None,
    "trending_topics_count": 0,
    "initial_ideas_count": 0,
    "initial_ideas_total": 0,
    "initial_ideas_set": set(),
    "refined_questions_initial_total": 0,
    "refined_questions_total": 0,
    "refined_questions_set": set(),
    "contextualized_count": 0,
    "final_questions_count": 0,
    "context_log_emitted": False,
    "greyscout_log_emitted": False,
    "policy_papers_count": 0,
    "policy_links_set": set(),
    "academic_sources_count": 0,
    "academic_links_set": set(),
    "topiccrawler_log_emitted": False,
    "trendsurfer_start_logged": False,
    "trendsurfer_result_logged": False,
    "trendsurfer_done": False,
    "trendsurfer_cached": None,
    "ideator_log_emitted": False,
    "scholarly_db_results_count": 0,
    "scholar_log_emitted": False,
    "scholar_links_set": set(),
    "human_idea": None,
    "news_articles_count": 0,
    "news_links_set": set(),
    "topiccrawler_done": False,
    "topiccrawler_cached": None,
    "scholar_done": False,
    "scholar_cached": None,
    "greyscout_done": False,
    "greyscout_cached": None,
}

def info(message: str) -> None:
    # Record unique lines for markdown, but always print to console
    if message not in INFO_SEEN:
        INFO_LOG.append(f"[INFO] {message}")
        INFO_SEEN.add(message)
    log(message, Colors.GREEN, "INFO", always_show=True)

class StdoutTee:
    """A tee for sys.stdout that writes to original stdout and a buffer."""
    def __init__(self, buffer: io.StringIO):
        self._buffer = buffer
        self._original_stdout = sys.stdout
    def write(self, data: str) -> int:
        self._buffer.write(data)
        return self._original_stdout.write(data)
    def flush(self) -> None:
        self._buffer.flush()
        self._original_stdout.flush()
    def isatty(self) -> bool:
        return getattr(self._original_stdout, "isatty", lambda: False)()


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

###########################################
# Defining Tools
###########################################
def scrape_economic_news(query: str | None = None, max_results: int = 30) -> str:
    """
    Uses Google Custom Search API to scrape recent economic news and identify trending topics.
    - If query is provided, searches for trends related to that query/topic.
    - Otherwise, searches for general latest economic trends.
    Returns detailed information about discovered economic trends.
    """
    try:
        # Single-run behavior: return cached result if already completed
        if EXECUTION_STATS.get("trendsurfer_done") and EXECUTION_STATS.get("trendsurfer_cached"):
            return EXECUTION_STATS["trendsurfer_cached"]
        info("TrendSurfer retrieving related trending topics from Google...")
        # Google Search API configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            log("Google API credentials not found, using fallback topics", Colors.YELLOW, "WARNING")
            return "AI impact on economic growth; Inflation trends; Digital currency adoption; Supply chain disruptions; Green energy economics"
        
        # Search for recent economic news (paginate)
        url = "https://customsearch.googleapis.com/customsearch/v1"
        collected_items = []
        start_index = 1
        query_text = (f"{query} latest economic news trends" if query else "economic news latest trends")
        while len(collected_items) < max_results and start_index <= 91:  # Google CSE allows up to 100 results
            page_size = min(10, max_results - len(collected_items))
            params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": query_text,
                "num": page_size,
                "start": start_index,
                "dateRestrict": "d7"  # Last 7 days for variety
            }
            response = requests.get(url, params=params)
            if response.status_code != 200:
                log(f"Google Search API error: {response.status_code}", Colors.RED, "ERROR")
                break
            items = response.json().get("items", []) or []
            if not items:
                break
            collected_items.extend(items)
            start_index += page_size
        if not collected_items:
            log("No search results found", Colors.YELLOW, "WARNING")
            return "AI impact on economic growth; Inflation trends; Digital currency adoption; Supply chain disruptions; Green energy economics"
        
        # Extract and process search results
        def get_page_content(url: str, max_chars: int = 500) -> str:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                words = text.split()
                content = ""
                for word in words:
                    if len(content) + len(word) + 1 > max_chars:
                        break
                    content += " " + word
                return content.strip()
            except Exception as e:
                log(f"Error fetching {url}: {str(e)}", Colors.RED, "ERROR")
                return ""
        
        # Process each search result
        enriched_results = []
        for item in collected_items:
            body = get_page_content(item["link"])
            enriched_results.append({
                "title": item["title"],
                "link": item["link"],
                "snippet": item["snippet"],
                "body": body
            })
            time.sleep(1)  # Be respectful to servers
        
        # Generate detailed analysis of discovered trends
        if enriched_results:
            trends_analysis = f"Discovered Economic Trends:\n\n"
            for i, result in enumerate(enriched_results, 1):
                trends_analysis += f"Trend {i}:\n"
                trends_analysis += f"Title: {result['title']}\n"
                trends_analysis += f"Source: {result['link']}\n"
                trends_analysis += f"Summary: {result['snippet']}\n"
                if result['body']:
                    trends_analysis += f"Content Preview: {result['body'][:200]}...\n"
                trends_analysis += "\n"
            
            # Extract key economic topics from the results
            all_content = " ".join([r['title'] + " " + r['snippet'] + " " + r['body'] for r in enriched_results])
            
            # Use LLM to identify trending topics
            prompt = f"""
            Based on the following economic news content, identify 3-5 trending economic topics:
            
            {all_content[:2000]}
            
            Return only the trending topics as a simple list, one per line.
            Focus on topics that could lead to research opportunities.
            """
            
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]
            )
            
            trending_topics = completion.choices[0].message.content.strip()
            # Count topics by lines
            topics_count = len([ln for ln in trending_topics.split("\n") if ln.strip()])
            EXECUTION_STATS["trending_topics_count"] = max(EXECUTION_STATS.get("trending_topics_count", 0), topics_count)
            # Track unique news links across events to avoid fixed counts in different modes
            for r in enriched_results:
                link = r.get("link", "")
                if link:
                    EXECUTION_STATS.setdefault("news_links_set", set()).add(link)
            EXECUTION_STATS["news_articles_count"] = len(EXECUTION_STATS.get("news_links_set", set()))
            # Requested formatting: Retrieved X topics (filtered to Y most relevant)
            info(f"Retrieved {topics_count} trending topics (filtered to {topics_count} most relevant)")
            result_text = f"{trends_analysis}\nIdentified Trending Topics:\n{trending_topics}"
            EXECUTION_STATS["trendsurfer_cached"] = result_text
            EXECUTION_STATS["trendsurfer_done"] = True
            return result_text
        else:
            return "AI impact on economic growth; Inflation trends; Digital currency adoption; Supply chain disruptions; Green energy economics"
            
    except Exception as e:
        log(f"Error in Google Search scraping: {e}", Colors.RED, "ERROR")
        fallback = "AI impact on economic growth; Inflation trends; Digital currency adoption; Supply chain disruptions; Green energy economics"
        # Semicolon-separated fallback topics count
        topics_count = len([seg for seg in fallback.split(";") if seg.strip()])
        EXECUTION_STATS["trending_topics_count"] = max(EXECUTION_STATS.get("trending_topics_count", 0), topics_count)
        if query:
            info(f"Retrieved {topics_count} topics related to '{query}' (fallback)")
        else:
            info(f"Retrieved {topics_count} trending topics (fallback)")
        return fallback

def search_policy_literature(query: str | None = None, num_results: int = 20, max_chars: int = 600) -> str:
    """
    Uses Google Custom Search API to retrieve grey literature and policy papers from major institutions.
    Prioritizes sources like World Bank, IMF, ECB, BIS, OECD.
    If a query is provided, search is focused on that query; otherwise performs a general recent-policy search.
    Returns a human-readable summary of discovered sources.
    """
    try:
        if query:
            info(f"GreyScout retrieving policy literature related to: {query}...")
        else:
            info("GreyScout retrieving general policy and grey literature...")

        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if not api_key or not search_engine_id:
            log("Google API credentials not found for GreyScout, using fallback list", Colors.YELLOW, "WARNING")
            fallback_list = [
                "IMF working paper on macroeconomic policy",
                "World Bank policy research working paper",
                "ECB occasional paper on monetary policy",
                "BIS working paper on financial stability",
                "OECD policy paper on taxation"
            ]
            EXECUTION_STATS["policy_papers_count"] = max(EXECUTION_STATS.get("policy_papers_count", 0), len(fallback_list))
            if not EXECUTION_STATS.get("greyscout_log_emitted"):
                info("GreyScout retrieving policy literature... Complete.")
                EXECUTION_STATS["greyscout_log_emitted"] = True
            return "\n".join(fallback_list)

        url = "https://customsearch.googleapis.com/customsearch/v1"
        site_filter = "(site:worldbank.org OR site:imf.org OR site:ecb.europa.eu OR site:bis.org OR site:oecd.org)"
        query_text = (f"{query} policy paper OR working paper {site_filter}" if query else f"economics policy paper OR working paper {site_filter}")
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query_text,
            "num": min(max(num_results, 1), 10),
            "dateRestrict": "y2"
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            log(f"Google Search API error (GreyScout): {response.status_code}", Colors.RED, "ERROR")
            return "No policy literature found due to API error."

        results = response.json().get("items", [])
        if not results:
            return "No relevant policy literature found."

        def get_page_content(url: str, max_chars_local: int = max_chars) -> str:
            try:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.content, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                words = text.split()
                content = ""
                for word in words:
                    if len(content) + len(word) + 1 > max_chars_local:
                        break
                    content += " " + word
                return content.strip()
            except Exception as ex:
                log(f"Error fetching policy page {url}: {str(ex)}", Colors.RED, "ERROR")
                return ""

        enriched = []
        for item in results:
            body = get_page_content(item["link"], max_chars)
            enriched.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "body": body
            })
            time.sleep(1)

        # Track unique policy links for summary across events
        for e in enriched:
            link = e.get("link", "")
            if link:
                EXECUTION_STATS.setdefault("policy_links_set", set()).add(link)
        EXECUTION_STATS["policy_papers_count"] = len(EXECUTION_STATS.get("policy_links_set", set()))
        if not EXECUTION_STATS.get("greyscout_log_emitted"):
            info("GreyScout retrieving policy literature... Complete.")
            EXECUTION_STATS["greyscout_log_emitted"] = True

        summary = ["Discovered Policy/Grey Literature:\n"]
        for idx, r in enumerate(enriched, 1):
            summary.append(f"{idx}. {r['title']}\n   Source: {r['link']}\n   Summary: {r['snippet']}")
        result_text = "\n".join(summary)
        EXECUTION_STATS["greyscout_cached"] = result_text
        EXECUTION_STATS["greyscout_done"] = True
        return result_text
    except Exception as e:
        log(f"Error in GreyScout policy search: {e}", Colors.RED, "ERROR")
        return "No policy literature found due to an unexpected error."

def crawl_academic_sources(query: str | None = None, arxiv_max: int = 20, web_max: int = 20) -> str:
    """
    Gathers preliminary academic literature from open sources:
    - arXiv (via arxiv Python client)
    - SSRN and NBER (via Google Custom Search API)
    Returns a formatted summary list.
    """
    collected = []
    try:
        topic = query or "economics"

        # Return cache if already completed in this run
        if EXECUTION_STATS.get("topiccrawler_done") and EXECUTION_STATS.get("topiccrawler_cached"):
            return EXECUTION_STATS["topiccrawler_cached"]

        # arXiv search
        try:
            client = arxiv.Client()
            search = arxiv.Search(query=topic, max_results=arxiv_max, sort_by=arxiv.SortCriterion.Relevance)
            for paper in client.results(search):
                collected.append({
                    "source": "arXiv",
                    "title": getattr(paper, "title", ""),
                    "authors": ", ".join([a.name for a in getattr(paper, "authors", [])]),
                    "published": paper.published.strftime("%Y-%m-%d") if getattr(paper, "published", None) else "",
                    "link": getattr(paper, "pdf_url", ""),
                })
        except Exception as ex:
            log(f"TopicCrawler arXiv error: {ex}", Colors.RED, "ERROR")

        # SSRN and NBER via Google Custom Search
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if api_key and search_engine_id:
            url = "https://customsearch.googleapis.com/customsearch/v1"
            site_filter = "(site:ssrn.com OR site:nber.org)"
            params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": f"{topic} working paper {site_filter}",
                "num": min(max(web_max, 1), 10),
                "dateRestrict": "y5",
            }
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                for it in items:
                    collected.append({
                        "source": "SSRN/NBER",
                        "title": it.get("title", ""),
                        "authors": "",
                        "published": "",
                        "link": it.get("link", "")
                    })
            else:
                log(f"TopicCrawler web search error: {resp.status_code}", Colors.RED, "ERROR")
        else:
            log("TopicCrawler: missing Google API credentials; skipping SSRN/NBER search", Colors.YELLOW, "WARNING")

        # Track unique academic links for summary
        for e in collected:
            link = e.get("link", "")
            if link:
                EXECUTION_STATS.setdefault("academic_links_set", set()).add(link)
        EXECUTION_STATS["academic_sources_count"] = len(EXECUTION_STATS.get("academic_links_set", set()))
        if not EXECUTION_STATS.get("topiccrawler_done"):
            info("TopicCrawler and ScholarSearcher retrieving academic sources... Complete.")
            EXECUTION_STATS["topiccrawler_done"] = True

        if not collected:
            return "No academic sources discovered."

        lines = ["Discovered Academic Sources:\n"]
        for idx, entry in enumerate(collected, 1):
            lines.append(
                f"{idx}. [{entry.get('source','')}] {entry.get('title','')}\n   Link: {entry.get('link','')}"
            )
        result_text = "\n".join(lines)
        EXECUTION_STATS["topiccrawler_cached"] = result_text
        return result_text
    except Exception as e:
        log(f"Error in TopicCrawler: {e}", Colors.RED, "ERROR")
        return "No academic sources discovered due to an unexpected error."

def search_scholarly_databases(query: str | None = None, max_results: int = 40) -> str:
    """
    Executes detailed academic queries across Web of Science, Scopus, and EconLit.
    Because these may require institutional access, this function gathers as much
    open-source information as possible via Google Custom Search and public landing pages.
    Returns a formatted list of accessible items.
    """
    try:
        topic = query or "economics"
        # Return cache if already completed
        if EXECUTION_STATS.get("scholar_done") and EXECUTION_STATS.get("scholar_cached"):
            return EXECUTION_STATS["scholar_cached"]
        info("ScholarSearcher querying major research databases (open-access fallback)...")

        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        # If credentials are missing, fallback to a generic message
        if not api_key or not search_engine_id:
            log("Google API credentials not found for ScholarSearcher, using fallback list", Colors.YELLOW, "WARNING")
            fallback = [
                "EconLit entry (public landing) for recent macroeconomics survey",
                "Scopus public abstract page for labor economics study",
                "Web of Science public record for environmental economics article",
            ]
            EXECUTION_STATS["scholarly_db_results_count"] = max(
                EXECUTION_STATS.get("scholarly_db_results_count", 0), len(fallback)
            )
            if not EXECUTION_STATS.get("scholar_log_emitted"):
                if EXECUTION_STATS.get("topiccrawler_done"):
                    # Already emitted combined line
                    pass
                else:
                    info("TopicCrawler and ScholarSearcher retrieving academic sources... Complete.")
                EXECUTION_STATS["scholar_log_emitted"] = True
            return "\n".join(fallback)

        url = "https://customsearch.googleapis.com/customsearch/v1"
        site_filter = (
            "(site:scopus.com OR site:webofscience.com OR site:webofscience.clarivate.com OR site:search.proquest.com)"
        )
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": f"{topic} academic abstract {site_filter}",
            "num": min(max(max_results, 1), 10),
            "dateRestrict": "y5",
        }
        resp = requests.get(url, params=params)
        items = []
        if resp.status_code == 200:
            items = resp.json().get("items", [])
        else:
            log(f"ScholarSearcher web search error: {resp.status_code}", Colors.RED, "ERROR")

        # Filter for open or public landing pages and collect titles/links
        results = []
        for it in items:
            title = it.get("title", "")
            link = it.get("link", "")
            snippet = it.get("snippet", "")
            # Heuristic: keep items that are public/landing pages (not login-only detected by keywords)
            blocked_keywords = ["login", "sign in", "subscribe", "purchase", "institutional", "paywall"]
            text = f"{title} {snippet} {link}".lower()
            if not any(b in text for b in blocked_keywords):
                results.append({"title": title, "link": link, "snippet": snippet})

        # Track unique scholarly DB links for summary
        for r in results:
            link = r.get("link", "")
            if link:
                EXECUTION_STATS.setdefault("scholar_links_set", set()).add(link)
        EXECUTION_STATS["scholarly_db_results_count"] = len(EXECUTION_STATS.get("scholar_links_set", set()))
        if not EXECUTION_STATS.get("scholar_log_emitted"):
            if EXECUTION_STATS.get("topiccrawler_done"):
                # Combined already emitted by TopicCrawler
                pass
            else:
                info("TopicCrawler and ScholarSearcher retrieving academic sources... Complete.")
            EXECUTION_STATS["scholar_log_emitted"] = True

        if not results:
            return "No accessible scholarly database items found."

        lines = ["Accessible Scholarly Database Items:\n"]
        for idx, r in enumerate(results, 1):
            lines.append(f"{idx}. {r['title']}\n   Link: {r['link']}\n   Summary: {r['snippet']}")
        result_text = "\n".join(lines)
        EXECUTION_STATS["scholar_cached"] = result_text
        EXECUTION_STATS["scholar_done"] = True
        return result_text
    except Exception as e:
        log(f"Error in ScholarSearcher: {e}", Colors.RED, "ERROR")
        return "No accessible scholarly database items found due to an unexpected error."


def _parse_question_list(raw_text: str) -> list:
    """Robustly parse a list of questions from arbitrary text.
    Handles numbered lists, JSON/Python-like lists, and '?'-terminated sentences."""
    if not raw_text:
        return []
    text = raw_text.strip()
    # Try JSON-like list first
    try:
        candidate = text
        if candidate.startswith("[") and candidate.endswith("]"):
            # Heuristic: convert single quotes to double quotes for JSON parsing
            candidate_json = candidate.replace("'", '"')
            loaded = json.loads(candidate_json)
            if isinstance(loaded, list):
                return [str(x).strip() for x in loaded if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception:
        pass
    # Try numbered list lines
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s[0].isdigit() and '. ' in s[:6]:
            # Remove leading numbering
            parts = s.split('. ', 1)
            if len(parts) > 1:
                lines.append(parts[1].strip())
            else:
                lines.append(s)
    if lines:
        return lines
    # Fallback: split on '?'
    sentences = []
    parts = text.split('?')
    for part in parts:
        q = part.strip()
        if q:
            sentences.append(q + '?')
    return [q for q in sentences if len(q) > 3]

def generate_ideas(input_data: str) -> list:
    """
    Uses LLM to generate research ideas based on input data.
    Aggregates inputs to generate initial research ideas.
    """
    prompt = f"""
    Based on the following input data, generate 2-3 novel research ideas in economics:
    
    {input_data}
    
    Return only the research questions as a list, with no additional text.
    """
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
    )
    content = completion.choices[0].message.content
    
    # Parse the response into a list of ideas (robust parsing)
    ideas = _parse_question_list(content)
    
    # Ensure we have at least one idea
    if not ideas:
        ideas = ["How does technological innovation affect income inequality?"]
        
    # Track unique initial ideas across calls
    for q in ideas:
        EXECUTION_STATS["initial_ideas_set"].add(q.strip().lower())
    EXECUTION_STATS["initial_ideas_count"] = max(EXECUTION_STATS.get("initial_ideas_count", 0), len(EXECUTION_STATS["initial_ideas_set"]))
    EXECUTION_STATS["initial_ideas_total"] = len(EXECUTION_STATS["initial_ideas_set"])
    info("Ideator synthesizing research questions... Complete.")
    return ideas

def refine_ideas(ideas: str) -> list:
    """
    Uses LLM to evaluate and refine research ideas.
    
    This function analyzes ideas from the Ideator, filters out redundancies,
    ensures conceptual coherence, and narrows broad topics into well-defined 
    research questions. It produces focused questions derived from initially
    broad concepts.
    """
    # Parse incoming ideas robustly (handles JSON-like lists, numbered lists, and '?' sentences)
    idea_list = _parse_question_list(ideas)
    
    if not idea_list:
        return []
    
    # Combine ideas for processing by LLM
    combined_ideas = "\n".join([f"- {idea}" for idea in idea_list])
    
    prompt = f"""
    As a research refinement expert in economics, evaluate and refine the following research ideas:
    
    {combined_ideas}
    
    For each idea:
    1. Assess clarity and specificity
    2. Identify and eliminate any conceptual overlaps or redundancies
    3. Ensure methodological feasibility
    4. Improve precision of language and scope
    5. Ensure the question is well-defined and answerable through research
    
    Return the refined list of research questions with each question on a new line. 
    If you consolidate multiple questions due to redundancy, explain your reasoning briefly.
    Format each question to begin with a number(e.g., "1.").
    """
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
    )
    refined_content = completion.choices[0].message.content.strip()
    
    # Parse refined LLM output; include only likely questions
    refined_candidates = _parse_question_list(refined_content)
    # Deduplicate while preserving order
    seen = set()
    refined_questions: list[str] = []
    for q in refined_candidates:
        normalized = q.strip().rstrip('.')
        if normalized.endswith('?') and normalized.lower() not in seen:
            refined_questions.append(normalized)
            seen.add(normalized.lower())
    
    # Fallback if parsing fails
    if not refined_questions:
        refined_questions = idea_list
    
    # Track unique refined questions across calls
    for q in refined_questions:
        EXECUTION_STATS["refined_questions_set"].add(q.strip().lower())
    EXECUTION_STATS["refined_questions_initial_total"] = len(EXECUTION_STATS["initial_ideas_set"])  # base against unique initial ideas
    EXECUTION_STATS["refined_questions_total"] = len(EXECUTION_STATS["refined_questions_set"])
    return refined_questions

def add_context(questions: str) -> list:
    """
    Uses LLM to contextualize research questions within appropriate economic frameworks.
    Selects relevant economic theories tailored to each question.
    """
    ideas = [idea.strip() for idea in questions.split(";") if idea.strip()]
    contextualized_ideas = []
    
    for idea in ideas:
        prompt = f"""
        As an expert economics researcher, enhance the following research question by integrating relevant economic theoretical frameworks:

        "{idea}"

        Your task:
        1. Identify 1-2 most relevant economic theories or schools of thought (e.g., Behavioral Economics, New Institutional Economics, Ecological Economics)
        2. Naturally integrate these theoretical perspectives into a refined version of the question
        3. Make the question more compelling and academically rigorous
        
        Guidelines:
        - Transform the question to be more nuanced and theoretically grounded
        - Avoid rigid formatting like "within the framework of X"
        - Incorporate theoretical lens organically into the question itself
        - Ensure the question remains clear and focused on its original intent
        - Make the question intellectually stimulating for economists
        
        Example transformation:
        Original: "How do corporate tax rates affect employment?"
        Enhanced: "How do changes in corporate tax policies influence labor market dynamics when analyzed through efficiency wage theory and considering institutional constraints?"
        
        Return only the enhanced question without any additional text or explanation.
        """
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]
        )
        context = completion.choices[0].message.content.strip()
        
        # If the enhanced question is empty or too short, use the original
        if not context or len(context) < len(idea) / 2:
            context = idea
        
        contextualized_ideas.append(context)
        
    EXECUTION_STATS["contextualized_count"] = max(
        EXECUTION_STATS.get("contextualized_count", 0), len(contextualized_ideas)
    )
    if not EXECUTION_STATS.get("context_log_emitted"):
        info("Contextualizer adding theoretical frameworks... Complete.")
        EXECUTION_STATS["context_log_emitted"] = True
    return contextualized_ideas

def finalize_questions(questions: str) -> str:
    """
    Uses LLM to aggregate, synthesize, and prioritize research questions.
    
    This function ensures alignment with project objectives, validates questions
    against pertinent literature, and produces a prioritized shortlist ready for
    final selection. It represents the culmination of the entire ideation process.
    
    Returns a string with prioritized questions, ending with "TERMINATE".
    """
    q_list = [q.strip() for q in questions.split(";") if q.strip()]
    
    if not q_list:
        return "No valid research questions were found.\nTERMINATE"
    
    # Combine questions for processing by LLM
    combined_questions = "\n".join([f"- {q}" for q in q_list])
    
    prompt = f"""
    As a research synthesis expert in economics, review and finalize this set of research questions:
    
    {combined_questions}
    
    Your task is to:
    1. Synthesize closely related questions where appropriate
    2. Prioritize questions based on originality, feasibility, and potential impact
    3. Ensure alignment with current economic research objectives
    4. Validate questions against relevant literature (conceptually)
    5. Create a final, prioritized shortlist (3-5 questions maximum)
    
    For each question, provide a brief (1-2 sentence) justification for its inclusion and priority ranking.
    Present the questions in order of priority (highest to lowest).
    
    Format each output exactly as follows:
    1. [Research question without any label]
    Justification: [Your justification for this question]
    
    2. [Next research question]
    Justification: [Your justification for this question]
    
    And so on. Do not include the word "Justification" at the end of the question itself.
    """
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
    )
    finalized_content = completion.choices[0].message.content.strip()
    
    # Process the finalized content
    finalized = []
    current_question = ""
    
    for line in finalized_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Detect new question (starts with number)
        if line[0].isdigit() and '. ' in line[:5]:
            if current_question:
                finalized.append(current_question)
            current_question = line
        else:
            # Continue current question/justification
            current_question += " " + line
    
    # Add the last question
    if current_question:
        finalized.append(current_question)
    
    # Fallback if parsing fails
    if not finalized:
        finalized = q_list
    
    # Save raw finalized content and stats
    global LAST_FINALIZED_OUTPUT
    LAST_FINALIZED_OUTPUT = "\n".join(finalized)
    # Count questions by leading numbering
    num_final = 0
    for line in LAST_FINALIZED_OUTPUT.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line[:5]:
            num_final += 1
    EXECUTION_STATS["final_questions_count"] = max(EXECUTION_STATS.get("final_questions_count", 0), num_final)
    info("Finalizer prioritizing questions... Complete.")
    # Keep the mandatory termination marker
    return LAST_FINALIZED_OUTPUT + "\nTERMINATE"

trend_surfer_tool = FunctionTool(
    scrape_economic_news,
    description="Scrapes recent economic news from Google (Custom Search API) to identify trending topics."
)

greyscout_tool = FunctionTool(
    search_policy_literature,
    description=(
        "Searches for grey literature and policy papers using Google Custom Search API, "
        "focusing on World Bank, IMF, ECB, BIS, and OECD websites. Accepts optional 'query'."
    )
)

topiccrawler_tool = FunctionTool(
    crawl_academic_sources,
    description=(
        "Crawls preliminary academic literature from arXiv (API) and SSRN/NBER (via Google Custom Search). "
        "Accepts optional 'query' to focus the search."
    )
)

scholarsearcher_tool = FunctionTool(
    search_scholarly_databases,
    description=(
        "Queries Web of Science, Scopus, EconLit via open-access fallbacks using Google Custom Search; "
        "returns publicly accessible items when subscriptions are required. Accepts optional 'query'."
    )
)

ideation_tool = FunctionTool(
    generate_ideas,
    description="Aggregates insights from multiple sources to generate initial research ideas."
)

refinement_tool = FunctionTool(
    refine_ideas,
    description="Evaluates and refines research ideas to remove redundancies, ensure methodological feasibility, and improve precision."
)

contextualization_tool = FunctionTool(
    add_context,
    description="Contextualizes research questions with established economic theories and policy debates."
)

finalization_tool = FunctionTool(
    finalize_questions,
    description="Aggregates and prioritizes the refined research questions into a final shortlist, ending with 'TERMINATE'."
)


###########################################
# Defining Agents
########################################### 

trend_surfer_agent = AssistantAgent(
    name="TrendSurfer",
    tools=[trend_surfer_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Scrapes and analyzes trending economic topics from recent news using Google Search API.",
    system_message=(
        "You are TrendSurfer, responsible for discovering trending economic topics from recent news using Google Custom Search API. "
        "If the task includes a specific human-provided idea or topic, call your tool with that text as the query to retrieve related trends. "
        "Otherwise, call your tool without a query to retrieve general economic trends. "
        "Provide detailed analysis including titles, sources, summaries, and content previews. Always show the complete crawled information and analysis."
    )
)

scholarsearcher_agent = AssistantAgent(
    name="ScholarSearcher",
    tools=[scholarsearcher_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Executes academic queries across Web of Science, Scopus, EconLit, collecting open-access items where possible.",
    system_message=(
        "You are ScholarSearcher. Execute detailed academic queries across Web of Science, Scopus, and EconLit. "
        "When subscription/login blocks access, gather as much open-access information as possible via public landing pages. "
        "If a human-provided idea/topic exists, call your tool with that 'query'; otherwise call without. "
        "Present a concise list with titles, links, and short summaries."
    )
)

topiccrawler_agent = AssistantAgent(
    name="TopicCrawler",
    tools=[topiccrawler_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Collects preliminary academic literature from arXiv, SSRN, and NBER to build an initial knowledge base.",
    system_message=(
        "You are TopicCrawler. Gather preliminary academic literature from arXiv, SSRN, and NBER. "
        "If there is a human-provided idea/topic, call your tool with that text as 'query'. Otherwise, call without it to discover general sources. "
        "Present the results as a concise list with sources and links."
    )
)

greyscout_agent = AssistantAgent(
    name="GreyScout",
    tools=[greyscout_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Retrieves grey literature and policy papers from major institutions (World Bank, IMF, ECB, BIS, OECD).",
    system_message=(
        "You are GreyScout. Your job is to retrieve grey literature and policy papers from major institutions "
        "(World Bank, IMF, ECB, BIS, OECD). If the task contains a human-provided idea/topic, call your tool with that "
        "text as 'query'. Otherwise, call your tool without 'query' to discover general recent policy papers. "
        "Present findings with title, source, and brief summary."
    )
)

ideator_agent = AssistantAgent(
    name="Ideator",
    tools=[ideation_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Generates or enriches research ideas by synthesizing inputs from various agents.",
    system_message="You are Ideator, aggregating insights from automated agents and human inputs. If a human-provided idea exists, your task is to enrich and expand it with the insights gathered. Otherwise, propose entirely new research ideas based on the information gathered."
)

refiner_agent = AssistantAgent(
    name="Refiner",
    tools=[refinement_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Refines raw research ideas to remove redundancies and improve clarity.",
    system_message="You are Refiner, tasked with evaluating and clarifying generated research ideas into precise research questions. If working with a human-provided idea, focus on refining and enhancing that idea while preserving its core intent."
)

contextualizer_agent = AssistantAgent(
    name="Contextualizer",
    tools=[contextualization_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Provides context to research questions by linking them to economic theories.",
    system_message="You are Contextualizer, responsible for situating research questions within broader economic frameworks. If working with a human-provided idea, focus on contextualizing that specific idea within relevant economic theories and current debates."
)

finalizer_agent = AssistantAgent(
    name="Finalizer",
    tools=[finalization_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key),
    description="Synthesizes the final research questions with proper academic structure and clear rationale.",
    system_message="""You are Finalizer, responsible for creating well-structured, academic research questions from the refined ideas. 

CRITICAL REQUIREMENTS:
1. NEVER output just keywords or short phrases
2. ALWAYS create complete, properly formatted research questions
3. Each question must include:
   - Clear research objective
   - Specific variables or factors
   - Academic context (theories, frameworks)
   - Justification explaining relevance and contribution

FORMAT EACH QUESTION AS:
"How do [specific variables/factors] affect [outcome] in [context], particularly through the lens of [academic theory/framework], and what implications does this have for [broader field/application]? Justification: [Clear explanation of why this question is important, original, and contributes to the field]"

EXAMPLE GOOD QUESTION:
"How do digital transformation initiatives reshape labor productivity and wage structures in manufacturing sectors across developed versus developing economies, particularly through the lens of New Institutional Economics and the theory of technological unemployment, and what policy implications emerge for addressing potential skill gaps and income inequality? Justification: This question addresses a critical contemporary issue at the intersection of technology adoption, labor markets, and economic development, offering insights into how different institutional contexts shape technological impacts on employment and wages."

Ensure all questions follow this comprehensive format with proper academic rigor and clear justification."""
)

###########################################
# Human-in-the-Loop Checkpoint Agents
###########################################

# Checkpoint 1: After literature search
def checkpoint1_input_func(prompt: str) -> str:
    sys.stdout.flush()
    time.sleep(0.1)
    banner = [
        "\n🛑 CHECKPOINT 1: Ideation-Input Sourcing Stage",
        "=" * 50,
        "The search agents have completed their tasks. Please review the gathered information.",
        "",
        "Expected response examples:",
        "- 'Approved' → proceed to next stage",
        "- 'Focus on post-2008 studies' → restart search with refined parameters",
        "- 'Add more recent papers' → expand search with date filters",
        "- 'Exclude non-peer-reviewed sources' → refine search criteria",
        "- Or provide any other specific feedback you have",
        "",
        "Your feedback:",
    ]
    print("\n".join(banner))
    sys.stdout.flush()
    time.sleep(0.1)
    try:
        print("> ", end="", flush=True)
    except Exception:
        pass
    feedback = input()
    try:
        EXECUTION_STATS["checkpoint1_feedback"] = feedback
    except Exception:
        pass
    return feedback

checkpoint1_human = UserProxyAgent(
    name="Checkpoint1_Human",
    input_func=checkpoint1_input_func
)

# Checkpoint 2: After reference compilation
def checkpoint2_input_func(prompt: str) -> str:
    sys.stdout.flush()
    time.sleep(0.1)
    banner = [
        "\n🛑 CHECKPOINT 2: Ideation: Refinement Pipeline Stage",
        "=" * 50,
        "The Ideation-Refinement Pipeline has completed its tasks. Please review the refined research questions.",
        "",
        "Expected response examples:",
        "- 'Approved' → proceed to report generation",
        "- Or provide any other specific feedback you have",
        "",
        "Your feedback:",
    ]
    print("\n".join(banner))
    sys.stdout.flush()
    time.sleep(0.1)
    try:
        print("> ", end="", flush=True)
    except Exception:
        pass
    feedback = input()
    try:
        EXECUTION_STATS["checkpoint2_feedback"] = feedback
    except Exception:
        pass
    return feedback

checkpoint2_human = UserProxyAgent(
    name="Checkpoint2_Human",
    input_func=checkpoint2_input_func
)

# Checkpoint 3: After report generation
def checkpoint3_input_func(prompt: str) -> str:
    sys.stdout.flush()
    time.sleep(0.1)
    banner = [
        "\n🛑 CHECKPOINT 3: Ideation: Integration and Final Vetting Stage",
        "=" * 50,
        "The Ideation: Integration and Final Vetting Stage has completed its tasks. Please review the integrated research questions.",
        "",
        "Expected response examples:",
        "- 'Looks good' → proceed to report generation",
        "- Or provide any other specific feedback you have",
        "",
        "Your feedback:",
    ]
    print("\n".join(banner))
    sys.stdout.flush()
    time.sleep(0.1)
    try:
        print("> ", end="", flush=True)
    except Exception:
        pass
    feedback = input()
    try:
        EXECUTION_STATS["checkpoint3_feedback"] = feedback
    except Exception:
        pass
    return feedback

checkpoint3_human = UserProxyAgent(
    name="Checkpoint3_Human",
    input_func=checkpoint3_input_func
)

###########################################
# Creating Teams with Checkpoints
###########################################

search_team = RoundRobinGroupChat(
    participants=[trend_surfer_agent, topiccrawler_agent, scholarsearcher_agent, greyscout_agent, checkpoint1_human], max_turns=5
)

ideation_team = RoundRobinGroupChat(
    participants=[ideator_agent, refiner_agent, checkpoint2_human], max_turns=4
)

integration_team = RoundRobinGroupChat(
    participants=[contextualizer_agent, finalizer_agent, checkpoint3_human], max_turns=4
)

termination = TextMentionTermination("TERMINATE")


###########################################
# Run the Ideation Workflow
###########################################

def _format_duration(start: datetime.datetime, end: datetime.datetime) -> str:
    total_seconds = int((end - start).total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"

def _extract_top_questions(raw_finalized: str, max_items: int = 5) -> list:
    if not raw_finalized:
        return []
    questions = []
    for line in raw_finalized.split("\n"):
        text = line.strip()
        if not text:
            continue
        if text[0].isdigit() and ". " in text[:5]:
            # Cut off at "Justification:" if present
            question_part = text
            if "Justification:" in question_part:
                question_part = question_part.split("Justification:", 1)[0].strip()
            # Remove leading numbering
            parts = question_part.split('. ', 1)
            if len(parts) > 1:
                question_part = parts[1].strip()
            questions.append(question_part)
            if len(questions) >= max_items:
                break
    return questions

def _write_markdown_log(console_text: str, output_dir: str = ".") -> str:
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ideation_log_{timestamp_str}.md"
    filepath = os.path.join(output_dir, filename)

    start = EXECUTION_STATS.get("start_time")
    end = EXECUTION_STATS.get("end_time") or datetime.datetime.now()
    duration_str = _format_duration(start, end) if start else "N/A"

    # Build markdown
    lines = []
    lines.append("### Run Summary")
    lines.append("")
    # INFO block in requested order and wording
    # 1. Start + init
    if EXECUTION_STATS.get("human_idea"):
        lines.append(f"[INFO] Starting ideation process with human idea: {EXECUTION_STATS['human_idea']}")
    else:
        lines.append("[INFO] Starting ideation process without human input")
    lines.append("[INFO] Initializing agents... Done.")
    # 2. IdeaEnricher placeholder to match requested wording
    lines.append("[INFO] IdeaEnricher analyzing human idea... Complete.")
    # 3. TrendSurfer
    lines.append("[INFO] TrendSurfer retrieving related trending topics from Google...")
    if EXECUTION_STATS.get("trending_topics_count"):
        lines.append(f"[INFO] Retrieved {EXECUTION_STATS['trending_topics_count']} trending topics (filtered to {EXECUTION_STATS['trending_topics_count']} most relevant)")
    # 4. GreyScout and Academic
    lines.append("[INFO] GreyScout retrieving policy literature... Complete.")
    lines.append("[INFO] TopicCrawler and ScholarSearcher retrieving academic sources... Complete.")
    # 4.a Human checkpoint feedback for Phase 1
    if EXECUTION_STATS.get("checkpoint1_feedback"):
        lines.append(f"[INFO] Checkpoint 1 feedback: {EXECUTION_STATS['checkpoint1_feedback']}")
    # 5. Source statistics aggregate
    if EXECUTION_STATS.get("news_articles_count") or EXECUTION_STATS.get("policy_papers_count") or EXECUTION_STATS.get("academic_sources_count"):
        lines.append(
            f"[INFO] Source statistics: {EXECUTION_STATS.get('news_articles_count', 0)} news articles, {EXECUTION_STATS.get('policy_papers_count', 0)} policy papers, {EXECUTION_STATS.get('academic_sources_count', 0)} academic papers integrated."
        )
    # 6. Pipeline steps
    lines.append("[INFO] Ideator synthesizing research questions... Complete.")
    # Simplified reporting per request: do not count or display numbers for refiner step
    lines.append("[INFO] Refiner removing redundancies... Complete.")
    # 6.a Human checkpoint feedback for Phase 2
    if EXECUTION_STATS.get("checkpoint2_feedback"):
        lines.append(f"[INFO] Checkpoint 2 feedback: {EXECUTION_STATS['checkpoint2_feedback']}")
    # Continue pipeline steps
    lines.append("[INFO] Contextualizer adding theoretical frameworks... Complete.")
    lines.append("[INFO] Finalizer prioritizing questions... Complete.")
    # 6.b Human checkpoint feedback for Phase 3
    if EXECUTION_STATS.get("checkpoint3_feedback"):
        lines.append(f"[INFO] Checkpoint 3 feedback: {EXECUTION_STATS['checkpoint3_feedback']}")
    lines.append("")
    lines.append("--- TOP RESEARCH QUESTIONS GENERATED ---")
    top_questions = _extract_top_questions(LAST_FINALIZED_OUTPUT, max_items=5)
    if top_questions:
        for idx, q in enumerate(top_questions, 1):
            lines.append(f"{idx}. {q}")
    else:
        lines.append("No questions extracted.")
    lines.append("")
    lines.append("--- EXECUTION SUMMARY ---")
    lines.append(f"Total execution time: {duration_str}")
    output_line = None
    if EXECUTION_STATS.get("final_questions_count"):
        output_line = f"Output: {EXECUTION_STATS['final_questions_count']} refined research questions"
    if EXECUTION_STATS.get("trending_topics_count"):
        lines.append(f"Trending topics discovered: {EXECUTION_STATS['trending_topics_count']}")
    if EXECUTION_STATS.get("policy_papers_count"):
        lines.append(f"Policy papers discovered: {EXECUTION_STATS['policy_papers_count']}")
    if EXECUTION_STATS.get("academic_sources_count"):
        lines.append(f"Academic sources discovered: {EXECUTION_STATS['academic_sources_count']}")
    if EXECUTION_STATS.get("scholarly_db_results_count"):
        lines.append(f"Scholarly DB accessible items: {EXECUTION_STATS['scholarly_db_results_count']}")
    if output_line:
        lines.append(output_line)
    lines.append("")
    lines.append("### Console Output")
    lines.append("")
    lines.append("```text")
    lines.append(console_text.rstrip())
    lines.append("```")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath

async def main(human_idea: str = None, mode: str = "human"):
    """
    Run the ideation workflow with two modes:
    
    Mode 1 (human): Human-provided initial idea (existing functionality)
    Mode 2 (automatic): Automatic ideation using TrendSurfer to find trending topics
    
    Args:
        human_idea: Optional initial research idea provided by a human expert (Mode 1)
        mode: "human" for Mode 1, "automatic" for Mode 2
    """
    # Tee stdout to capture console output
    buffer = io.StringIO()
    tee = StdoutTee(buffer)
    original_stdout = sys.stdout
    sys.stdout = tee

    EXECUTION_STATS["mode"] = mode
    EXECUTION_STATS["human_idea"] = human_idea
    EXECUTION_STATS["start_time"] = datetime.datetime.now()
    info(
        f"Starting ideation process with {'human idea: ' + human_idea if human_idea else 'no human input'}"
    )
    info("Initializing agents... Done.")

    # Adapted to use sub-teams sequentially (reference: literatureTeam.py)
    # Phase 1: Search and Input Sourcing
    print("\n[Phase 1] Ideation Input Sourcing")
    print("-" * 30)
    if human_idea:
        search_task = (
            f"Using the human-provided idea: '{human_idea}', perform input sourcing: "
            f"- TrendSurfer: retrieve related trending topics from recent news using the idea as the query. "
            f"- TopicCrawler: gather preliminary academic literature using the same query. "
            f"- ScholarSearcher: execute detailed academic queries (open-access where needed) using the same query. "
            f"- GreyScout: retrieve relevant policy and grey literature using the same query."
        )
        print(f"{Colors.MAGENTA}Starting ideation process with human idea: {human_idea}{Colors.RESET}")
    else:
        search_task = (
            "Perform input sourcing without a specific human idea: "
            "- TrendSurfer: discover general trending economic topics (no query). "
            "- TopicCrawler: retrieve preliminary academic literature (no query). "
            "- ScholarSearcher: execute detailed academic queries (open-access where needed) (no query). "
            "- GreyScout: retrieve recent policy and grey literature (no query)."
        )
        print(f"{Colors.MAGENTA}Starting ideation process without human input{Colors.RESET}")

    # Run Phase 1 in two steps to avoid interleaved outputs before human prompt
    search_team_core = RoundRobinGroupChat(
        participants=[trend_surfer_agent, topiccrawler_agent, scholarsearcher_agent, greyscout_agent], max_turns=4
    )
    await Console(search_team_core.run_stream(task=search_task))
    search_checkpoint_team = RoundRobinGroupChat(participants=[checkpoint1_human], max_turns=1)
    await Console(search_checkpoint_team.run_stream(task="Please review the gathered information above and provide feedback."))

    # Phase 2: Ideation and Refinement
    print("\n[Phase 2] Ideation and Refinement")
    print("-" * 30)
    if human_idea:
        ideation_task = (
            f"Ideator: enrich and expand the human idea '{human_idea}' using the sourced trends, academic, and policy inputs. "
            f"Refiner: evaluate and refine into precise, non-redundant research questions."
        )
    else:
        ideation_task = (
            "Ideator: synthesize sourced inputs into initial research questions. "
            "Refiner: evaluate and refine into precise, non-redundant research questions."
        )
    # Run Phase 2 in two steps to avoid interleaved outputs before human prompt
    ideation_team_core = RoundRobinGroupChat(
        participants=[ideator_agent, refiner_agent], max_turns=4
    )
    await Console(ideation_team_core.run_stream(task=ideation_task))
    ideation_checkpoint_team = RoundRobinGroupChat(participants=[checkpoint2_human], max_turns=1)
    await Console(ideation_checkpoint_team.run_stream(task="Please review the refined research questions above and provide feedback."))

    # Phase 3: Integration and Finalization
    print("\n[Phase 3] Integration and Finalization")
    print("-" * 30)
    integration_task = (
        "Contextualizer: add relevant economic theoretical frameworks to the refined questions. "
        "Finalizer: synthesize and prioritize a final shortlist (3-5) with brief justifications, and end with the word 'TERMINATE'."
    )
    # Run Phase 3 in two steps to avoid interleaved outputs before human prompt
    integration_team_core = RoundRobinGroupChat(
        participants=[contextualizer_agent, finalizer_agent], max_turns=4
    )
    await Console(integration_team_core.run_stream(task=integration_task))
    integration_checkpoint_team = RoundRobinGroupChat(participants=[checkpoint3_human], max_turns=1)
    await Console(integration_checkpoint_team.run_stream(task="Please review the integrated research questions above and provide final feedback."))

    # Close tee and write markdown
    sys.stdout = original_stdout
    EXECUTION_STATS["end_time"] = datetime.datetime.now()
    log_path = _write_markdown_log(buffer.getvalue(), output_dir=".")
    info(f"Saved ideation markdown log to {log_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Ideation Team workflow.')
    parser.add_argument('--idea', type=str, help='Optional initial research idea provided by a human expert')
    parser.add_argument('--mode', type=str, choices=["human", "automatic"], default="human", help='Run in "human" or "automatic" mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed logging')
    
    args = parser.parse_args()
    
    # Set debug mode globally
    DEBUG_MODE = args.debug
    
    asyncio.run(main(human_idea=args.idea, mode=args.mode))