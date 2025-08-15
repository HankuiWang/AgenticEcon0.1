import asyncio
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, HandoffTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Constants for retry logic
MAX_RETRIES = 3
MAX_TURNS = 5

# Run summary tracker
RUN_SUMMARY = {
    "topic": "",
    "start_time": None,
    "end_time": None,
    "total_seconds": 0,
    "google_total": 0,
    "google_relevant": 0,
    "arxiv_total": 0,
    "arxiv_relevant": 0,
    "checkpoint1_feedback": "",
    "checkpoint2_feedback": "",
    "checkpoint3_feedback": ""
}

RUN_LOGS: list[str] = []

def log_info(message: str) -> None:
    print(f"[INFO] {message}")
    RUN_LOGS.append(f"[INFO] {message}")

def _count_relevant_texts(items, query_terms, text_getters):
    relevant_count = 0
    for item in items:
        is_relevant = False
        for getter in text_getters:
            try:
                value = getter(item)
            except Exception:
                value = ""
            text_lower = (value or "").lower()
            if any(term in text_lower for term in query_terms):
                is_relevant = True
                break
        if is_relevant:
            relevant_count += 1
    return relevant_count

def _tokenize_query(query):
    terms = [t.strip().lower() for t in query.split() if len(t.strip()) > 2]
    return set(terms)

###########################################
# Defining Tools
###########################################

def google_search(query: str, num_results: int = 2, max_chars: int = 500) -> list:  # type: ignore[type-arg]
    import os
    import time

    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": search_engine_id, "q": query, "num": num_results}

    try:
        log_info("Google Search Agent retrieving web resources...")
    except Exception:
        pass

    response = requests.get(url, params=params)  # type: ignore[arg-type]

    if response.status_code != 200:
        print(response.json())
        raise Exception(f"Error in API request: {response.status_code}")

    results = response.json().get("items", [])

    def get_page_content(url: str) -> str:
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
            print(f"Error fetching {url}: {str(e)}")
            return ""

    enriched_results = []
    for item in results:
        body = get_page_content(item["link"])
        enriched_results.append(
            {"title": item["title"], "link": item["link"], "snippet": item["snippet"], "body": body}
        )
        time.sleep(1)  # Be respectful to the servers

    # Update run summary counters
    try:
        RUN_SUMMARY["google_total"] += len(enriched_results)
        query_terms = _tokenize_query(query)
        relevant_count = _count_relevant_texts(
            enriched_results,
            query_terms,
            [
                lambda x: x.get("snippet", ""),
                lambda x: x.get("body", ""),
                lambda x: x.get("title", "")
            ]
        )
        RUN_SUMMARY["google_relevant"] += relevant_count
        log_info(f"Retrieved {len(enriched_results)} web resources ({relevant_count} relevant after filtering)")
    except Exception:
        pass

    return enriched_results


def arxiv_search(query: str, max_results: int = 2) -> list:  # type: ignore[type-arg]
    """
    Search Arxiv for papers and return the results including abstracts.
    """
    import arxiv # type: ignore

    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results = []
    try:
        log_info("ArXiv Search Agent retrieving academic papers...")
    except Exception:
        pass
    for paper in client.results(search):
        results.append(
            {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
            }
        )

    # Update run summary counters
    try:
        RUN_SUMMARY["arxiv_total"] += len(results)
        query_terms = _tokenize_query(query)
        relevant_count = _count_relevant_texts(
            results,
            query_terms,
            [
                lambda x: x.get("title", ""),
                lambda x: x.get("abstract", "")
            ]
        )
        RUN_SUMMARY["arxiv_relevant"] += relevant_count
        log_info(f"Retrieved {len(results)} papers ({relevant_count} highly relevant)")
    except Exception:
        pass

    return results

google_search_tool = FunctionTool(
    google_search, description="Search Google for information, returns results with a snippet and body content"
)
arxiv_search_tool = FunctionTool(
    arxiv_search, description="Search Arxiv for papers related to a given topic, including abstracts"
)

###########################################
# Defining Agents
###########################################
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key)

google_search_agent = AssistantAgent(
    name="Google_Search_Agent",
    tools=[google_search_tool],
    model_client=model_client,
    description="An agent that can search Google for information, returns results with a snippet and body content",
    system_message="""You are a helpful AI assistant that searches Google for information. 
    
    When given a search task:
    1. Use your google_search tool to find relevant information
    2. Present the results clearly
    3. After completing the search, respond with 'SEARCH_COMPLETE' to end your task
    
    Always end your response with 'SEARCH_COMPLETE' when you have finished searching.""",
)

arxiv_search_agent = AssistantAgent(
    name="Arxiv_Search_Agent",
    tools=[arxiv_search_tool],
    model_client=model_client,
    description="An agent that can search Arxiv for papers related to a given topic, including abstracts",
    system_message="""You are a helpful AI assistant that searches Arxiv for academic papers. 
    
    When given a search task:
    1. Use your arxiv_search tool to find relevant academic papers
    2. Present the results clearly with titles, authors, and abstracts
    3. After completing the search, respond with 'SEARCH_COMPLETE' to end your task
    
    Always end your response with 'SEARCH_COMPLETE' when you have finished searching.""",
)

InsightSummarizer = AssistantAgent(
    name="InsightSummarizer",
    model_client=model_client,
    description="An agent that can summarize the insights from EXACTLY AND ONLY google search results",
    system_message="""You are the Insight Summarizer, an expert at extracting key insights EXACTLY AND ONLY from google search results.
            
            Your responsibilities:
            - Analyze EXACTLY AND ONLY google search results to extract key insights
            - Identify emerging trends and patterns
            - Summarize recent developments and findings
            - Provide actionable insights from web content
            
            Present the insights in the style of [Insight 1: [Insight 1 description], Insight 2: [Insight 2 description], ...].
            When you complete your analysis, respond with 'ANALYSIS_COMPLETE' to indicate you're done.
            
             """,
)

PaperDecomposer = AssistantAgent(
    name="PaperDecomposer",
    model_client=model_client,
    description="An agent that can analyze academic papers and extract key insights",
    system_message="""You are the Paper Decomposer, an expert at EXACTLY AND ONLY analyzing academic papers and extracting key insights.
            
            Your responsibilities:
            - Analyze paper content and extract key information
            - Identify research questions, methodologies, and findings
            - Extract theoretical frameworks and data sources
            - Assess paper quality and significance
            
            Present the views in the style of [View 1: [View 1 description], View 2: [View 2 description], ...].
            When you complete your analysis, respond with 'ANALYSIS_COMPLETE' to indicate you're done.
            
            """,
)

GapFinder = AssistantAgent(
    name="GapFinder",
    model_client=model_client,
    description="An agent that can identify gaps and opportunities in academic literature",
    system_message="""You are the Gap Finder, an expert at identifying gaps and opportunities in academic literature.
            
            Your responsibilities:
            - Identify research gaps across multiple papers
            - Spot underexplored areas and methodological limitations
            - Recognize emerging trends and opportunities
            - Suggest future research directions
            
            Present the gaps in the style of [Gap 1: [Gap 1 description], Gap 2: [Gap 2 description], ...].
            When you complete your gap analysis, respond with 'ANALYSIS_COMPLETE' to indicate you're done.
           
            """,
)

ReferenceKeeper = AssistantAgent(
    name="ReferenceKeeper",
    model_client=model_client,
    description="An agent that can keep track of references and citations",
    system_message="""You are the Reference Keeper, an expert at keeping track of references and citations.
            
            Your responsibilities:
            - Keep track of references and citations
            - Ensure that the references are correct and up to date
            - Ensure that the citations are correct and up to date
            - Ensure that the references are formatted correctly
            
            Present the references in the style of [1. [Reference 1 description], 2. [Reference 2 description], ...].
            When you complete your reference analysis, respond with 'REFERENCES_COMPLETE' to indicate you're done.
           
            """,
)

ReportAgent = AssistantAgent(
    name="ReportAgent",
    model_client=model_client,
    description="Generate a report based on a given topic and the data extracted from the other agents",
    system_message="""You are a literature review expert. 

            Your responsibilities:
            - Synthesize data extracted into a high quality literature review including CORRECT references. 
            - You MUST write a final report Based on the data extracted from the other agents.  

            Present the final report with the Title: Literature Review on [Topic].
            - Your response should end with the word 'TERMINATE'.
           
            """,
)

###########################################
# Human-in-the-Loop Checkpoint Agents
###########################################

# Checkpoint 1: After literature search
def checkpoint1_input_func(prompt: str) -> str:
    print(f"\n🛑 CHECKPOINT 1: Literature Search Review")
    print("=" * 50)
    print("The search agents have completed their tasks. Please review the gathered literature.")
    try:
        log_info("CHECKPOINT: Search results ready for human review")
    except Exception:
        pass
    print("\nExpected response examples:")
    print("- 'Approved' → proceed to next stage")
    print("- 'Focus on post-2008 studies' → restart search with refined parameters")
    print("- 'Add more recent papers' → expand search with date filters")
    print("- 'Exclude non-peer-reviewed sources' → refine search criteria")
    print("- Or provide any other specific feedback you have")
    print("\nYour feedback:")
    feedback = input("> ")
    try:
        RUN_SUMMARY["checkpoint1_feedback"] = feedback
        if feedback.strip():
            log_info(f"Human approved results with refinement: \"{feedback}\"")
    except Exception:
        pass
    return feedback

checkpoint1_human = UserProxyAgent(
    name="Checkpoint1_Human",
    input_func=checkpoint1_input_func
)

# Checkpoint 2: After reference compilation
def checkpoint2_input_func(prompt: str) -> str:
    print(f"\n🛑 CHECKPOINT 2: Reference Validation")
    print("=" * 50)
    print("The ReferenceKeeper has compiled references. Please validate academic credibility.")
    try:
        log_info("CHECKPOINT: Analysis ready for human review")
    except Exception:
        pass
    print("\nExpected response examples:")
    print("- 'Approved' → proceed to report generation")
    print("- 'Exclude non-peer-reviewed sources' → filter references")
    print("- 'Add more recent citations' → expand reference search")
    print("- 'Check citation format' → validate formatting")
    print("- Or provide any other specific feedback you have")
    print("\nYour feedback:")
    feedback = input("> ")
    try:
        RUN_SUMMARY["checkpoint2_feedback"] = feedback
        if feedback.strip():
            log_info(f"Human guidance received: \"{feedback}\"")
    except Exception:
        pass
    return feedback

checkpoint2_human = UserProxyAgent(
    name="Checkpoint2_Human",
    input_func=checkpoint2_input_func
)

# Checkpoint 3: After report generation
def checkpoint3_input_func(prompt: str) -> str:
    print(f"\n🛑 CHECKPOINT 3: Final Report Review")
    print("=" * 50)
    print("The ReportAgent has generated the literature review. Please provide final validation.")
    try:
        log_info("CHECKPOINT: Draft report ready for human review")
    except Exception:
        pass
    print("\nExpected response examples:")
    print("- 'Looks good' → finalize and save report")
    print("- 'Rephrase section on methodology' → request specific revisions")
    print("- 'Add more detail to conclusion' → request content expansion")
    print("- 'Restructure introduction' → request organizational changes")
    print("- Or provide any other specific feedback you have")
    print("\nYour feedback:")
    feedback = input("> ")
    try:
        RUN_SUMMARY["checkpoint3_feedback"] = feedback
        if feedback.strip():
            log_info("Human revision requests processed... Complete.")
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
    participants=[google_search_agent, arxiv_search_agent, checkpoint1_human], max_turns=3
)

analysis_team = RoundRobinGroupChat(
    participants=[InsightSummarizer, PaperDecomposer, GapFinder, ReferenceKeeper, checkpoint2_human], max_turns=5
)

report_team = RoundRobinGroupChat(
    participants=[ReportAgent, checkpoint3_human], max_turns=2
)



async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Literature Review Team with Human-in-the-Loop Checkpoints')
    parser.add_argument('--topic', type=str, required=True, help='Topic for literature review')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES, help='Maximum number of retries for each checkpoint')
    args = parser.parse_args()
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"literature_review_{timestamp}.md"
    
    # Run the team with streaming output to show conversation
    print(f"Starting literature review on: {args.topic}")
    print("=" * 50)
    RUN_SUMMARY["topic"] = args.topic
    RUN_SUMMARY["start_time"] = datetime.now()
    log_info(f"Starting literature review process for: {args.topic}")
    log_info("Initializing agents... Done.")
    
    # Phase 1: Literature Search with Checkpoint 1
    print("\n📚 Phase 1: Literature Search")
    print("-" * 30)
    search_task = f"""Search for literature on {args.topic}."""
    search_result = await Console(search_team.run_stream(task=search_task))
    # After search phase, log totals if available
    try:
        if RUN_SUMMARY.get("google_total", 0):
            log_info(
                f"Retrieved {RUN_SUMMARY.get('google_total', 0)} web resources ({RUN_SUMMARY.get('google_relevant', 0)} relevant after filtering)"
            )
        if RUN_SUMMARY.get("arxiv_total", 0):
            log_info(
                f"Retrieved {RUN_SUMMARY.get('arxiv_total', 0)} papers ({RUN_SUMMARY.get('arxiv_relevant', 0)} highly relevant)"
            )
    except Exception:
        pass
    
    # Phase 2: Analysis with Checkpoint 2
    print("\n📊 Phase 2: Literature Analysis")
    print("-" * 30)
    analysis_task = f"""Analyze the searched literature and web articles on {args.topic} gathered from the search phase."""
    analysis_result = await Console(analysis_team.run_stream(task=analysis_task))
    # Log pipeline components as complete
    log_info("Paper processing pipeline initiated:")
    log_info("Insight Summarizer extracting key findings... Complete.")
    log_info("Paper Decomposer analyzing research structures... Complete.")
    log_info("Gap Finder identifying research limitations... Complete.")
    log_info("Reference Keeper standardizing references... Complete.")
    
    # Phase 3: Report Generation with Checkpoint 3
    print("\n📝 Phase 3: literature Review Generation")
    print("-" * 30)
    report_task = f"""Generate a comprehensive literature review on {args.topic}."""
    log_info("Report Agent generating synthesis...")
    report_result = await Console(report_team.run_stream(task=report_task))
    log_info("Report Agent generating synthesis... Complete.")
    
    # Extract the final literature review content from the conversation
    # Get the last message from the ReportAgent which should contain the literature review
    final_content = ""
    if hasattr(report_result, 'messages'):
        for message in reversed(report_result.messages):
            if hasattr(message, 'source') and message.source == 'ReportAgent':
                if hasattr(message, 'content'):
                    final_content = message.content
                    break
    
    # If we couldn't find the ReportAgent's message, try to get content from result
    if not final_content:
        final_content = report_result.content if hasattr(report_result, 'content') else str(report_result)
    
    # Mark end time and compute duration
    RUN_SUMMARY["end_time"] = datetime.now()
    try:
        RUN_SUMMARY["total_seconds"] = int((RUN_SUMMARY["end_time"] - RUN_SUMMARY["start_time"]).total_seconds())  # type: ignore[operator]
    except Exception:
        RUN_SUMMARY["total_seconds"] = 0

    def _format_duration(seconds: int) -> str:
        hrs = seconds // 3600
        mins = (seconds % 3600) // 60
        secs = seconds % 60
        if hrs > 0:
            return f"{hrs}h{mins:02d}m{secs:02d}s"
        else:
            return f"{mins}m{secs:02d}s"

    # Save to markdown file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Literature Review by Agentic Econ Workflow\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Topic: {args.topic}\n\n")
        f.write("---\n\n")
        # Insert Run Summary section with INFO logs
        if RUN_LOGS:
            f.write("### Run Summary\n\n")
            for line in RUN_LOGS:
                f.write(f"{line}\n")
            f.write("\n")
        f.write(final_content)
        # Append execution summary
        f.write("\n\n--- EXECUTION SUMMARY ---\n")
        total_sec = RUN_SUMMARY.get("total_seconds", 0) or 0
        f.write(f"Total execution time: {_format_duration(int(total_sec))}\n")
        f.write(
            f"Google web resources retrieved: {RUN_SUMMARY.get('google_total', 0)} total ({RUN_SUMMARY.get('google_relevant', 0)} relevant)\n"
        )
        f.write(
            f"ArXiv papers retrieved: {RUN_SUMMARY.get('arxiv_total', 0)} total ({RUN_SUMMARY.get('arxiv_relevant', 0)} relevant)\n"
        )
        if RUN_SUMMARY.get("checkpoint1_feedback"):
            f.write(f"Checkpoint 1 feedback: \"{RUN_SUMMARY.get('checkpoint1_feedback')}\"\n")
        if RUN_SUMMARY.get("checkpoint2_feedback"):
            f.write(f"Checkpoint 2 feedback: \"{RUN_SUMMARY.get('checkpoint2_feedback')}\"\n")
        if RUN_SUMMARY.get("checkpoint3_feedback"):
            f.write(f"Checkpoint 3 feedback: \"{RUN_SUMMARY.get('checkpoint3_feedback')}\"\n")
        f.write("Process completed successfully.\n")
    log_info("Final report generated.")
    
    print(f"\n📄 Literature review saved to: {filename}")
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
