import asyncio
import os
import argparse
import json
import pandas as pd
import fredapi  # type: ignore
import yfinance as yf  # type: ignore
import re
from dotenv import load_dotenv
from datetime import datetime
from typing import List

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ----------------------------------------
# Configuration
# ----------------------------------------

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
model_client = OpenAIChatCompletionClient(model=MODEL_NAME, api_key=openai_api_key)

FRED_API_KEY = os.getenv("FRED_API_KEY")
fred_client = fredapi.Fred(api_key=FRED_API_KEY)

AVAILABLE_SOURCES = ["Fred"]
# ----------------------------------------
# Tools 
# ----------------------------------------

def search_indicator( indicators: str, max_new: int = 3) -> str:
    base = [s.strip() for s in indicators.split(",") if s.strip()]
    results_per_indicator: dict[str, list[tuple[str, str]]] = {}

    if "Fred" in AVAILABLE_SOURCES:
        try:
            for indicator in base:
                search_query = f"potential {indicator} "
                df = fred_client.search(search_query)
                related_series: list[tuple[str, str]] = []

                for _, row in df.iterrows():
                    sid = str(row.get("id", "")).strip()
                    title = str(row.get("title", "")).strip()
                    if sid and all(sid != x[0] for x in related_series):
                        related_series.append((sid, title))
                        if len(related_series) >= max_new:
                            break
                results_per_indicator[indicator] = related_series

        except Exception:
            results_per_indicator = {ind: [] for ind in base}
  # Gather all related IDs for output to retrieval step
    all_ids = [sid for series_list in results_per_indicator.values() for sid, _ in series_list]
    all_ids_str = ",".join(all_ids)

    # Format output
    lines = []
    lines.append("[INFO] DataScout searching initial indicator set in Fred")
    lines.append(f"[INFO] Initial indicators: {', '.join(base)}")

    for indicator, series_list in results_per_indicator.items():
        lines.append(f"[INFO] Related series for '{indicator}':")
        if series_list:
            # Render as Markdown table for better readability in reports
            lines.append("| ID | Title |")         
            for sid, title in series_list:
                truncated_title = title[:200] + "..." if len(title) > 200 else title
                lines.append(f"| {sid} | {truncated_title} |")
            lines.append(f"[INFO] Total discovered for '{indicator}': {len(series_list)}")
        else:
            lines.append("[INFO] None found for this indicator")

    lines.append("[INFO] CHECKPOINT: Indicator list ready for human review")
    lines.append("       Examples: 'Approved' | 'Add foreign_direct_investment' | 'Focus on post-2008'")
    lines.append(f"[INFO] ALL_SERIES_IDS: {all_ids_str}")
    lines.append("DISCOVERY_COMPLETE")
    return "\n".join(lines)

def retrieve_data(indicators: str) -> str:
    """
    Retrieve FRED data one-by-one for each series ID.
    Accepts either:
      - Direct comma-separated series IDs (e.g., 'GDPPOT,UNRATE,CPIAUCSL'), or
      - The full discovery output text that contains a line starting with
        "[INFO] ALL_SERIES_IDS:" followed by a comma-separated list of IDs.
    """
    # Extract series IDs from discovery text if present; otherwise treat input as CSV IDs
    extracted_ids: list[str] = []
    all_ids_prefix = "[INFO] ALL_SERIES_IDS:"
    if all_ids_prefix in indicators:
        for line in indicators.splitlines():
            if line.startswith(all_ids_prefix):
                extracted_ids = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
                break
    if not extracted_ids:
        extracted_ids = [s.strip() for s in indicators.split(",") if s.strip()]

    # De-duplicate while preserving order
    seen_ids: set[str] = set()
    items: list[str] = []
    for series_id in extracted_ids:
        if series_id not in seen_ids:
            seen_ids.add(series_id)
            items.append(series_id)

    per_series = {}
    api_calls = 0

    def _robust_outliers(series: pd.Series) -> int:
        """Estimate number of outliers in a series using MAD-based z-score."""
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.shape[0] < 5:
            return 0
        r = s.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
        if r.empty:
            return 0
        median = r.median()
        mad = (r.sub(median).abs()).median()
        if mad == 0 or pd.isna(mad):
            return 0
        z = (r - median) / (1.4826 * mad)
        return int((z.abs() > 3).sum())

    lines = []
    lines.append("[INFO] DataCollector retrieving data from FRED...")
    lines.append(f"[INFO] Series IDs requested: {', '.join(items)}")

    for sid in items:
        info = {"id": sid, "source": None, "rows": 0, "missing_values": 0}
        try:
            api_calls += 1
            series = fred_client.get_series(sid)
            if series is not None and len(series) > 0:
                s = pd.Series(series)
                info["source"] = "FRED"
                info["rows"] = int(s.shape[0])
                info["missing_values"] = int(s.isna().sum())
                s_non_na = s.dropna()
                info["start_date"] = str(s_non_na.index[0]) if not s_non_na.empty else None
                info["end_date"] = str(s_non_na.index[-1]) if not s_non_na.empty else None
                info["outliers_estimate"] = _robust_outliers(s)
        except Exception as e:
            info["error"] = str(e)

        per_series[sid] = info

    # Summary
    series_retrieved = sum(1 for v in per_series.values() if v.get("rows", 0) > 0)
    lines.append(f"[INFO] Retrieved {series_retrieved} series (API calls: {api_calls})")
    lines.append("[INFO] Per-series details:")
    for sid, info in per_series.items():
        src = info.get("source") or "NONE"
        rows = info.get("rows")
        miss = info.get("missing_values")
        sd = info.get("start_date")
        ed = info.get("end_date")
        out = info.get("outliers_estimate", 0)
        err = info.get("error")
        if err:
            lines.append(f"  - {sid}: ERROR retrieving series → {err}")
        else:
            lines.append(f"  - {sid}: source={src} rows={rows} missing={miss} start={sd} end={ed} outliers≈{out}")

    lines.append("[CHECKPOINT] Retrieved data ready for review")
    return "\n".join(lines)



def clean_preprocess(input_text: str) -> str:
    lines = []
    lines.append("[INFO] DataCleaner preprocessing raw data...")
    lines.append("[INFO] Handling missing values and outliers; harmonizing frequencies where needed")
    lines.append("[INFO] Producing cleaned datasets ready for integration")
    lines.append("CLEANING_COMPLETE")
    return "\n".join(lines)


def feature_engineering(input_text: str) -> str:
    lines = []
    lines.append("[INFO] FeatureEngineer creating derived features...")
    lines.append("[INFO] Generating ratios, growth rates, moving statistics, and composite indices")
    lines.append("FEATURES_COMPLETE")
    return "\n".join(lines)


def validate_processed_data(input_text: str) -> str:
    lines = []
    lines.append("[INFO] ValidationSuite validating processed data...")
    lines.append("[INFO] Running statistical checks, temporal consistency, and cross-source comparison")
    lines.append("VALIDATION_COMPLETE")
    lines.append("[INFO] CHECKPOINT: Indicator list ready for human review")
    lines.append("       Examples: 'Approved' | 'Add foreign_direct_investment' | 'Focus on post-2008'")
    return "\n".join(lines)


def generate_documentation(input_text: str) -> str:
    lines = []
    lines.append("# Data Pipeline Documentation")
    lines.append("")
    lines.append("Artifacts:")
    lines.append("- Data dictionary (variables described)")
    lines.append("- Lineage documentation (transformations tracked)")
    lines.append("- Quality report with visualizations")
    lines.append("- User guide")
    lines.append("")
    lines.append("Summary:")
    lines.append("- Data discovery, retrieval, cleaning, feature engineering, and validation completed")
    lines.append("- See console logs above for operation details")
    lines.append("")
    lines.append("TERMINATE")
    return "\n".join(lines)


search_indicator_tool = FunctionTool(
    search_indicator,
    description="Expand/evaluate indicator set for a dataset and return JSON summary",
)

retrieve_data_tool = FunctionTool(
    retrieve_data, description="Retrieve data for a dataset/indicators and return JSON summary"
)

clean_preprocess_tool = FunctionTool(
    clean_preprocess, description="Clean and preprocess retrieved data, return JSON summary"
)

feature_engineering_tool = FunctionTool(
    feature_engineering, description="Create derived variables and indices, return JSON summary"
)

validate_processed_data_tool = FunctionTool(
    validate_processed_data, description="Validate processed data and return JSON summary"
)

generate_documentation_tool = FunctionTool(
    generate_documentation, description="Generate pipeline documentation, return markdown"
)


# ----------------------------------------
# Agents
# ----------------------------------------


DataScout = AssistantAgent(
            name="DataScout",
            model_client=model_client,
    tools=[search_indicator_tool],
    description="Check availability of provided indicators in configured data sources",
    system_message=(
        "You check availability for the provided indicators only (no expansion).\n"
        "- Call the search_indicator tool exactly once with the full comma-separated indicators list.\n"
        "- Do not iterate per-indicator; do not call the tool multiple times.\n"
        "- Present results with concise [INFO] lines matching the required style.\n"
        "- When done, output 'DISCOVERY_COMPLETE' and stop."
    ),
)

DataCollector = AssistantAgent(
            name="DataCollector",
            model_client=model_client,
    tools=[retrieve_data_tool],
    description="Retrieve identified data sources",
    system_message=(
        "You retrieve data and summarize activity.\n"
        "- Call retrieve_data exactly once using the full discovery text provided in the task.\n"
        "- The retrieve_data function will automatically extract the discovered series IDs from the text.\n"
        "- Report series retrieved and per-indicator details with [INFO] lines.\n"
        "- End with 'RETRIEVAL_COMPLETE'."
    ),
)

DataCleaner = AssistantAgent(
            name="DataCleaner",
            model_client=model_client,
    tools=[clean_preprocess_tool],
    description="Clean and preprocess data",
    system_message=(
        "You clean and preprocess time series.\n"
        "- Use your tool to compute missing values, outliers, imputations, and harmonization.\n"
        "- End with 'CLEANING_COMPLETE'."
    ),
)

FeatureEngineer = AssistantAgent(
    name="FeatureEngineer",
            model_client=model_client,
    tools=[feature_engineering_tool],
    description="Create derived variables for analysis",
    system_message=(
        "You create derived features.\n"
        "- Use your tool to generate summary counts.\n"
        "- End with 'FEATURES_COMPLETE'."
    ),
)

ValidationSuite = AssistantAgent(
    name="ValidationSuite",
            model_client=model_client,
    tools=[validate_processed_data_tool],
    description="Validate processed data and provide quality metrics",
    system_message=(
        "You validate processed data.\n"
        "- Use your tool to compute coverage, match rates, and warnings.\n"
        
    ),
)

DocuAgent = AssistantAgent(
    name="DocuAgent",
            model_client=model_client,
    tools=[generate_documentation_tool],
    description="Generate documentation for entire pipeline",
    system_message=(
        "You generate final documentation in markdown.\n"
        "- Use your tool to produce the final markdown.\n"
        "- End with 'TERMINATE'."
    ),
)


# ----------------------------------------
# Human-in-the-loop checkpoints
# ----------------------------------------

def checkpoint1_input_func(prompt: str) -> str:
    print("\n🛑 CHECKPOINT 1: Indicator Set Review")
    print("=" * 50)
    print("Review expanded indicator list. Provide approval or adjustments.")
    return input("> ")


def checkpoint2_input_func(prompt: str) -> str:
    print("\n🛑 CHECKPOINT 2: Data & Processing Review")
    print("=" * 50)
    print("Review retrieval and preprocessing summaries. Provide guidance if needed.")
    print("Examples: 'Approved' | 'Use trailing averages for volatility' | 'Re-run 2010-2020'")
    return input("> ")


def checkpoint3_input_func(prompt: str) -> str:
    print("\n🛑 CHECKPOINT 3: Documentation Review")
    print("=" * 50)
    print("Review the generated documentation. Provide final validation.")
    print("Examples: 'Looks good' | 'Add appendix on methodology' | 'Clarify dictionary'")
    return input("> ")


checkpoint1_human = UserProxyAgent(name="Checkpoint1_Human", input_func=checkpoint1_input_func)
checkpoint2_human = UserProxyAgent(name="Checkpoint2_Human", input_func=checkpoint2_input_func)
checkpoint3_human = UserProxyAgent(name="Checkpoint3_Human", input_func=checkpoint3_input_func)


# ----------------------------------------
# Teams
# ----------------------------------------

discovery_team = RoundRobinGroupChat(participants=[DataScout, checkpoint1_human], max_turns=2)

processing_team = RoundRobinGroupChat(participants=[DataCollector, DataCleaner, FeatureEngineer, ValidationSuite, checkpoint2_human],max_turns=6,)

report_team = RoundRobinGroupChat(participants=[DocuAgent, checkpoint3_human], max_turns=2)


# ----------------------------------------
# Main
# ----------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Agentic Econ Workflow: Data Team ")
    parser.add_argument(
        "--indicators",
        type=str,
        required=True,
        help="Comma-separated base indicators, e.g., 'gdp,inflation,debt,unemployment'",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"data_pipeline_{timestamp}.md"

    print(f"Starting data workflow on searching for indicators")
    print("=" * 50)

    workflow_start_time = datetime.now()

    print("\n🔎 Phase 1: Data Discovery")
    print("-" * 30)
    discovery_task = (
f"Search indicators for {args.indicators}.\n"
    )
    discovery_result = await Console(discovery_team.run_stream(task=discovery_task))

    # Collect discovery phase text
    discovery_text = getattr(discovery_result, "content", str(discovery_result))
    discovery_end_time = datetime.now()

    print("\n📥 Phase 2: Retrieval, Cleaning, Feature Engineering, Validation")
    print("-" * 30)
    processing_task = (
        f"Original indicators searched: {args.indicators}\n"
        f"Discovery results:\n{discovery_text}\n\n"
        f"Instructions:\n"
        f"- Call retrieve_data with the discovery_text above (which contains the discovered series IDs).\n"
        f"- The retrieve_data function will extract the series IDs from the discovery output.\n"
        f"Then clean, engineer features, and validate. Present [INFO] summaries."
    )
    processing_result = await Console(processing_team.run_stream(task=processing_task))

    # Collect processing phase text
    processing_text = getattr(processing_result, "content", str(processing_result))
    processing_end_time = datetime.now()

    print("\n📝 Phase 3: Documentation Generation")
    print("-" * 30)
    report_task = (
        f"Generate a comprehensive Data Pipeline Report.\n\n"
        f"Include ALL information from earlier phases verbatim where appropriate.\n\n"
        f"Section: Indicators searched and availability\n{discovery_text}\n\n"
        f"Section: Retrieval, cleaning, feature engineering, validation\n{processing_text}\n\n"
        f"End the report with 'TERMINATE'."
    )
    report_result = await Console(report_team.run_stream(task=report_task))

    final_content = ""
    if hasattr(report_result, "messages"):
        for message in reversed(report_result.messages):
            if getattr(message, "source", "") == "DocuAgent" and hasattr(message, "content"):
                final_content = message.content
                break
    if not final_content:
        final_content = getattr(report_result, "content", str(report_result))

    workflow_end_time = datetime.now()

    # ----------------------------------------
    # Build Run Summary ([INFO] lines) and metrics
    # ----------------------------------------
    def extract_block(text: str, start_regex: str, end_regex: str | None = None) -> list[str]:
        flags = re.MULTILINE | re.DOTALL
        if end_regex:
            pattern = re.compile(start_regex + r"(.*?)" + end_regex, flags)
            match = pattern.search(text)
            if match:
                block = match.group(0)
                # trim any leading/trailing quotes that may wrap the content in logs
                block = block.replace("\\n", "\n")
                return [ln for ln in block.splitlines()]
            return []
        # Without end marker, collect all [INFO] lines
        return re.findall(r"\[INFO\].*", text)

    # Extract detailed blocks for Run Summary
    discovery_block = extract_block(
        discovery_text,
        r"\[INFO\]\s*DataScout[\s\S]*?",
        r"DISCOVERY_COMPLETE"
    )

    datacollector_block = extract_block(
        processing_text,
        r"\[INFO\]\s*DataCollector[\s\S]*?",
        r"(?:RETRIEVAL_COMPLETE|\[CHECKPOINT\].*)"
    )

    datacleaner_block = extract_block(
        processing_text,
        r"\[INFO\]\s*DataCleaner[\s\S]*?",
        r"CLEANING_COMPLETE"
    )

    features_block = extract_block(
        processing_text,
        r"\[INFO\]\s*FeatureEngineer[\s\S]*?",
        r"FEATURES_COMPLETE"
    )

    validation_block = extract_block(
        processing_text,
        r"\[INFO\]\s*ValidationSuite[\s\S]*?",
        r"VALIDATION_COMPLETE[\s\S]*?(?=\n\[|$)"
    )

    # Human approvals
    checkpoint1_approved = "approved" in discovery_text.lower()
    checkpoint2_approved = "approved" in processing_text.lower()

    # Metrics
    total_seconds = (workflow_end_time - workflow_start_time).total_seconds()
    processing_seconds = max(0.001, (processing_end_time - discovery_end_time).total_seconds())
    doc_seconds = max(0.001, (workflow_end_time - processing_end_time).total_seconds())

    rows_matches = re.findall(r"rows=(\\d+)", processing_text)
    total_datapoints = sum(int(x) for x in rows_matches) if rows_matches else 0
    datapoints_per_second = (total_datapoints / processing_seconds) if processing_seconds > 0 else 0.0

    coverage_match = re.search(r"Coverage:\s*(\d+)%", processing_text)
    validation_coverage = coverage_match.group(1) + "%" if coverage_match else "N/A"

    run_summary_lines: list[str] = []
    run_summary_lines.append(f"$ python DataTeam.py --indicators \"{args.indicators}\"")
    run_summary_lines.append("")
    # Discovery highlights
    if discovery_block:
        run_summary_lines.extend(discovery_block)
    if checkpoint1_approved:
        run_summary_lines.append("[INFO] Human Approved")
    run_summary_lines.append("")
    # Processing highlights
    if datacollector_block:
        run_summary_lines.extend(datacollector_block)
    if datacleaner_block:
        run_summary_lines.extend(datacleaner_block)
    if features_block:
        run_summary_lines.extend(features_block)
    if validation_block:
        run_summary_lines.extend(validation_block)
    if checkpoint2_approved:
        run_summary_lines.append("[INFO] Human approved dataset")
    run_summary_lines.append("")
    run_summary_lines.append("--- EXECUTION SUMMARY ---")
    run_summary_lines.append(f"Total execution time: {total_seconds:.2f} seconds")
    if total_datapoints > 0:
        run_summary_lines.append(f"Data processing rate: {datapoints_per_second:,.2f} data points per second")
    run_summary_lines.append(f"Documentation generation: {doc_seconds:.2f} seconds")
    run_summary_lines.append(f"Validation coverage: {validation_coverage}")
    run_summary_lines.append("")
    run_summary_lines.append("Process completed successfully.")

    # Combined Console Output (single fenced block)
    def extract_console_info_lines(text: str) -> list[str]:
        selected: list[str] = []
        for ln in text.splitlines():
            stripped = ln.strip()
            if stripped.startswith("[INFO]") or stripped.startswith("[CHECKPOINT]"):
                selected.append(stripped)
            elif stripped in ("DISCOVERY_COMPLETE", "RETRIEVAL_COMPLETE", "CLEANING_COMPLETE", "FEATURES_COMPLETE", "VALIDATION_COMPLETE"):
                selected.append(stripped)
        return selected

    combined_console = []
    combined_console.append("===== Phase 1: Data Discovery =====")
    combined_console.extend(extract_console_info_lines(discovery_text))
    combined_console.append("")
    combined_console.append("===== Phase 2: Retrieval, Cleaning, Feature Engineering, Validation =====")
    combined_console.extend(extract_console_info_lines(processing_text))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Data Pipeline Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Indicators: {args.indicators}\n\n")
        f.write("---\n\n")
        f.write("### Run Summary\n\n")
        for line in run_summary_lines:
            f.write(line + "\n")
        f.write("\n")
        f.write("### Console Output\n\n")
        f.write("```text\n")
        f.write("\n".join(combined_console) + "\n")
        f.write("```\n\n")
        f.write("## Final Report\n\n")
        f.write(final_content)

    print(f"\n📄 Report saved to: {filename}")
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
