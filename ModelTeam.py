import asyncio
import os
import argparse
from datetime import datetime
import io
import sys

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# ------------------------------
# Utilities
# ------------------------------

def info(message: str) -> None:
    print(f"[INFO] {message}")


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


def _format_duration(start: datetime, end: datetime) -> str:
    total_seconds = int((end - start).total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _write_markdown_log(
    console_text: str,
    model_type: str,
    focus: str,
    start_time: datetime,
    end_time: datetime,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    log_filename = f"modeling_log_{timestamp}.md"

    lines: list[str] = []
    lines.append("### Run Summary")
    lines.append("")
    lines.append("[INFO] Starting modeling process")
    lines.append("[INFO] Initializing agents... Done.")
    lines.append("[INFO] Theorist Agent developing theoretical framework... Complete.")
    lines.append("[INFO] CHECKPOINT: Theoretical framework ready for human review")
    lines.append("[INFO] ModelDesigner Agent translating theory to mathematical formulation... Complete.")
    lines.append("[INFO] CHECKPOINT: Mathematical model ready for human review")
    lines.append("[INFO] Calibrator Agent proposing/performing estimation and robustness... Complete.")
    lines.append("[INFO] CHECKPOINT: Calibration results ready for human review")
    lines.append("")
    lines.append("--- EXECUTION SUMMARY ---")
    lines.append(f"Total execution time: {_format_duration(start_time, end_time)}")
    lines.append(f"Model type: {model_type}")
    lines.append(f"Focus: {focus}")
    lines.append("Stages completed: Theory, Modeling, Calibration")
    lines.append("Process completed successfully.")
    lines.append("")
    lines.append("### Console Output")
    lines.append("")
    lines.append("```text")
    lines.append(console_text.rstrip())
    lines.append("```")

    with open(log_filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return log_filename


# No hard-coded tools. All reasoning, model selection, derivations, and calibration design are LLM-driven.


# ------------------------------
# Agent Definitions
# ------------------------------

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key)

theorist_agent = AssistantAgent(
    name="Theorist",
    model_client=model_client,
    description="Selects or develops the theoretical framework that underpins the economic model.",
    system_message=(
        "You are Theorist.\n"
        "Responsibilities:\n"
        "- Select/develop the theoretical framework; define assumptions; choose functional forms; justify economic rationale.\n"
        "- Consider any economic model class (DSGE, OLG, IO, trade, urban, environmental, labor, development, finance, etc.).\n"
        "- Ensure internal consistency and alignment with literature.\n\n"
        "Output strictly in this structure:\n"
        "1) Framework: <name + brief description>\n"
        "2) Assumptions: <bulleted list>\n"
        "3) Mechanisms: <channels and intuition>\n"
        "4) Candidate functional forms: <options>\n"
        "5) Notes for modeling: <constraints like ZLB, heterogeneity, frictions>\n\n"
        "End with THEORY_STAGE_COMPLETE."
    ),
)

model_designer_agent = AssistantAgent(
    name="ModelDesigner",
    model_client=model_client,
    description="Translates the theoretical framework into precise mathematical and computational models.",
    system_message=(
        "You are ModelDesigner.\n"
        "Responsibilities:\n"
        "- Formalize equations; define variables/parameters; specify solution/algorithmic approach.\n"
        "- Handle constraints like ZLB, occasionally binding constraints, heterogeneity, frictions, or search.\n"
        "- Provide solver choice rationale (perturbation, projection, value function iteration, structural IO, GMM, MLE, Bayesian).\n\n"
        "Output strictly in this structure:\n"
        "1) Variables & Parameters\n"
        "2) Equations (numbered)\n"
        "3) Solution/Simulation algorithm (steps or pseudocode/code)\n"
        "4) Implementation notes\n\n"
        "End with MODEL_STAGE_COMPLETE."
    ),
)

calibrator_agent = AssistantAgent(
    name="Calibrator",
    model_client=model_client,
    description="Designs and executes calibration/estimation and sensitivity/robustness analysis.",
    system_message=(
        "You are Calibrator.\n"
        "Responsibilities:\n"
        "- Propose calibration/estimation approach (e.g., Bayesian with priors, GMM, simulated method of moments, MLE).\n"
        "- Specify data requirements and sources; if unavailable, propose synthetic-data strategy.\n"
        "- Report estimated parameters, diagnostics (R-hat/ESS for Bayesian, fit metrics), and sensitivity/robustness plan.\n\n"
        "Output strictly in this structure:\n"
        "1) Data & Priors (or moments/targets)\n"
        "2) Estimation/Calibration procedure\n"
        "3) Results & Diagnostics\n"
        "4) Sensitivity & Robustness plan (and findings if applicable)\n\n"
        "End with CALIBRATION_STAGE_COMPLETE and finally TERMINATE."
    ),
)


# ------------------------------
# Human-in-the-Loop Checkpoints
# ------------------------------

def checkpoint1_input_func(prompt: str) -> str:
    print("\n🛑 CHECKPOINT 1: Theoretical Framework Review")
    print("=" * 50)
    print("Review the selected framework, assumptions, and functional forms.")
    print("Examples: 'Approved' | 'Add financial frictions' | 'Focus on heterogeneous agents'")
    print("\nYour feedback:")
    return input("> ").strip()


def checkpoint2_input_func(prompt: str) -> str:
    print("\n🛑 CHECKPOINT 2: Mathematical Model Review")
    print("=" * 50)
    print("Validate equations, variables, and computational algorithms.")
    print("Examples: 'Approved' | 'Add ZLB constraint' | 'Refine investment block'")
    print("\nYour feedback:")
    return input("> ").strip()


def checkpoint3_input_func(prompt: str) -> str:
    print("\n🛑 CHECKPOINT 3: Calibration Results Review")
    print("=" * 50)
    print("Evaluate parameter estimates, fit metrics, and sensitivity analysis.")
    print("Examples: 'Approved' | 'Run robustness for fiscal multipliers' | 'Tighten priors'")
    print("\nYour feedback:")
    return input("> ").strip()


checkpoint1_human = UserProxyAgent(name="Checkpoint1_Human", input_func=checkpoint1_input_func)
checkpoint2_human = UserProxyAgent(name="Checkpoint2_Human", input_func=checkpoint2_input_func)
checkpoint3_human = UserProxyAgent(name="Checkpoint3_Human", input_func=checkpoint3_input_func)


# ------------------------------
# Teams
# ------------------------------

theory_team = RoundRobinGroupChat(
    participants=[theorist_agent, checkpoint1_human], max_turns=4
)

modeling_team = RoundRobinGroupChat(
    participants=[model_designer_agent, checkpoint2_human], max_turns=4
)

calibration_team = RoundRobinGroupChat(
    participants=[calibrator_agent, checkpoint3_human], max_turns=4
)


# ------------------------------
# Main entry
# ------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Model Team with Human-in-the-Loop Checkpoints")
    parser.add_argument("--model_type", type=str, required=True, help="Model class, e.g., DSGE, RBC, NK, static")
    parser.add_argument("--focus", type=str, required=True, help="Topical focus, e.g., fiscal policy impacts")
    parser.add_argument("--periods", type=int, default=240, help="Number of periods for synthetic data if used")
    args = parser.parse_args()

    topic = f"{args.model_type} model with {args.focus} focus"
    # Prepare console tee to capture full transcript
    buffer = io.StringIO()
    tee = StdoutTee(buffer)
    original_stdout = sys.stdout
    sys.stdout = tee

    start_time = datetime.now()
    info(f"Starting model development process for: {topic}")
    info("Initializing agents... Done.")

    # Phase 1: Theoretical Framework
    info("Theorist Agent developing theoretical framework...")
    theory_task = (
        f"Develop a comprehensive theoretical framework for a {args.model_type} focusing on {args.focus}. "
        "Follow your output structure and conclude with THEORY_STAGE_COMPLETE."
    )
    await Console(theory_team.run_stream(task=theory_task))
    info("CHECKPOINT: Theoretical framework ready for human review")

    # Phase 2: Mathematical Model
    info("ModelDesigner Agent translating theory to mathematical formulation...")
    modeling_task = (
        "Translate the reviewed theory into a complete mathematical specification and solution/simulation algorithm. "
        "Follow your output structure and conclude with MODEL_STAGE_COMPLETE."
    )
    await Console(modeling_team.run_stream(task=modeling_task))
    info("CHECKPOINT: Mathematical model ready for human review")

    # Phase 3: Calibration
    info("Calibrator Agent proposing/performing estimation and robustness...")
    calibration_task = (
        f"Design and execute an estimation/calibration for the model. If real data is unavailable, propose a synthetic-data strategy with {args.periods} periods. "
        "Report parameters, diagnostics, and sensitivity/robustness. "
        "Follow your output structure. Conclude with CALIBRATION_STAGE_COMPLETE and then TERMINATE."
    )
    await Console(calibration_team.run_stream(task=calibration_task))
    info("CHECKPOINT: Calibration results ready for human review")

    # Restore stdout and write structured markdown log
    sys.stdout = original_stdout
    end_time = datetime.now()
    log_filename = _write_markdown_log(
        console_text=buffer.getvalue(),
        model_type=args.model_type,
        focus=args.focus,
        start_time=start_time,
        end_time=end_time,
    )

    info(f"Modeling log saved to: {log_filename}")
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())