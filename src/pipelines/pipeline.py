import re
from collections.abc import Callable

from src.agents.agents import (
    build_scrape_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
    reviser_chain,
)

# Matches http(s) URLs so we can hand the search results' links to the scrape agent
URL_PATTERN = re.compile(r"https?://[^\s)\]]+")

MAX_URLS_TO_SCRAPE = 3
MAX_REVISIONS = 2
StatusCallback = Callable[[str, str, int], None]


def _emit_status(
    callback: StatusCallback | None, agent: str, message: str, progress: int
) -> None:
    """Send an optional progress update to callers such as the Streamlit UI."""
    if callback:
        callback(agent, message, progress)


def _run_agent(agent, prompt: str) -> str:
    """Invoke a LangGraph ReAct agent with a single user message and return its final reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result["messages"][-1].content


def _gather_research_notes(
    topic: str, status_callback: StatusCallback | None = None
) -> str:
    """Step 1: search the web for the topic, then scrape the most promising URLs for detail."""
    _emit_status(
        status_callback, "Search Agent", "Searching the web for credible sources...", 10
    )
    search_agent = build_search_agent()
    search_notes = _run_agent(
        search_agent,
        f"Use the web_search tool to research the topic: '{topic}'. "
        "Search for it and summarize the most relevant, credible results you find, "
        "including their titles, URLs, and key facts.",
    )

    urls = URL_PATTERN.findall(search_notes)[:MAX_URLS_TO_SCRAPE]

    scraped_notes = []
    if urls:
        scrape_agent = build_scrape_agent()
        for index, url in enumerate(urls, start=1):
            progress = 15 + int((index / len(urls)) * 30)
            _emit_status(
                status_callback,
                "Scrape Agent",
                f"Reading source {index} of {len(urls)}: {url}",
                progress,
            )
            scraped_notes.append(
                _run_agent(
                    scrape_agent,
                    f"Use the scrape_url tool to fetch the content of {url} "
                    f"and summarize the information relevant to the topic: '{topic}'.",
                )
            )

    sections = [f"### Web Search Results\n{search_notes}"]
    for url, notes in zip(urls, scraped_notes):
        sections.append(f"### Scraped Content: {url}\n{notes}")

    return "\n\n".join(sections)


def _is_approved(critique: str) -> bool:
    return "NEEDS_REVISION" not in critique.upper()


def run_research_pipeline(
    topic: str, status_callback: StatusCallback | None = None
) -> dict:
    """
    Run the full research pipeline for a topic: search -> scrape -> write -> critique -> revise.

    Args:
        topic (str): The research topic to write a report on.

    Returns:
        dict: {
            "topic": str,
            "research_notes": str,
            "report": str,       # final (possibly revised) report
            "critique": str,     # critic's feedback on the final report
            "revisions": int,    # number of revision passes performed
        }
    """
    research_notes = _gather_research_notes(topic, status_callback)

    _emit_status(status_callback, "Writer Agent", "Drafting the research report...", 55)
    report = writer_chain.invoke({"topic": topic, "research_notes": research_notes})
    _emit_status(
        status_callback, "Critic Agent", "Reviewing facts, structure, and citations...", 70
    )
    critique = critic_chain.invoke(
        {"topic": topic, "research_notes": research_notes, "report": report}
    )

    revisions = 0
    while not _is_approved(critique) and revisions < MAX_REVISIONS:
        _emit_status(
            status_callback,
            "Reviser Agent",
            f"Revising the report (pass {revisions + 1} of {MAX_REVISIONS})...",
            80 + (revisions * 8),
        )
        report = reviser_chain.invoke(
            {
                "topic": topic,
                "research_notes": research_notes,
                "report": report,
                "feedback": critique,
            }
        )
        revisions += 1
        _emit_status(
            status_callback,
            "Critic Agent",
            f"Reviewing revision {revisions}...",
            88 + (revisions * 4),
        )
        critique = critic_chain.invoke(
            {"topic": topic, "research_notes": research_notes, "report": report}
        )

    _emit_status(status_callback, "System", "Research pipeline complete.", 100)
    return {
        "topic": topic,
        "research_notes": research_notes,
        "report": report,
        "critique": critique,
        "revisions": revisions,
    }
