"""Streamlit interface for the multi-agent research pipeline.

Run from the project directory with:
    streamlit run app.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable

import streamlit as st

try:
    import psutil
except ImportError:  # The UI remains usable before requirements are reinstalled.
    psutil = None


_previous_windows_cpu_times: tuple[int, int] | None = None


st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)


def run_pipeline(topic: str, status_callback: Callable[[str, str, int], None]) -> dict:
    """Import lazily so configuration/import errors can be shown in the UI."""
    from src.pipelines.pipeline import run_research_pipeline

    return run_research_pipeline(topic, status_callback=status_callback)


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def sample_system_usage() -> tuple[float | None, float | None]:
    """Return current whole-system CPU and RAM utilization percentages."""
    if psutil is not None:
        return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
    if sys.platform == "win32":
        return sample_windows_system_usage()
    return None, None


def sample_windows_system_usage() -> tuple[float | None, float | None]:
    """Read Windows CPU and memory counters without third-party packages."""
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    def filetime_value(value: wintypes.FILETIME) -> int:
        return (value.dwHighDateTime << 32) | value.dwLowDateTime

    try:
        kernel32 = ctypes.windll.kernel32
        idle_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel32.GetSystemTimes(
            ctypes.byref(idle_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None, None

        idle = filetime_value(idle_time)
        total = filetime_value(kernel_time) + filetime_value(user_time)

        global _previous_windows_cpu_times
        cpu_percent = 0.0
        if _previous_windows_cpu_times is not None:
            previous_idle, previous_total = _previous_windows_cpu_times
            idle_delta = idle - previous_idle
            total_delta = total - previous_total
            if total_delta > 0:
                cpu_percent = 100.0 * (1.0 - idle_delta / total_delta)
                cpu_percent = min(max(cpu_percent, 0.0), 100.0)
        _previous_windows_cpu_times = (idle, total)

        memory = MemoryStatus()
        memory.dwLength = ctypes.sizeof(MemoryStatus)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return cpu_percent, None

        return cpu_percent, float(memory.dwMemoryLoad)
    except (AttributeError, OSError, ValueError):
        return None, None


def execute_with_live_status(topic: str) -> dict:
    """Run the blocking pipeline while keeping live Streamlit metrics responsive."""
    updates: queue.Queue[tuple[str, str, int]] = queue.Queue()
    outcome: dict = {}

    def status_callback(agent: str, message: str, progress: int) -> None:
        updates.put((agent, message, progress))

    def worker_target() -> None:
        try:
            outcome["result"] = run_pipeline(topic, status_callback)
        except Exception as exc:  # Pass the exception back to the Streamlit thread.
            outcome["error"] = exc

    status_slot = st.empty()
    progress_bar = st.progress(0)
    cpu_column, ram_column, time_column = st.columns(3)
    cpu_slot = cpu_column.empty()
    ram_slot = ram_column.empty()
    time_slot = time_column.empty()
    activity_slot = st.empty()

    started_at = time.perf_counter()
    peak_cpu: float | None = None
    peak_ram: float | None = None
    activity: list[str] = []
    current_status = ("System", "Starting the research pipeline...", 0)

    # Prime psutil's non-blocking CPU sampler before starting the polling loop.
    sample_system_usage()
    worker = threading.Thread(target=worker_target, daemon=True)
    worker.start()

    while worker.is_alive():
        while True:
            try:
                current_status = updates.get_nowait()
                agent, message, progress = current_status
                activity.append(f"- **{agent}:** {message}")
                progress_bar.progress(min(max(progress, 0), 100))
            except queue.Empty:
                break

        agent, message, _ = current_status
        status_slot.info(f"**{agent}** — {message}")

        cpu_usage, ram_usage = sample_system_usage()
        if cpu_usage is not None and ram_usage is not None:
            peak_cpu = max(peak_cpu or 0.0, cpu_usage)
            peak_ram = max(peak_ram or 0.0, ram_usage)
            cpu_slot.metric("CPU usage", f"{cpu_usage:.1f}%")
            ram_slot.metric("RAM usage", f"{ram_usage:.1f}%")
        else:
            cpu_slot.metric("CPU usage", "N/A")
            ram_slot.metric("RAM usage", "N/A")
        time_slot.metric("Running time", format_duration(time.perf_counter() - started_at))

        if activity:
            activity_slot.markdown("#### Agent activity\n" + "\n".join(activity))
        time.sleep(0.5)

    worker.join()
    while not updates.empty():
        current_status = updates.get_nowait()
        agent, message, progress = current_status
        activity.append(f"- **{agent}:** {message}")
        progress_bar.progress(min(max(progress, 0), 100))

    elapsed = time.perf_counter() - started_at
    agent, message, _ = current_status
    time_slot.metric("Execution time", format_duration(elapsed))
    if activity:
        activity_slot.markdown("#### Agent activity\n" + "\n".join(activity))

    if "error" in outcome:
        status_slot.error(f"**{agent}** — The pipeline stopped because of an error.")
        raise outcome["error"]

    status_slot.success(f"**{agent}** — {message}")
    result = outcome["result"]
    result["execution_time"] = elapsed
    result["peak_cpu"] = peak_cpu
    result["peak_ram"] = peak_ram
    return result


def build_download(result: dict) -> str:
    """Create one Markdown document containing the report and review metadata."""
    return (
        f"# {result['topic']}\n\n"
        f"{result['report']}\n\n"
        "---\n\n"
        "## Editorial review\n\n"
        f"Revision passes: {result['revisions']}\n\n"
        f"{result['critique']}\n\n"
        "---\n\n"
        "## Research notes\n\n"
        f"{result['research_notes']}\n"
    )


st.title("🔎 Multi-Agent Research System")
st.caption(
    "Search the web, inspect source pages, draft a report, and run an editorial "
    "review with local Ollama models."
)

with st.sidebar:
    st.header("Before you begin")
    st.markdown(
        "- Make sure Ollama is running.\n"
        "- Install the `qwen3:8b` and `deepseek-r1:8b` models.\n"
        "- Add `TAVILY_API_KEY` to the project's `.env` file."
    )
    st.info("Research can take several minutes because sources are processed one at a time.")

with st.form("research_form"):
    topic = st.text_area(
        "Research topic",
        value=st.session_state.get("topic", ""),
        placeholder="Example: The impact of artificial intelligence on the job market",
        height=100,
        help="A focused topic usually produces a more useful report.",
    )
    submitted = st.form_submit_button("Start research", type="primary")

if submitted:
    clean_topic = topic.strip()
    if not clean_topic:
        st.warning("Enter a research topic before starting.")
    else:
        st.session_state["topic"] = clean_topic
        st.session_state.pop("research_result", None)
        try:
            st.session_state["research_result"] = execute_with_live_status(clean_topic)
        except Exception as exc:
            st.error("The research pipeline could not complete.")
            st.exception(exc)

result = st.session_state.get("research_result")
if result:
    st.success(
        f"Research complete. The report went through {result['revisions']} "
        f"revision pass{'es' if result['revisions'] != 1 else ''}."
    )

    summary_columns = st.columns(3)
    summary_columns[0].metric(
        "Execution time", format_duration(result.get("execution_time", 0))
    )
    peak_cpu = result.get("peak_cpu")
    peak_ram = result.get("peak_ram")
    summary_columns[1].metric(
        "Peak CPU", f"{peak_cpu:.1f}%" if peak_cpu is not None else "N/A"
    )
    summary_columns[2].metric(
        "Peak RAM", f"{peak_ram:.1f}%" if peak_ram is not None else "N/A"
    )

    report_tab, notes_tab, review_tab = st.tabs(
        ["Final report", "Research notes", "Editorial review"]
    )

    with report_tab:
        st.markdown(result["report"])

    with notes_tab:
        st.markdown(result["research_notes"])

    with review_tab:
        st.metric("Revision passes", result["revisions"])
        st.markdown(result["critique"])

    safe_name = "-".join(result["topic"].lower().split())[:60] or "research-report"
    st.download_button(
        "Download full research (.md)",
        data=build_download(result),
        file_name=f"{safe_name}.md",
        mime="text/markdown",
    )
