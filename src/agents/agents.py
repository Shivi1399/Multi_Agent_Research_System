from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

llm1 = ChatOllama(
    model="qwen3:8b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

llm2 = ChatOllama(
    model="deepseek-r1:8b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)




def build_search_agent():
    tools = [web_search]
    agent = create_agent(
        model=llm1,
        tools=tools,
    )
    return agent


def build_scrape_agent():
    tools = [scrape_url]
    agent = create_agent(
        model=llm2,
        tools=tools,
    )
    return agent


# writer chain
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher and writer. 
            You will be given a topic and a set of research notes. 
            Your task is to write a comprehensive, well-structured, and informative report based on the provided notes. 
            Ensure that the reports are clear, concise, and engaging for the reader.""",
        ),
        (
            "human",
            """Write a detailed report on the following topic: {topic}. 
            Use the following research notes to support your writing: {research_notes}. 
            Ensure that the report is well-organized, with appropriate headings and subheadings. 
            Provide citations where necessary.
            
            Topic: {topic}

            Research Notes: {research_notes}

            Structure the report as:
            - Introduction
            - Key Findings
            - Conclusions
            - Sources and References

            Be detailed, factual and professional in your writing. 
            Avoid personal opinions and ensure that the content is suitable for an academic or professional audience.
            """,
        ),
    ]
)

writer_chain = writer_prompt | llm1 | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a meticulous editor and fact-checking critic reviewing research reports.
            You will be given the original topic, the research notes the report was based on, and a draft report.
            Your task is to critically evaluate the draft and identify concrete issues that need fixing.
            Focus on factual accuracy against the research notes, completeness, clarity, structure, and citation quality.
            Be specific and actionable in your feedback — do not rewrite the report yourself.""",
        ),
        (
            "human",
            """Review the following draft report against the topic and research notes it was supposed to be based on.

            Topic: {topic}

            Research Notes: {research_notes}

            Draft Report: {report}

            Evaluate the draft on:
            - Factual accuracy: does every claim trace back to something in the research notes? Flag anything unsupported or contradicted.
            - Completeness: are there important points from the research notes that the report omits?
            - Structure and clarity: is it well-organized with clear headings, and easy to follow?
            - Citations: are sources referenced appropriately and consistently?

            Respond in this format:
            Verdict: APPROVED or NEEDS_REVISION
            Issues:
            - <specific issue 1, with the exact section/claim it relates to>
            - <specific issue 2>
            ...
            (If APPROVED, state "No significant issues found." under Issues.)
            """,
        ),
    ]
)

critic_chain = critic_prompt | llm2 | StrOutputParser()

reviser_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher and writer revising a report in response to editorial feedback.
            You will be given the topic, the original research notes, the previous draft, and a critic's feedback.
            Your task is to produce a revised, complete report that fixes every issue the critic raised
            while preserving what already worked in the previous draft.""",
        ),
        (
            "human",
            """Revise the following draft report to address the critic's feedback.

            Topic: {topic}

            Research Notes: {research_notes}

            Previous Draft: {report}

            Critic Feedback: {feedback}

            Produce a complete, standalone revised report (not just a list of changes), keeping the structure:
            - Introduction
            - Key Findings
            - Conclusions
            - Sources and References

            Be detailed, factual and professional in your writing.
            Avoid personal opinions and ensure that the content is suitable for an academic or professional audience.
            """,
        ),
    ]
)

reviser_chain = reviser_prompt | llm1 | StrOutputParser()
