# crew_ai.py

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import json
import re


# ==============================
# LLM CONFIG
# ==============================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)


# ==============================
# HELPERS
# ==============================

def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON found in LLM output")
    return json.loads(match.group())


# ==============================
# AGENTS
# ==============================

word_agent = Agent(
    role="Crossword Word Generator",
    goal="Generate real English crossword words",
    backstory="Expert crossword puzzle editor",
    llm=llm,
    verbose=True
)

clue_agent = Agent(
    role="Crossword Clue Writer",
    goal="Write concise crossword clues",
    backstory="Professional crossword writer",
    llm=llm,
    verbose=True
)


# ==============================
# TASKS
# ==============================

def word_task(count):
    return Task(
        description=f"""
Generate {count} English crossword words.

Rules:
- 4–8 letters
- No plurals
- No names
- Uppercase

Return ONLY valid JSON:
{{"words":["HOUSE","RIVER","STONE"]}}
""",
        expected_output="JSON object with key 'words' and a list of strings.",
        agent=word_agent
    )


def clue_task(words):
    return Task(
        description=f"""
Write crossword clues for these words:

{", ".join(words)}

Return ONLY valid JSON:
{{"clues":["Clue 1","Clue 2"]}}
""",
        expected_output="JSON object with key 'clues' and a list of clue strings.",
        agent=clue_agent
    )


# ==============================
# PUBLIC API
# ==============================

def generate_words(count=10):
    print(" CrewAI: Generating words...")

    crew = Crew(
        agents=[word_agent],
        tasks=[word_task(count)],
        verbose=True
    )

    result = crew.kickoff()
    raw = result.raw if hasattr(result, "raw") else str(result)

    print(" RAW OUTPUT:", raw)

    data = extract_json(raw)
    return data["words"]


def generate_clues(words):
    print(" CrewAI: Generating clues...")

    crew = Crew(
        agents=[clue_agent],
        tasks=[clue_task(words)],
        verbose=True
    )

    result = crew.kickoff()
    raw = result.raw if hasattr(result, "raw") else str(result)

    data = extract_json(raw)
    return dict(zip(words, data["clues"]))
