# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
from IPython.display import Markdown
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path='./fake_resume.md')
semantic_search_resume = MDXSearchTool(mdx='./fake_resume.md')

# from IPython.display import Markdown, display
# display(Markdown("./fake_resume.md"))

# Agent 1: Researcher
researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on "
         "job posting to help job applicants",
    tools = [scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    )
)