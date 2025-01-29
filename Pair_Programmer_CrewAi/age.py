import warnings
warnings.filterwarnings("ignore")
from crewai import Agent, Task, Crew,LLM
from crewai_tools import SerperDevTool,ScrapeWebsiteTool,WebsiteSearchTool,PDFSearchTool,FileReadTool
from IPython.display import Markdown
from langchain_groq import ChatGroq
import getpass
import os
import dotenv
from langchain_community.tools import JinaSearch,DuckDuckGoSearchRun



dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm=LLM(model="groq/llama3-8b-8192",api_key=GROQ_API_KEY)

path="/opt/CodeRepo/Agentic_Ai/Pair_Programmer_CrewAi/AnsumanComputerScienceGraduateResume.pdf"

# search_tool =  JinaSearch()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path=path)

# semantic_search_resume = PDFSearchTool(file_path='./AnsumanComputerScienceGraduateResume.pdf')
print(read_resume)

# Agent 2: Profiler
profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do increditble research on resume,Github Profile and job market "
         "to help them build a professional portfolio website to highlight their skills and projects and stand out in the job market",
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    ),
   tools = [scrape_tool,
             read_resume, ],
   verbose=True
)

profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the resume, GitHub ({github_url}) URLs, Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)

builder_crew=Crew(
    agents=[profiler],
    tasks=[profile_task],
    verbose=True
)

user_input={
    'github_url':"https://github.com/Ansumanbhujabal"
}


result=builder_crew.kickoff(inputs=user_input)

print(result)