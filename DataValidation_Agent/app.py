import warnings
warnings.filterwarnings("ignore")
from crewai import Agent, Task, Crew,LLM
from crewai_tools import SerperDevTool,ScrapeWebsiteTool,WebsiteSearchTool,EXASearchTool
from crewai.tools import tool
from langchain_community.tools import TavilySearchResults,JinaSearch
from langchain_community.utilities import GoogleSerperAPIWrapper
from IPython.display import Markdown
from langchain_groq import ChatGroq
import getpass
import os
import dotenv
dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
print("The api key is-------------------------->>>>>>>>>>>>>>>>>>")
## Initialize Llama 
# llama-3.3-70b-versatile
# deepseek-r1-distill-llama-70b
llm = ChatGroq(model="groq/llama3-8b-8192",api_key=GROQ_API_KEY)
# llm=LLM(model="groq/llama-3.3-70b-versatile",api_key=GROQ_API_KEY)

# famous_personality_reviewer_agent
# famous_personality_reviewer_task
# famous_personality_agnet
# famous_personality_task
scrape_tool = EXASearchTool()
# search_tool =  JinaSearch()
# scrape_tool=TavilySearchResults(
#     max_results=5,
#     include_answer=True,
#     include_raw_content=True,
#     include_images=True,
#     # search_depth="advanced",
#     # include_domains = []
#     # exclude_domains = []
# )

search = GoogleSerperAPIWrapper()

@tool("GoogleSearchTool")
def Google_search_tool(search_query: str):
    """Performs a search using the GoogleSearchTool."""
    return search().run(search_query)


@tool("Tavilysearchtool")
def Tavily_search_tool(search_query: str):
    """Performs a search using the Tavily search tool."""
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool().run(search_query)



# search_tool=Tavily_search_tool()

famous_personality_reviewer_agent = Agent(role="Senior Content Validator and fact checker",
                          goal='Validate the content with facts and get the most correct content', 
                          backstory="""You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field if {famous_personalities_name} You will give best fact based review, suggestions of new names and feedback """,
                          llm=llm,
                          tools = [scrape_tool,Google_search_tool],
                          verbose=True)


famous_personality_reviewer_task = Task(description="""
                          Your task is to validate and factcheck     
                          for the {career_name}field if {famous_personalities_name}
                          are correct or not
                          on the ground of 
                          1. The persons are/were working in the {career_name} field directly.
                          2. They have contributed significantly to that field and people consider them as great for their Knowledge,Inoovation, Dedication to that specific{career_name} field .
                          3. They are not involved in Criminal offences
                                        and add new names if necessary. """,
                        expected_output="Fact based correct feedback and suggestions and add new names if necessary",
                        agent=famous_personality_reviewer_agent)


famous_personality_agnet = Agent(role="Senior Content writer ",
                            goal='Give the best and correct content as per the feedback and review', 
                            backstory="""You are a Senior Content writer who rewrites the {famous_personalities_name} for the {career_name}field
                              based on the feedback and review  .
                              You should stick to the fact based content and add new names if necessary""",
                            llm=llm,
                            tools = [scrape_tool],
                            verbose=True)
                            
famous_personality_task = Task(description="""
                               A Content reviewer has reviewed the {famous_personalities_name} for the {career_name} field
                               to do  fact checking 
                               make any changes and add new names if necessary. 
                            """,
                          expected_output="""Return the updated list of Famous personality names only with what contribution they are famous for in bracket. Don't print any other texts""" ,
                          agent=famous_personality_agnet)

crew = Crew(agents=[ famous_personality_reviewer_agent,famous_personality_agnet], 
            tasks=[famous_personality_reviewer_task,famous_personality_task], 
            verbose=True)
            
inputs = {
    "career_name": "Automobile Engineer",
    "famous_personalities_name":"""1. Chetan Maini
                                   2. Pawan Goenka
                                 """
}
result = crew.kickoff(inputs=inputs)


# final output
print(result.raw)          