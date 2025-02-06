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

##  Defining all the Tools
scrape_tool = EXASearchTool()
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
# tavily_search_tool=Tavily_search_tool()



## Defining Agents

famous_personality_reviewer_agent = Agent(role=
                          "Senior Content Validator and fact checker",
                          goal="Validate the content with facts and get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field if {famous_personalities_name} You will give best fact based 
                          """,
                          llm=llm,
                          tools = [Google_search_tool],
                          verbose=True)


master_agent = Agent(role="Head Content Validator and approver",
                          goal=
                          """ 
                          Making sure that the labels given by subordinates are correct 
                          """, 
                          backstory=
                          """
                          You are the Head Content Validator and approver  who has been assigned to oversee the task 
                          of data labelling done by the other Content Validators.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)


## Defining Tasks for Agents

famous_personality_reviewer_task = Task(description=
                        """
                          Your task is to validate and fact-check .             
                          for the {career_name} field if {famous_personalities_name} are correct famous personality name or not.
                          on the ground of 
                          1. The persons have workied in the {career_name} field directly in the real work.
                          2. They have contributed significantly to that field and people consider them as great for their Knowledge,Innovation, Dedication to that specific {career_name} field .
                          3. They are not involved in Criminal offences.
                          4. They are not  passive contributors like investors,industrialists, activists. 
                        """,
                        expected_output=""" Give label "RIGHT" if all names satisfies all grounds , "WRONG" if even one ground is not satisfied """,
                        agent=famous_personality_reviewer_agent)


master_task = Task(description=
                            """
                               A Content reviewer  has labelled the data after  fact checking.
                               You need to make sure the following grounds
                               1. Even a single "WRONG" label is given to the data , return will be "REJECTED".
                               2. If all labels given are only "RIGHT" , return will be "ACCEPTED"
                            """,
                          expected_output=""" Return your deciscions in a single word only "ACCEPTED" or "REJECTED" """ ,
                          agent=master_agent)




# famous_personality_agnet = Agent(role="Senior Content writer ",
#                             goal='Give the best and correct content as per the feedback and review .remove irrelivant names  and add new names if necessary', 
#                             backstory="""You are a Senior Content writer who rewrites the {famous_personalities_name} for the {career_name}field
#                               based on the feedback and review  .
#                               You should stick to the fact based content and add new names if necessary""",
#                             llm=llm,
#                             tools = [Google_search_tool],
#                             verbose=True)
                            
# famous_personality_task = Task(description="""
#                                A Content reviewer has reviewed the {famous_personalities_name} for the {career_name} field
#                                to do  fact checking 
#                                remove irrelivant names  and add new names if necessary. 
#                             """,
#                           expected_output="""Return the updated list of Famous personality names only with what contribution they are famous for in bracket. Don't print any other texts""" ,
#                           agent=famous_personality_agnet)





crew = Crew(agents=[ famous_personality_reviewer_agent,master_agent], 
            tasks=[famous_personality_reviewer_task,master_task], 
            verbose=True)


inputs = {
    "career_name": "Automobile Engineer",
    "famous_personalities_name":"""
                                   1. Chetan Maini
                                   2. Pawan Goenka
                                   3. Abdul Kalam
                                   4. Avinash Kumar Agarwal
                                   5. Romesh Batra
                                """
}
result = crew.kickoff(inputs=inputs)


# final output
print(result.raw)          