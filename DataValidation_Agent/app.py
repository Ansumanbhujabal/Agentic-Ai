import warnings
warnings.filterwarnings("ignore")
from crewai import Agent, Task, Crew,LLM
from crewai_tools import SerperDevTool,ScrapeWebsiteTool,WebsiteSearchTool,EXASearchTool
from crewai.tools import tool
from langchain_community.tools import TavilySearchResults,JinaSearch,DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from IPython.display import Markdown
from langchain_groq import ChatGroq
import getpass
import os
import dotenv
dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY=os.getenv("SERPER_API_KEY")

print("The api key is-------------------------->>>>>>>>>>>>>>>>>>")
print(GROQ_API_KEY)
## suggested models

# deepseek-r1-distill-llama-70b
# llama-3.3-70b-versatile
# llama-3.1-8b-instant

# llm = ChatGroq(model="groq/llama3-8b-8192",api_key=GROQ_API_KEY)
llm=LLM(model="groq/llama-3.3-70b-versatile",api_key=GROQ_API_KEY)  ## This model works best 

##  Defining all the Tools


@tool('DuckDuckGoSearch')
def duck_search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)


scrape_tool = EXASearchTool()
# search = GoogleSerperAPIWrapper()
@tool("GoogleSearchTool")
def Google_search_tool(search_query: str):
    """Performs a search using the GoogleSearchTool."""
    return GoogleSerperAPIWrapper().run(search_query)
web_search_tool = TavilySearchResults(k=3)    
@tool("Tavilysearchtool")
def Tavily_search_tool(search_query: str):
    """Performs a search using the Tavily search tool."""
    return web_search_tool().run(search_query)
# tavily_search_tool=Tavily_search_tool()



## Defining Agents

description_reviewer_agent = Agent(role=
                          "Senior Content Validator and fact checker",
                          goal="Validate the content with facts and get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field if {description} are correct or not.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)   

top_colleges_reviewer_agent = Agent(role=
                          "Senior Content Validator and fact checker",
                          goal="Validate the content with facts and get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field if {top_colleges}   are correct or not.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)                       

top_exams_reviewer_agent = Agent(role=
                          "Senior Content Validator and fact checker",
                          goal="Validate the content with facts and get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field if {top_exams}  are correct or not.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)                       


famous_personality_reviewer_agent = Agent(role=
                          "Senior Content Validator and fact checker",
                          goal="Validate the content with facts and get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field if {famous_personalities_name}   are correct or not.
                          The tools you have access accept strings as input parameter only.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          tools = [duck_search],

                          verbose=True)


top_companies_reviewer_agent = Agent(role=
                          "Senior Content Validator and fact checker",
                          goal="Validate the content with facts and get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field if {top_companies}   are correct companies or not.
                          The tools you have access accept strings as input parameter only.
                          """,
                          llm=llm,
                          tools = [duck_search],
                          # tools = [Google_search_tool],
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

description_reviewer_task = Task(description=
                        """
                          Your task is to validate and fact-check .             
                          for the {career_name} field if {description} are correct description  or not.
                          on the ground of 
                          1. The description is facutally correct.
                          2. The description matches with the field.
                        """,
                        expected_output=""" Give label "RIGHT" if all names satisfies all grounds , "WRONG" if even one ground is not satisfied 
                                            One liner reason only for the  label "WRONG" """,
                        agent=description_reviewer_agent)

top_colleges_reviewer_task = Task(description=
                        """
                          Your task is to validate and fact-check .             
                          for the {career_name} field if {top_colleges}  are correct and suggested Colleges  or not.
                          on the ground of 
                          1. The suggested colleges are top colleges for that field.
                        """,
                        expected_output=""" Give label "RIGHT" if all names satisfies all grounds , "WRONG" if even one ground is not satisfied
                                            One liner reason only for the  label "WRONG" """,
                        agent=top_colleges_reviewer_agent)


top_exams_reviewer_task = Task(description=
                        """
                          Your task is to validate and fact-check .             
                          for the {career_name} field if {top_exams}  are correct and suggested exams  or not to study in that field.
                          on the ground of 
                          1. The exams are not outdated and still conducted.
                          2. The exams are famous and a significant number of students appear it in India.
                        """,
                        expected_output=""" Give label "RIGHT" if all names satisfies all grounds , "WRONG" if even one ground is not satisfied 
                                            One liner reason only for the  label "WRONG" """,
                        agent=top_exams_reviewer_agent)



famous_personality_reviewer_task = Task(description=
                        """
                          Your task is to validate and fact-check .             
                          for the {career_name} field if {famous_personalities_name}  are correct famous personality name or not.
                          on the ground of 
                          1. The persons have workied in the {career_name} field directly in the real work.
                          2. THey must predominantly known for this specific {career_name} field , not nay other.
                          3. They have contributed significantly to that field and people consider them as great for their Knowledge,Innovation, Dedication to that specific {career_name} field .
                          4. They are not involved in Criminal offences.
                          5. They are not  passive contributors like investors,industrialists, activists. 
                        """,
                        expected_output=""" Give label "RIGHT" if all names satisfies all grounds , "WRONG" if even one ground is not satisfied 
                                            One liner reason only for the  label "WRONG" """,
                        agent=famous_personality_reviewer_agent)


top_companies_reviewer_task = Task(description=
                        """
                          Your task is to validate and fact-check .             
                          for the {career_name} field if {top_companies}  are correct and suggested companies or not .
                          on the ground of 
                          1. The companies work directly in the field.
                          2. They have a reputation of working and employing people in that field.
                        """,
                        expected_output=""" Give label "RIGHT" if all names satisfies all grounds , "WRONG" if even one ground is not satisfied 
                                            One liner reason only for the  label "WRONG" """,
                        agent=top_companies_reviewer_agent)


master_task = Task(description=
                            """
                               A Content reviewer  has labelled the data after  fact checking.
                               You need to make sure the following grounds
                               1. Even a single "WRONG" label is given to the data from any subordinate , return will be "REJECTED".
                               2. If all labels given are only "RIGHT" , return will be "ACCEPTED"
                               3. If you are returning "REJECTED" then also add it's reason from the agent.
                            """,
                          expected_output=""" 
                          Status: Return your deciscions in a single word only "ACCEPTED" or "REJECTED"
                          Reason: If the status is "REJECTED", state  the reason  """ ,
                          agent=master_agent)


## Initializing the Crew

crew = Crew(agents=[ description_reviewer_agent,top_colleges_reviewer_agent,famous_personality_reviewer_agent,top_companies_reviewer_agent,master_agent], 
            tasks=[description_reviewer_task,top_colleges_reviewer_task,famous_personality_reviewer_task,top_companies_reviewer_task,master_task], 
           verbose=True)


# inputs={
#   "career_id": "116",
#   "career_name": "Architect",
#   "description": "An Architect is responsible for designing, planning, and overseeing the construction of buildings, homes, and other structures. They work in various sectors, including residential, commercial, and public works. Architects must have a strong understanding of design principles, construction methods, and zoning regulations. They also need to consider aesthetic appeal, functionality, and sustainability when creating designs.",
#   "top_colleges": "School of Planning and Architecture  Delhi,, Indian Institute of Technology (IIT), Kharagpur, National Institute of Technology (NIT), Tiruchirappalli, Jadavpur University, Kolkata, Indian Institute of Technology (IIT), Roorkee",
#   "famous_personalities_name": "B.V. Doshi, Charles Correa, Raj Rewal, Hafeez Contractor",
#   "top_companies": "Hafeez Contractor, Morphogenesis, C.P. Kukreja Associates, Shashi Prabhu & Associates, Raja Aederi Consultants"
# }
# result = crew.kickoff(inputs=inputs)


# # final output
# # print(result.raw)          
# final_result={
# "id":inputs["career_id"],
# "status":result.raw      
# }
# print(f"The result is {final_result}")