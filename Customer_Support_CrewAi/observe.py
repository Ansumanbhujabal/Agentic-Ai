import warnings
warnings.filterwarnings("ignore")
from crewai import Agent, Task, Crew,LLM
from crewai_tools import ScrapeWebsiteTool
from IPython.display import Markdown
from langchain_groq import ChatGroq
import getpass
import os
import dotenv
dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from agentneo import AgentNeo, Tracer
from agentneo import launch_dashboard
# Initialize
neo_session = AgentNeo(session_name="CustomerSupport_CrewAi")
neo_session.create_project(project_name="CustomerSupport_CrewAi_2")

# Start tracing
tracer = Tracer(session=neo_session)
tracer.start()

## Initialize Llama
# llm = ChatGroq(model="llama3-8b-8192",api_key=GROQ_API_KEY)
llm=LLM(model="groq/llama3-8b-8192",api_key=GROQ_API_KEY)

##Agents 

#Support Agent 
support_agent = Agent(
    llm=llm,## By Deafult is set to OpenAI
    role="Senior Support Representative",
	goal="Be the most friendly and helpful "
        "support representative in your team",
	backstory=(
		"You work at Invest4Edu (https://www.invest4edu.com/) and "
        " are now working on providing "
		"support to {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions."
	),
	allow_delegation=False,
	verbose=True
)
# Support Quality Assurance Agent 
support_quality_assurance_agent = Agent(
    llm=llm,
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the "
    "best support quality assurance in your team",
	backstory=(
		"You work at Invets4Edu (https://www.invest4edu.com/) and "
        "are now working with your team "
		"on a request from {customer} ensuring that "
        "the support representative is "
		"providing the best support possible.\n"
		"You need to make sure that the support representative "
        "is providing full"
		"complete answers, and make no assumptions."
	),
	verbose=True
)

## Tools Initialization

# search_tool = SerperDevTool()
# scrape_tool = ScrapeWebsiteTool()
docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://www.invest4edu.com/blog/pte-test-pattern-fee-structure"
)


## Tasks

## Inquiry resolution task with tools on task level 

inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
	    "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
		"Make sure to use everything you know "
        "to provide the best support possible."
		"You must strive to provide a complete "
        "and accurate response to the customer's inquiry."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)

## QA task

quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company "
	    "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)



## Crew Making
crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution, quality_assurance_review],
#   embedder={
#             "provider": "huggingface",
#             "config": {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
#         },
  verbose=True,
  memory=False ## If we keep true it asks OpenAI key 
)


## Input demo

inputs = {
    "customer": "PMEC,Berhampur",
    "person": "Ansuman Bhujabala",
    "inquiry": "I need help with making my students aware"
               " about study abroad opportunities , specifically "
               " how and which eaxm they should appear and which of your service they can use  "
               " Can you provide guidance?"
}
# result = crew.kickoff(inputs=inputs)



@tracer.trace_agent("customer_agent")
async def run_agent(input):
    result = crew.kickoff(inputs=input)
    return result

# Usage
text = run_agent(inputs)


# After your traces are complete
tracer.stop()
launch_dashboard(port=6000)

## Print the result in MarkDown

Markdown(text)
