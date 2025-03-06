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
## suggested models

# deepseek-r1-distill-llama-70b
# llama-3.3-70b-versatile
# llama-3.1-8b-instant

# llm = ChatGroq(model="groq/llama3-8b-8192",api_key=GROQ_API_KEY)
llm=LLM(model="groq/llama-3.3-70b-versatile",api_key=GROQ_API_KEY)  ## This model works best 

# llm=LLM(model="groq/llama-3.1-8b-instant",api_key=GROQ_API_KEY)
##  Defining all the Tools


@tool('DuckDuckGoSearch')
def duck_search(search_query: str):
    """Search the web for information on a given topic, The input parameter should be a string"""
    answer=DuckDuckGoSearchRun().run(search_query)
    print(f"The answer is {answer}")
    return answer
    
# duck_search=DuckDuckGoSearchRun()

# search = GoogleSerperAPIWrapper()
@tool("GoogleSearchTool")
def Google_search_tool(search_query: str):
    """Performs a search using the GoogleSearchTool."""
    return GoogleSerperAPIWrapper().run(search_query)
## Defining Agents

section_finder_agent = Agent(role=
                          "Senior Content Validator and fact corrector",
                          goal="Detect because of which sections the content was rejected ", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact corrector .
                          for the {career_name} field  the content are rejetced because of {status}.
                          Your job is to find the exact sections among the sections 
                          because of which the content was rejected and then pass that to the respective corrector agent.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True) 

description_corrector_agent = Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Correcter  who has been assigned to correct the content if necessary  
                          for the {career_name} field's {description}.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True) 
  
responsibilities_corrector_agent=Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Correcter  who has been assigned to correct the content if necessary  
                          for the {career_name} field's {responsibilities} and {work_context} .
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True) 


progression_path_corrector_agent=Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field's {progression_path}.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True) 

education_path_corrector_agent=Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field's {path1},{path2},{path3},{path4} .
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)                           

strength_weakness_corrector_agent=Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field's {strength} and {weakness} .
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)                           

top_colleges_corrector_agent = Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field's {top_colleges}.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)                       

                  
famous_personality_corrector_agent = Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field's {famous_personalities_name}.
                          The tools you have access accept strings as input parameter only.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          tools = [duck_search],                                      
                          verbose=True)


top_companies_corrector_agent = Agent(role=
                          "Senior Content Correcter ",
                          goal="Get the most correct content", 
                          backstory=
                          """
                          You are a Senior Content Validator and fact checker  who has been assigned to validate the task 
                          for the {career_name} field's {top_companies}.
                          The tools you have access accept strings as input parameter only.
                          """,
                          llm=llm,
                          tools = [duck_search],
                          # tools = [Google_search_tool],
                          verbose=True)


master_agent = Agent(role="Head Content Validator and approver",
                          goal=
                          """ 
                          Making sure that the correction made by subordinates are correct.
                          Only Making corrections in those sections which have errors as per the section_finder_agent.
                          Make sure you only use the agents which are required.
                          """, 
                          backstory=
                          """
                          You are the Head Content Validator and approver  who has been assigned to oversee the task 
                          of data correcting done by the other Content Validators.
                          """,
                          llm=llm,
                          # tools = [Google_search_tool],
                          verbose=True)


## Defining Tasks for Agents

section_finder_task = Task(description=
                        """
                          Your task is to  find the section and reason because of which the career data got rejetced. .             
                          for the {career_name} field , by analyzing {status} .
                          later we will pass that reason to the corresponding agent and ask to make acorrections.
                          1. The description is facutally correct.
                          2. The description matches with the field.
                        """,
                        expected_output=""" Section name or names and the mistakes  because of which the content was rejected
                        and then passing that to the corresponding agent.  
                        """,
                        agent=section_finder_agent)

description_corrector_task = Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if {description} are correct description.
                          on the ground of 
                          1. The description is facutally correct.
                          2. The description matches with the field.
                        """,
                        expected_output=""" Return corrected data 
                        """,
                        agent=description_corrector_agent)

responsibilities_corrector_task= Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if {responsibilities} and {work_context} .
                          on the ground of 
                          1. The responsibilities is facutally correct.
                          2. The responsibilities matches with the field.
                        """,
                        expected_output=""" Return corrected data """,
                        agent=responsibilities_corrector_agent)

strength_weakness_corrector_task= Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if {strength} and {weakness}.
                          on the ground of 
                          1. The responsibilities is facutally correct.
                          2. The responsibilities matches with the field.
                        """,
                        expected_output=""" Return corrected data """,
                        agent=strength_weakness_corrector_agent)



progression_path_corrector_task= Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if {progression_path}  are correct   or not.
                          on the ground of 
                          1. The responsibilities is facutally correct.
                          2. The responsibilities matches with the field.
                        """,
                        expected_output=""" Return corrected data """,
                        agent=progression_path_corrector_agent)

education_path_corrector_task= Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if  {path1},{path2},{path3},{path4} are correct education paths   or not.
                          on the ground of 
                          1. The responsibilities is facutally correct.
                          2. The responsibilities matches with the field.
                        """,
                        expected_output=""" Return corrected data """,
                        agent=progression_path_corrector_agent)                        

top_colleges_corrector_task = Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if {top_colleges}  are correct and suggested Colleges  or not.
                          on the ground of 
                          1. The suggested colleges are top colleges for that field.
                        """,
                        expected_output=""" Return corrected data """,
                        agent=top_colleges_corrector_agent)

famous_personality_corrector_task = Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if {famous_personalities_name}  are correct famous personality name or not.
                          on the ground of 
                          1. The persons have workied in the {career_name} field directly in the real work.
                          2. THey must predominantly known for this specific {career_name} field , not any other.
                          3. They have contributed significantly to that field and people consider them as great for their Knowledge,Innovation, Dedication to that specific {career_name} field .
                          4. They are not involved in Criminal offences.
                          5. They are not  passive contributors like investors,industrialists, activists. 
                          
                        """,
                        expected_output=""" Return corrected data """,
                        agent=famous_personality_corrector_agent)

top_companies_corrector_task = Task(description=
                        """
                          Your task is to correct the data  .             
                          for the {career_name} field if {top_companies}  are correct and suggested companies or not .
                          on the ground of 
                          1. The companies work directly in the field.
                          2. They have a reputation of working and employing people in that field.
                        """,
                        expected_output=""" Return corrected data """,
                        agent=top_companies_corrector_agent)

master_task = Task(description=
                            """
                               Subordinates have made corrections to some sections after the suggestions of section_finder_agent.
                               You need to make sure the following grounds
                               1. Only the sections which has issue are corrected.
                               2. All the corrections are correct and fact based.
                            """,
                          expected_output=""" 
                               Return the whole input after making corrections.
                          """,
                          agent=master_agent)


## Initializing the Crew

# crew = Crew(agents=[ description_corrector_agent,responsibilities_corrector_agent,progression_path_corrector_agent,education_path_corrector_agent,strength_weakness_corrector_agent,top_colleges_corrector_agent,famous_personality_corrector_agent,top_companies_corrector_agent,master_agent,section_finder_agent], 
#             tasks=[description_corrector_task,progression_path_corrector_task,education_path_corrector_task,strength_weakness_corrector_task,responsibilities_corrector_task,top_colleges_corrector_task,famous_personality_corrector_task,top_companies_corrector_task,master_task,section_finder_task], 
#             verbose=True)

crew = Crew(
    agents=[
        section_finder_agent, master_agent, top_companies_corrector_agent, famous_personality_corrector_agent,
        top_colleges_corrector_agent, strength_weakness_corrector_agent, education_path_corrector_agent,
        progression_path_corrector_agent, responsibilities_corrector_agent, description_corrector_agent
    ],
    tasks=[
        section_finder_task, master_task, top_companies_corrector_task, famous_personality_corrector_task,
        top_colleges_corrector_task, responsibilities_corrector_task, strength_weakness_corrector_task,
        education_path_corrector_task, progression_path_corrector_task, description_corrector_task
    ],
    verbose=True
)




inputs={
  "career_id": "9",
  "career_name": "Aerospace Engineer",
  "description": "Aerospace engineers play an important role in the aviation and space industries, designing, producing, and testing aeroplanes, spacecraft, and missiles. They operate in an array of industries, including defence, space research, and commercial aircraft, making aeronautical engineering a vibrant and important discipline.\n\nAerospace engineering spans various specialized fields, including a thorough grasp of aerodynamics, propulsion systems, and material science. Aerodynamics is an understanding of how air reacts with solid matter, which is essential in creating efficient and safe aeroplanes and spacecraft. Propulsion systems are the engines and motors that generate the necessary thrust for these vehicles, and materials science is concerned with selecting and applying materials that can resist harsh circumstances.\n\nAerospace engineering encompasses far more than only the design and creation of new aircraft and spacecraft. Aerospace engineers also work to maintain and enhance current technology, assuring their safety and efficiency. They may work on initiatives such as creating new aviation technology, increasing fuel efficiency, or enhancing aircraft durability.\n\nAn aeronautical engineering career is both exciting and financially lucrative. Aerospace engineering salaries are competitive, reflecting the field's high degree of competence and responsibility. According to numerous industry statistics, aerospace engineers may expect to make a good living, with prospects for advancement as they gain expertise and work on increasingly difficult projects.\n\nSo, is aerospace engineering a good career? The answer is a bold yes. Aerospace engineering provides plenty of opportunities for people interested in innovation, problem-solving, and working with cutting-edge technologies. The demand for experienced aerospace engineers remains high, and their work is crucial to advances in aviation and space exploration.\n\nIn a nutshell, aerospace engineers are critical to the advancement of aviation and space technology. Their knowledge of aerodynamics, propulsion systems, and materials science is critical for developing and enhancing aeroplanes and spacecraft. With a potential aerospace engineering salary and a diverse range of tasks, choosing a career in aerospace engineering is both gratifying and influential.",
  "top_colleges": "Indian Institute of Technology (IIT)  Bombay,, Indian Institute of Technology (IIT), Madras, Indian Institute of Technology (IIT), Kanpur, Indian Institute of Science (IISc), Bangalore, National Institute of Technology (NIT), Trichy",
  "famous_personalities_name": "Kalpana Chawla, Dr. Mylswamy Annadurai, G. Madhavan Nair, K. Radhakrishnan, Shreedhar Vembu",
  "top_companies": "Hindustan Aeronautics Limited (HAL), Indian Space Research Organisation (ISRO), Bharat Electronics Limited (BEL), Tata Advanced Systems, Mahindra Aerospace",
  "path1": "12th - Science (Maths) -> ITI/Diploma - Diploma in Aeronautical Engineering (Specialization: Aerodynamics, Avionics, Propulsion) -> UG - B.Tech in Aerospace Engineering (Specialization: Aerodynamics, Avionics, Propulsion) -> PG - M.Tech in Aerospace Engineering (Specialization: Aerodynamics, Avionics, Propulsion) -> Certification - Certified Aerospace Technician (CAT), FAA Airframe and Powerplant (A&P), Certified Aerodynamics Engineer (CAE)",
  "path2": "12th - Science (Maths) -> UG - B.E. in CSE Engineering (Specialization: Space Engineering, Aircraft Design, Flight Mechanics) -> PG - M.E. in Aerospace Engineering (Specialization: Space Engineering, Aircraft Design, Flight Mechanics) -> Certification - Certified Flight Instructor (CFI), Project Management Professional (PMP), Six Sigma Black Belt",
  "path3": "12th - Science (Maths) -> UG - B.Sc in Aerospace Engineering (Specialization: Rocket Propulsion, Satellite Technology, Aircraft Maintenance) -> PG - M.Sc in Aerospace Engineering (Specialization: Rocket Propulsion, Satellite Technology, Aircraft Maintenance) -> Certification - Certified Space Technician (CST), Certified Systems Engineering Professional (CSEP), ISO 9001 Quality Management Certification",
  "path4": "ITI/Diploma - ITI in Aeronautical Engineering -> 12th - Science (Maths) -> UG - B.Tech in Aerospace Engineering (Specialization: Aerodynamics, Avionics, Propulsion) -> PG - M.Tech in Aerospace Engineering (Specialization: Aerodynamics, Avionics, Propulsion) -> Certification - Certified Aerospace Technician (CAT), FAA Airframe and Powerplant (A&P), Certified Aerodynamics Engineer (CAE)",
  "progression_path": "Junior Aerospace Engineer, Aerospace Engineer, Senior Aerospace Engineer, Aerospace Engineering Manager, Director of Engineering, Chief Technology Officer (CTO)",
  "strength": "High demand for skills, Involvement in cutting-edge technology, High earning potential, Opportunities for innovation, Significant career growth",
  "weakness": "High pressure, Need for continuous learning, Potential for long hours, Risk of job burnout, Intense competition",
  "responsibilities": "Develop designs for aircraft, spacecraft, and missile systems based on requirements and specifications., Conduct aerodynamic and structural analyses to predict system performance., Carry out ground and flight tests to validate design performance and safety., Investigate new technologies and materials to enhance system performance and efficiency., Oversee aerospace projects, ensuring they are completed on time and within budget., Ensure all designs and processes comply with industry regulations and safety standards., Document the design and development process, including specifications, manufacturing instructions, and safety protocols.",
  "work_context": "Office setting, laboratories, and on-site testing facilities., Standard office hours, with occasional overtime or weekends depending on project deadlines., Regular interaction with team members, project managers, and clients., Occasional travel for on-site testing, client meetings, and conferences., Limited remote work opportunities, mostly project-based., High demand in sectors like defense, space exploration, and commercial aviation., Continuous learning to keep up with evolving technologies and industry trends.",
  "status": "Status: REJECTED\nReason: The field \"Education Paths\" is labelled as \"WRONG\" by a subordinate for not satisfying the given criteria, specifically the second and third paths being notably unrelated to the field of Agriculture Extension Executive. Additionally, other sections such as \"Suggested Colleges\" and \"Organizations\" are also labelled as \"WRONG\" due to incomplete verification of the given content."

}



result = crew.kickoff(inputs=inputs)


# # final output
print(result.raw)  