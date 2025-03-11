from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat, config_list_from_json
from autogen import register_function
from langchain_community.tools import DuckDuckGoSearchRun
import os
import json

# Configuration
# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
llm_config ={
    "cache_seed": 48,
    "config_list": [{
        "model": os.environ.get("OPENAI_MODEL_NAME", "llama-3.3-70b-versatile"),
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1")}
    ],
}

# Register tools
duck_search = DuckDuckGoSearchRun()
def duck_search_tool(search_query: str):
    return duck_search.run(search_query)

# def register_search_tools(agent):
#     register_function(
#         duck_search_tool,
#         caller=agent,
#         name="DuckDuckGoSearch",
#         description="Search the web for current information"
#     )

def register_search_tools(agent, executor):
    register_function(
        duck_search_tool,
        caller=agent,
        executor=executor,  # Correctly specify executor
        name="DuckDuckGoSearch",
        description="Search the web for current information"
    )

section_finder_agent = AssistantAgent(
    name="SectionFinder",
    system_message="""
    You are a Senior Content Validator. Analyze the rejection status and identify which sections need correction.
    Your final answer should clearly state the problematic sections and the reasons.
    """,
    llm_config=llm_config
)

description_corrector_agent = AssistantAgent(
    name="DescCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if {description} are correct description.
    on the ground of 
    1. The description is facutally correct.
    2. The description matches with the field.
    You correct the description section. Maintain original structure and Indian context.
    Only respond when description needs correction.
    """,
    llm_config=llm_config
)

responsibilities_corrector_agent = AssistantAgent(
    name="RespCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if {responsibilities} and {work_context} .
    on the ground of 
    1. The responsibilities is facutally correct.
    2. The responsibilities matches with the field.    
    Specialize in correcting responsibilities and work context sections.
    Only respond when these sections need fixes.
    """,
    llm_config=llm_config
)

progression_path_corrector_agent = AssistantAgent(
    name="ProgPathCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if {progression_path}  are correct   or not.
    on the ground of 
    1. The responsibilities is facutally correct.
    2. The responsibilities matches with the field.    
    Expert in validating and correcting career progression paths.
    Only engage when progression path needs correction.
    """,
    llm_config=llm_config
)

education_path_corrector_agent = AssistantAgent(
    name="EduPathCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if  {path1},{path2},{path3},{path4} are correct education paths   or not.
    on the ground of 
    1. The responsibilities is facutally correct.
    2. The responsibilities matches with the field.    
    Specialize in education path corrections. Verify Indian context and relevance.
    """,
    llm_config=llm_config
)

strength_weakness_corrector_agent = AssistantAgent(
    name="SWCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if {strength} and {weakness}.
    on the ground of 
    1. The responsibilities is facutally correct.
    2. The responsibilities matches with the field.
    Focus on validating and correcting strengths/weaknesses sections.
    Ensure factual accuracy and relevance.
    """,
    llm_config=llm_config
)

top_colleges_corrector_agent = AssistantAgent(
    name="CollegeCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if {top_colleges}  are correct and suggested Colleges  or not.
    on the ground of 
    1. The suggested colleges are top colleges for that field.    
    Validate and correct top colleges list. Use latest Indian rankings.
    """,
    llm_config=llm_config,
    function_map={"DuckDuckGoSearch": duck_search_tool}
)

famous_personality_corrector_agent = AssistantAgent(
    name="PersonalityCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if {famous_personalities_name}  are correct famous personality name or not.
    on the ground of 
    1. The persons have workied in the {career_name} field directly in the real work.
    2. THey must predominantly known for this specific {career_name} field , not any other.
    3. They have contributed significantly to that field and people consider them as great for their Knowledge,Innovation, Dedication to that specific {career_name} field .
    4. They are not involved in Criminal offences.
    5. They are not  passive contributors like investors,industrialists, activists. 
        
    Verify and correct famous personalities. Ensure direct field relevance.
    """,
    llm_config=llm_config,
    function_map={"DuckDuckGoSearch": duck_search_tool}
)

top_companies_corrector_agent = AssistantAgent(
    name="CompanyCorrector",
    system_message="""
    Your task is to correct the data  .             
    for the {career_name} field if {top_companies}  are correct and suggested companies or not .
    on the ground of 
    1. The companies work directly in the field.
    2. They have a reputation of working and employing people in that field.    
    Validate and correct top companies list. Check Indian context.
    """,
    llm_config=llm_config,
    function_map={"DuckDuckGoSearch": duck_search_tool}
)

# register_search_tools(top_colleges_corrector_agent)
# register_search_tools(famous_personality_corrector_agent)
# register_search_tools(top_companies_corrector_agent)



# master_agent = AssistantAgent(
#     name="MasterValidator",
#     system_message="""
#     Final authority on content approval. Ensure only necessary corrections are made.
#     Compile final output in specified JSON format.
#     """,
#     llm_config=llm_config
# )

# Update the master agent configuration
master_agent = AssistantAgent(
    name="MasterValidator",
    system_message="""
    Final authority on content approval. Rules:
    1. Verify only necessary corrections are made
    2. Maintain original JSON structure
    3. Required output format:
    {
      "corrected_sections": {
        "section1": "corrected content",
        "section2": ["item1", "item2"]
      }
    }
    4. Ensure Indian context in all fields
    5. Include ONLY sections mentioned in status: {status}
    """,
    llm_config=llm_config
)

# Create group chat
agents = [
    section_finder_agent,
    description_corrector_agent,
    responsibilities_corrector_agent,
    progression_path_corrector_agent,
    education_path_corrector_agent,
    strength_weakness_corrector_agent,
    top_colleges_corrector_agent,
    famous_personality_corrector_agent,
    top_companies_corrector_agent,
    master_agent
]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=20,
    speaker_selection_method="round_robin"
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# Define input data

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
  # "status": "Status: REJECTED\nReason: The field \"Education Paths\" is labelled as \"WRONG\" by a subordinate for not satisfying the given criteria, specifically the second and third paths being notably unrelated to the field of Agriculture Extension Executive. Additionally, other sections such as \"Suggested Colleges\" and \"Organizations\" are also labelled as \"WRONG\" due to incomplete verification of the given content."
  "status": "Status: REJECTED\nReason: The field \"Education Paths\" is labelled as \"WRONG\" by a subordinate for not satisfying the given criteria, specifically the second and third paths being notably unrelated to the field"

}


# Initiate chat
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="Continue",
    max_consecutive_auto_reply=10
)

def print_messages(recipient, messages, sender, config):
    print(f"Messages from {sender.name} to {recipient.name}: {messages[-1]['content']}")
    return False, None

for agent in agents:
    agent.register_reply(
        [AssistantAgent, UserProxyAgent],
        reply_func=print_messages, 
        config={"callback": None},
    )

# user_proxy.initiate_chat(
#     manager,
#     message=f"""
#     Analyze and correct this career data:
#     {json.dumps(inputs, indent=2)}
    
#     Required output format:
#     {{
#       "corrected_sections": {{
#         "section_name": "corrected_content",
#         ...
#       }}
#     }}
#     Only include corrected sections.
#     """
# )
# Add this before initiating chat
processed_input = json.dumps(inputs, indent=2)
status_reason = inputs["status"].split("Reason:")[-1].strip()
user_proxy.initiate_chat(
    manager,
    message=f"""
    CAREER DATA VALIDATION TASK:
    {processed_input}
    
    SPECIAL INSTRUCTIONS:
    1. Focus on these rejection reasons: {status_reason}
    2. Follow validation rules strictly
    3. Use Indian context only
    4. Maintain original structure
    
    REQUIRED OUTPUT FORMAT:
    {{
      "corrected_sections": {{
        "section_name": "corrected_content",
        ...
      }}
    }}
    """
)