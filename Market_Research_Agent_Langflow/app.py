import os
import dotenv
dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
LANGFLOW_TOKEN=os.getenv("LANGFLOW_TOKEN")
from langflow.load import run_flow_from_json
TWEAKS = {
  "Prompt-nse2S": {
    "template": "You are an experienced Market researcher write the below details based on company name entered by user {CompanyName}\nGive the details about the {CompanyName}\n1. About  the Company ( around 200 words)\n2.Operating Field \n3.Key Products and services \n4. Vision\n5. Product informations\n6. Market Position \n7. Competitors \n8. Recent News and Investments ",
    "CompanyName": ""
  },
  "Agent-JsZFq": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "Groq",
    "handle_parsing_errors": True,
    "input_value": "",
    "max_iterations": 30,
    "n_messages": 100,
    "order": "Ascending",
    "sender": "Machine and User",
    "sender_name": "",
    "session_id": "",
    "system_prompt": "You are a helpful assistant that can use tools to answer questions and perform tasks.",
    "template": "{sender_name}: {text}",
    "verbose": True,
    "groq_api_key": GROQ_API_KEY,
    "groq_api_base": "https://api.groq.com",
    "max_tokens": None,
    "temperature": 0.1,
    "n": None,
    "model_name": "llama-3.1-8b-instant"
  },
  "TavilyAISearch-Nwqeb": {
    "api_key": TAVILY_API_KEY,
    "include_answer": True,
    "include_images": True,
    "max_results": 5,
    "query": "",
    "search_depth": "advanced",
    "topic": "general"
  },
  "ChatInput-82kzj": {
    "files": "",
    "background_color": "",
    "chat_icon": "",
    "input_value": "Oppo",
    "sender": "User",
    "sender_name": "User",
    "session_id": "",
    "should_store_message": True,
    "text_color": ""
  },
  "Agent-V9h9I": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "Groq",
    "handle_parsing_errors": True,
    "input_value": "",
    "max_iterations": 15,
    "n_messages": 100,
    "order": "Ascending",
    "sender": "Machine and User",
    "sender_name": "",
    "session_id": "",
    "system_prompt": "",
    "template": "{sender_name}: {text}",
    "verbose": True,
    "groq_api_key": GROQ_API_KEY,
    "groq_api_base": "https://api.groq.com",
    "max_tokens": None,
    "temperature": 0.1,
    "n": None,
    "model_name": "llama-3.1-8b-instant"
  },
  "Prompt-nwU7W": {
    "template": "You are an AI and Data Science expert with an aim to find usecases and scopes where AI can be used to improve the company's performance , operational efficiency  and Customer satisfication based on {Companydescription}\nwrite the output in following format.\n1. AI Usecase number :\n2. Objective:\n3. Application:\n4. Benifit \n5. Github or kaggle links for similar usecase ",
    "Companydescription": ""
  },
  "ChatOutput-AchPO": {
    "background_color": "",
    "chat_icon": "",
    "data_template": "{text}",
    "input_value": "",
    "sender": "Machine",
    "sender_name": "AI",
    "session_id": "",
    "should_store_message": True,
    "text_color": ""
  },
  "CombineText-7d5YF": {
    "delimiter": " The usecases are ",
    "text1": "",
    "text2": ""
  }
}

result = run_flow_from_json(flow="Market_Research_Multiagent.json",
                            session_id="", 
                            fallback_to_env_vars=True, 
                            tweaks=TWEAKS)