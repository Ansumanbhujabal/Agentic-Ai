import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import os
from logging.handlers import RotatingFileHandler

from pydantic import BaseModel, Field,schema_json_of
from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.tools.duckduckgo import DuckDuckGoTools


# Setup directories and logger (same as previous)
Path("logs").mkdir(exist_ok=True)
Path("input").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

logger = logging.getLogger("exam_info")

# Define the complete Pydantic model structure based on the sample
class ExamCutoffs(BaseModel):
    general: int
    obc: int
    sc_st: int

class ExamEligibilityEducation(BaseModel):
    required_subjects: List[str]
    minimum_marks: Dict[str, int]

class ExamEligibility(BaseModel):
    age_limit: Dict[str, Optional[int]]
    education: ExamEligibilityEducation
    nationality: List[str]
    attempts: str

class ExamPatternDistribution(BaseModel):
    questions: int
    marks: int

class ExamPattern(BaseModel):
    total_marks: int
    total_questions: int
    question_distribution: Dict[str, Dict[str, ExamPatternDistribution]]
    marking_scheme: Dict[str, int]
    type: str
    duration: str

class ExamImportantDates(BaseModel):
    application_start: str
    application_end: str
    admit_card_release: str
    exam_date: str
    result_date: str
    counseling_start: str

class ExamInfo(BaseModel):
    id: str = Field(..., description="Official exam ID/shortcode")
    name: str = Field(..., description="Full official exam name")
    short_description: str = Field(..., description="Brief overview of the exam")
    tags: List[str] = Field(..., description="Relevant categorization tags")
    target_audience: str = Field(..., description="Intended participant demographic")
    
    highlights: Dict[str, str | List[str]] = Field(
        ...,
        description="Key features including mode, frequency, conducting body, etc."
    )
    
    important_dates: ExamImportantDates = Field(
        ...,
        description="All critical dates for 2025 cycle in YYYY-MM-DD format"
    )
    
    eligibility: ExamEligibility = Field(
        ...,
        description="Detailed eligibility criteria including age, education, etc."
    )
    
    exam_pattern: ExamPattern = Field(
        ...,
        description="Detailed exam structure and marking scheme"
    )
    
    syllabus: Dict[str, List[str] | str] = Field(
        ...,
        description="Detailed syllabus breakdown and PDF link, ensure the url is correct and verified"
    )
    
    preparation_resources: Dict[str, Dict[str, List[str]]] = Field(
        ...,
        description="Recommended study materials and practice resources, please provide genuine offical urls "
    )
    
    application_process: Dict[str, str | List | Dict] = Field(
        ...,
        description="Step-by-step application procedure and requirements"
    )
    
    participating_colleges: Dict[str, str | int | List] = Field(
        ...,
        description="List of participating institutions and seat details"
    )
    
    cutoffs_and_results: Dict[str, ExamCutoffs | str] = Field(
        ...,
        description="Previous year cutoffs and result checking process"
    )
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.tools.newspaper4k import Newspaper4kTools

# agent = Agent(
#     model=Groq(id="deepseek-r1-distill-llama-70b"),
#     tools=[DuckDuckGoTools(), Newspaper4kTools()],
#     description="Expert system for retrieving accurate exam information.",
#     # markdown=True,
#     show_tool_calls=True,
#     add_datetime_to_instructions=True,
# )

# Initialize agent (same as previous)
agent = Agent(
    model=AzureOpenAI(
        id="gpt-4",  # or gpt-4o if you have access
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        # api_version="2023-03-15-preview",
        api_version="2024-08-01-preview",  # Updated API version
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ),
    tools=[DuckDuckGoTools()],
    description="Expert system for retrieving accurate exam information.",
    # response_model=ExamInfo,
)

def process_exam(exam_name: str) -> Optional[ExamInfo]:
    """Process exam with full agent integration"""
    logger.info(f"Processing: {exam_name}")
    
    try:
        start_time = datetime.now()
        few_shot_prompt="""
             {
  "exam": {
    "id": "neet_ug",
    "name": "NEET UG",
    "short_description": "National level medical entrance exam for undergraduate courses like MBBS, BDS, and AYUSH.",
    "tags": ["medical", "undergraduate", "entrance", "national"],
    "target_audience": "Students who have completed or are appearing for 12th grade with PCB subjects.",
    "highlights": {
      "mode": "Offline (Pen & Paper)",
      "frequency": "Once a year",
      "conducting_body": "National Testing Agency (NTA)",
      "duration": "3 hours 20 minutes",
      "medium": ["English", "Hindi", "Tamil", "Telugu", "Urdu", "Bengali", "Kannada", "Gujarati", "Marathi", "Assamese", "Odia", "Punjabi"],
      "official_website": "https://neet.nta.nic.in/"
    },
    "important_dates": {
      "application_start": "2025-02-01",
      "application_end": "2025-03-01",
      "admit_card_release": "2025-04-15",
      "exam_date": "2025-05-05",
      "result_date": "2025-06-15",
      "counseling_start": "2025-07-01"
    },
    "eligibility": {
      "age_limit": {
        "minimum": 17,
        "maximum": null,
        "as_of_date": "2025-12-31"
      },
      "education": {
        "required_subjects": ["Physics", "Chemistry", "Biology"],
        "minimum_marks": {
          "general": 50,
          "obc": 40,
          "sc_st": 40
        }
      },
      "nationality": ["Indian", "NRIs", "OCIs", "PIOs", "Foreign Nationals"],
      "attempts": "No limit"
    },
    "exam_pattern": {
      "total_marks": 720,
      "total_questions": 200,
      "question_distribution": {
        "Physics": {"questions": 50, "marks": 180},
        "Chemistry": {"questions": 50, "marks": 180},
        "Biology": {
          "Botany": {"questions": 50, "marks": 180},
          "Zoology": {"questions": 50, "marks": 180}
        }
      },
      "marking_scheme": {
        "correct_answer": 4,
        "incorrect_answer": -1,
        "unattempted": 0
      },
      "type": "Multiple Choice Questions (MCQs)",
      "duration": "3 hours 20 minutes"
    },
    "syllabus": {
      "Physics": [
        "Class 11: Physical world, Motion, Thermodynamics",
        "Class 12: Electrostatics, Optics, etc."
      ],
      "Chemistry": [
        "Class 11: Basic concepts, States of Matter",
        "Class 12: Solid State, Electrochemistry, etc."
      ],
      "Biology": [
        "Class 11: Diversity of living world",
        "Class 12: Reproduction, Genetics, etc."
      ],
      "syllabus_pdf": "https://neet.nta.nic.in/document/syllabus-for-neet-ug-2025-examination-reg/"
    },
    "preparation_resources": {
      "books": {
        "Physics": ["Concepts of Physics by H.C. Verma", "NCERT Textbooks"],
        "Chemistry": ["Modern ABC", "NCERT Textbooks"],
        "Biology": ["Trueman's Biology", "NCERT Textbooks"]
      },
      "mock_tests": ["https://nta.ac.in/Student"],
      "previous_papers": [
        "https://neet.nta.nic.in/document-category/question-paper-2020/",
        "https://neet.nta.nic.in/document/english-set-g1-neet-qp-2020/"
      ],
      "video_courses": ["https://nta.ac.in/LecturesContent"]
    },
    "application_process": {
      "steps": [
        "Visit the official website",
        "Register using email and mobile number",
        "Fill in personal and academic details",
        "Upload documents",
        "Pay the application fee",
        "Download confirmation page"
      ],
      "documents_required": ["Passport size photo", "Signature", "Class 10 Certificate", "ID Proof"],
      "fees": {
        "general": 1700,
        "obc": 1600,
        "sc_st_pwd": 1000
      },
      "correction_window": {
        "available": true,
        "dates": {
          "start": "2025-03-05",
          "end": "2025-03-10"
        },
        "link": "https://neet.nta.nic.in/neetug-2025-correction-window/"
      }
    },
    "scholarships_and_fees": {
      "application_fee_waivers": ["EWS candidates", "Certain state boards"],
      "related_scholarships": ["Inspire Scholarship", "NTSE-based benefits"]
    },
    "exam_centers": {
      "total_centers": 543,
      "cities": ["Delhi", "Mumbai", "Chennai", "Bangalore", "Hyderabad", "..."],
      "center_allocation_rules": "Based on choices filled during registration and availability"
    },
    "cutoffs_and_results": {
      "previous_year_cutoffs": {
        "general": 137,
        "obc": 107,
        "sc_st": 107
      },
      "how_to_check": "Visit official website > Login > View Scorecard",
      "score_normalization": "Percentile based"
    },
    "participating_colleges": {
      "total_colleges": 542,
      "total_seats": 101388,
      "top_colleges": [
        {"name": "AIIMS Delhi", "seats": 107},
        {"name": "JIPMER Puducherry", "seats": 150}
      ],
      "counseling_body": "Medical Counseling Committee (MCC)"
    },
    "student_reviews_and_faqs": {
      "reviews": [
        {
          "student": "Riya Sharma",
          "year": 2024,
          "review": "Consistent NCERT-based preparation was the key. Mock tests helped improve my speed."
        }
      ],
      "faqs": [
        {
          "question": "Can I apply for NEET without biology in 12th?",
          "answer": "No, Biology is a mandatory subject to appear for NEET."
        },
        {
          "question": "Is there any limit on the number of attempts?",
          "answer": "No, there's no cap on attempts as long as you meet the age criteria."
        }
      ]
    }
"""
        # Final_prompt= f"Provide verified 2025 details for {exam_name}  refer the below example for output format only  {few_shot_prompt} . Return the output without any extra messages"
        Final_prompt = (
            f"Provide verified 2025 details for {exam_name} in strict JSON format matching this schema: \n"
            f"{ExamInfo.schema_json(indent=2)}\n"
            "Include official dates, URLs, and eligibility criteria. using the tools  "
            "Return ONLY the JSON object without any additional text."
            f"Here is a sample to refer {few_shot_prompt}"
            "Don't return any extra messages "
            "Make sure all urls are pointing to correct live pages and legit "
        )
        
        agent_result = agent.run(
         Final_prompt 
        # response_model=ExamInfo
         )
        print("==========================================================")
        print(" The agent result is ")
        print(agent_result)
        print("==========================================================")
        if not agent_result or not agent_result.content:
            logger.error(f"No valid content received for {exam_name}")
            return None

        exam_data = agent_result.content
        exam_data = agent_result.content.strip().replace('```json', '').replace('```', '').replace('\n    ','')
        exam_data = json.loads(exam_data) 
        print("==========================================================")
        print(" The exam_data  is ")
        print(exam_data)
        print("==========================================================")        
        processing_time = datetime.now() - start_time

        # Log agent execution details
        logger.debug(
            f"Execution Metrics:\n"
            f"Tokens Used: {getattr(agent_result, 'metrics', 'N/A')}\n"
            f"Model: {getattr(agent_result, 'model', 'Unknown')}\n"
            f"Sources: {len(getattr(agent_result, 'context', []))} verified"
        )
        output_payload = {
            "exam_info": exam_data,
            "system_metadata": {
                "execution_id": getattr(agent_result, 'run_id', None),
                "data_sources": [
                    ctx.source for ctx in getattr(agent_result, 'context', [])
                ],
                "processed_at": datetime.now().isoformat(),
                "content_type": getattr(agent_result, 'content_type', 'json')
            }
        }

        # Save structured output
        filename = f"output/{exam_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output_payload, f, indent=2)
            
        logger.info(f"Completed {exam_name} in {processing_time.total_seconds():.2f}s")
        return exam_data
        
    except Exception as e:
        logger.error(f"Processing failed: {exam_name} - {str(e)}", exc_info=True)
        return None

def read_input_file(file_path: str) -> List[str]:
    """Read and validate input JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Input JSON must be an array of exam names")
            
        logger.info(f"Successfully read input file with {len(data)} exams")
        return data
        
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        raise

def write_output_file(exam_data: ExamInfo,exam_name: str):
    """Write validated output to JSON file"""
    try:
        filename = f"output/{exam_name}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(filename, 'w') as f:
            json.dump(exam_data, f, indent=2)
            
        logger.debug(f"Successfully wrote output file: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error writing output file: {str(e)}")
        raise

def main(input_file: str = "input/exams.json"):
    """Main processing pipeline"""
    logger.info("=== Starting Exam Information System ===")
    
    try:
        # Read input
        exams = read_input_file(input_file)
        
        # Process all exams
        results = []
        for exam_name in exams:
            result = process_exam(exam_name)
            results.append({
                "exam": exam_name,
                "status": "success" if result else "failed",
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate summary report
        summary = {
            "total_exams": len(exams),
            "successful": sum(1 for r in results if r['status'] == 'success'),
            "failed": sum(1 for r in results if r['status'] == 'failed'),
            "processing_time": datetime.now().isoformat(),
            "details": results
        }
        
        with open("output/processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info("=== Processing Complete ===")
        logger.info(f"Summary: {summary}")
        
    except Exception as e:
        logger.critical(f"Fatal error in main process: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":    
    # Create sample input if none exists
    if not os.path.exists("input/exams.json"):
        with open("input/exams.json", 'w') as f:
            json.dump(["GATE 2025"], f, indent=2)
    
    main()