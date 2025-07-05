import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import os
from logging.handlers import RotatingFileHandler

from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.tools.duckduckgo import DuckDuckGoTools

# Setup directories
Path("logs").mkdir(exist_ok=True)
Path("input").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# Configure detailed logging
def setup_logger():
    logger = logging.getLogger("exam_info")
    logger.setLevel(logging.DEBUG)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        f"logs/exam_info_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

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
        description="Detailed syllabus breakdown and PDF link"
    )
    
    preparation_resources: Dict[str, Dict[str, List[str]]] = Field(
        ...,
        description="Recommended study materials and practice resources"
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
    
# Initialize agent (same as previous)
agent = Agent(
    model=AzureOpenAI(
        id="gpt-4",  # or gpt-4o if you have access
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-03-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ),
    tools=[DuckDuckGoTools()],
    description="Expert system for retrieving accurate exam information.",
    response_model=ExamInfo,
    max_retries=3,
    validation_level="strict",
)

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

def write_output_file(exam_data: ExamInfo):
    """Write validated output to JSON file"""
    try:
        filename = f"output/{exam_data.id}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(filename, 'w') as f:
            json.dump(exam_data.dict(), f, indent=2)
            
        logger.debug(f"Successfully wrote output file: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error writing output file: {str(e)}")
        raise

def process_exam(exam_name: str) -> Optional[ExamInfo]:
    """Process single exam with error handling"""
    logger.info(f"Starting processing for: {exam_name}")
    
    try:
        start_time = datetime.now()
        
        # Execute main processing
        response: RunResponse = agent.run(
            f"Provide complete 2025 details for {exam_name} "
            "with verified dates and official URLs."
        )
        
        processing_time = datetime.now() - start_time
        exam_data = response.content
        
        # Log success metrics
        logger.info(
            f"Successfully processed {exam_name} in {processing_time.total_seconds():.2f}s\n"
            f"Exam ID: {exam_data.id}\n"
            f"Official Website: {exam_data.highlights['official_website']}"
        )
        
        # Write output
        output_path = write_output_file(exam_data)
        logger.debug(f"Output file created at: {output_path}")
        
        return exam_data
        
    except Exception as e:
        logger.error(f"Failed to process {exam_name}: {str(e)}", exc_info=True)
        return None

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
    # Example input file format:
    # ["NEET UG 2025", "JEE Main 2025", "UPSC Civil Services 2025"]
    
    # Create sample input if none exists
    if not os.path.exists("input/exams.json"):
        with open("input/exams.json", 'w') as f:
            json.dump(["NEET UG 2025", "JEE Main 2025"], f, indent=2)
    
    main()