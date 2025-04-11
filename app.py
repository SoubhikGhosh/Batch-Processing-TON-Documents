from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import os
import zipfile
import io
import logging
import tempfile
import time
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import uuid
import shutil
import json
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import re
import uvicorn
from collections import Counter
import concurrent.futures
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for processing zip files containing different document types using Vertex AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vertex AI Configuration
project = "hbl-uat-ocr-fw-app-prj-spk-4d"
vertexai.init(project=project, location="asia-south1", api_endpoint='asia-south1-aiplatform.googleapis.com')

# Safety settings
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# Document types
DOCUMENT_TYPES = [
    "customer_request_letter",
    "form_15ca",
    "form_15cb",
    "form_a2",
    "invoice",
    "transport_document", 
    "fdd_stationery"
]

# Field definitions with their sources
FIELDS = [
    {"id": 1, "name": "currency", "source": "customer_request_letter"},
    {"id": 2, "name": "amount", "source": "customer_request_letter"},
    {"id": 3, "name": "beneficiary_account_number", "source": "customer_request_letter"},
    {"id": 4, "name": "beneficiary_name", "source": "customer_request_letter"},
    {"id": 5, "name": "beneficiary_address", "source": "customer_request_letter"},
    {"id": 6, "name": "beneficiary_bank_swift_code", "source": "customer_request_letter"},
    {"id": 7, "name": "beneficiary_bank_name", "source": "customer_request_letter"},
    {"id": 8, "name": "beneficiary_bank_address", "source": "customer_request_letter"},
    {"id": 9, "name": "charge_type", "source": "customer_request_letter"},
    {"id": 10, "name": "dd_number", "source": "fdd_stationery"},
    {"id": 11, "name": "intermediary_institution", "source": "customer_request_letter"},
    {"id": 12, "name": "ack_no_form_15ca", "source": "form_15ca"},
    {"id": 13, "name": "ack_no_form_15cb", "source": "form_15cb"},
    {"id": 14, "name": "account_to_be_debited_for_remittance", "source": "customer_request_letter"},
    {"id": 15, "name": "account_to_be_debited_for_charges", "source": "customer_request_letter"},
    {"id": 16, "name": "remittance_account", "source": "customer_request_letter"},
    {"id": 17, "name": "invoice_number", "source": "invoice"},
    {"id": 18, "name": "invoice_date", "source": "invoice"},
    {"id": 19, "name": "transport_document_number", "source": "transport_document"},
    {"id": 20, "name": "transport_document_date", "source": "transport_document"},
    {"id": 21, "name": "port_of_loading", "source": "transport_document"},
    {"id": 22, "name": "port_of_discharge", "source": "transport_document"},
    {"id": 23, "name": "on_board_date", "source": "transport_document"},
    {"id": 24, "name": "remittance_information", "source": "customer_request_letter"},
    {"id": 25, "name": "purpose_code", "source": "form_a2"},
    {"id": 26, "name": "form_15ca", "source": "form_15ca"},
    {"id": 27, "name": "customer_request_letter_date", "source": "customer_request_letter"},
    {"id": 28, "name": "payment_reference_drawer", "source": "customer_request_letter"},
    {"id": 29, "name": "goods_description", "source": "invoice"},
    {"id": 30, "name": "inco_terms", "source": "invoice"},
    {"id": 31, "name": "goods_carrier", "source": "transport_document"},
    {"id": 32, "name": "goods_shipment_date", "source": "transport_document"},
    {"id": 33, "name": "bullion", "source": "invoice"},
    {"id": 34, "name": "bullion_customer", "source": "invoice"},
    {"id": 35, "name": "bullion_delivery", "source": "invoice"},
    {"id": 36, "name": "bullion_weight", "source": "invoice"}
]


# Create a thread pool executor at the module level
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Keep a dictionary of futures for tracking
active_tasks = {}
processed_jobs = {}

# Create a mapping of document types to their fields
DOCUMENT_FIELDS = {}
for field in FIELDS:
    doc_type = field["source"]
    if doc_type not in DOCUMENT_FIELDS:
        DOCUMENT_FIELDS[doc_type] = []
    DOCUMENT_FIELDS[doc_type].append(field["name"])

class DocumentProcessor:
    """Helper class for document processing operations using Vertex AI's multimodal capabilities"""
    
    @staticmethod
    def perform_ocr(file_data: bytes, file_type: str) -> Dict[str, Any]:
        """Process documents using Vertex AI's multimodal capabilities."""
        try:
            result = {"text": "", "pages": []}
            
            # Initialize Vertex AI model
            model = GenerativeModel("gemini-1.5-flash-002", safety_settings=safety_settings)
            
            if file_type.lower() in ["image/jpeg", "image/jpg", "image/png", "image/tiff"]:
                # Process image directly with Vertex AI
                response = model.generate_content(
                    [
                        "Extract all text from this document image. Return only the extracted text without additional comments.",
                        {"mime_type": file_type, "data": file_data}
                    ]
                )
                text = response.text.strip()
                result["text"] = text
                result["pages"].append({"page_num": 1, "text": text})
                
            elif file_type.lower() in ["application/pdf", "pdf"]:
                # Convert PDF to images and process each page with Vertex AI
                images = convert_from_bytes(file_data)
                full_text = ""
                
                for i, image in enumerate(images):
                    # Convert PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="PNG")
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Process with Vertex AI
                    response = model.generate_content(
                        [
                            "Extract all text from this document image. Return only the extracted text without additional comments.",
                            {"mime_type": "image/png", "data": img_bytes}
                        ]
                    )
                    page_text = response.text.strip()
                    full_text += page_text + "\n\n"
                    result["pages"].append({"page_num": i+1, "text": page_text})
                
                result["text"] = full_text
                
            else:
                logger.warning(f"Unsupported file type for Vertex AI processing: {file_type}")
                result["text"] = "Unsupported file type for processing"
                result["error"] = f"Unsupported file type: {file_type}"
                
            return result
            
        except Exception as e:
            logger.error(f"Error during Vertex AI document processing: {str(e)}")
            return {"error": str(e), "text": "", "pages": []}
    
    @staticmethod
    def classify_document(text: str) -> Dict[str, Any]:
        """Classify the document type using Vertex AI."""
        try:
            model = GenerativeModel("gemini-1.5-flash-002", safety_settings=safety_settings)
            
            prompt = """
            You are a document classification expert. Examine the following text extracted from a document and determine which type of financial document it is.
            
            Classify the document into EXACTLY ONE of these categories:
            - customer_request_letter
            - form_15ca
            - form_15cb
            - form_a2
            - invoice
            - transport_document
            - fdd_stationery
            - unknown
            
            Look for specific headers, formatting patterns, and content that uniquely identify each document type:
            
            ### Customer Request Letter
            - Typically contains: "Request for remittance", "Outward remittance", or similar phrases
            - Usually has beneficiary details, bank details, and remittance amount
            - Often contains a formal request structure with date and signature
            
            ### Form 15CA
            - Official Indian tax form for foreign remittances
            - Contains "FORM 15CA" in the header
            - Has sections related to Income Tax Act and remittance declarations
            - Contains an acknowledgment number
            
            ### Form 15CB
            - Certificate from Chartered Accountant related to foreign remittances
            - Contains "FORM 15CB" in the header
            - Has CA certification and registration numbers
            
            ### Form A2
            - Foreign exchange transaction form
            - Contains "FORM A2" in the header
            - Has sections for purpose codes and forex transaction details
            
            ### Invoice
            - May have a Header mentioning Invoice
            - Contains line items with quantities, unit prices, and totals
            - Has invoice number, date, and payment terms
            - Lists buyer and seller information
            
            ### Transport Document
            - Bill of Lading (B/L), Airway Bill, or similar
            - Contains shipping details, ports, vessel information
            - Has consignor and consignee information
            
            ### FDD Stationery
            - Foreign Demand Draft document
            - Contains DD number and banking instrument details
            
            Provide your classification as a simple JSON with two fields:
            {
                "document_type": "one of the categories listed above",
                "confidence": 0.XX (a number between 0 and 1)
            }
            
            Return ONLY this JSON with no explanations or additional text.
            """
            
            classification_response = model.generate_content([prompt, text])
            json_str = classification_response.text.strip()
            
            # Clean up potential JSON formatting
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
                
            classification = json.loads(json_str.strip())
            return classification
            
        except Exception as e:
            logger.error(f"Error during document classification: {str(e)}")
            return {"document_type": "unknown", "confidence": 0.0, "error": str(e)}
    
    @staticmethod
    def extract_fields(text: str, document_type: str) -> Dict[str, Any]:
        """Extract fields from document using Vertex AI."""
        try:
            model = GenerativeModel("gemini-1.5-flash-002", safety_settings=safety_settings)       
            
            # Get fields for this document type
            doc_fields = DOCUMENT_FIELDS.get(document_type, [])
            if not doc_fields:
                return {
                    "extracted_fields": [],
                    "metadata": {
                        "analysis_timestamp": datetime.now().isoformat(),
                        "overall_confidence": 0.0,
                        "error": f"No fields defined for document type: {document_type}"
                    }
                }
            
            # Define field descriptions for better extraction
            field_descriptions = {
                "currency": "The currency for the transaction (e.g., USD, EUR, GBP)",
                "amount": "The amount of money being transferred",
                "beneficiary_account_number": "The bank account number of the recipient",
                "beneficiary_name": "The name of the person or entity receiving the funds",
                "beneficiary_address": "The address of the beneficiary",
                "beneficiary_bank_swift_code": "The SWIFT/BIC code of the beneficiary's bank",
                "beneficiary_bank_name": "The name of the beneficiary's bank",
                "beneficiary_bank_address": "The address of the beneficiary's bank",
                "charge_type": "The type of charges for the transaction (OUR, BEN, SHA)",
                "dd_number": "The Demand Draft number printed on the FDD stationery",
                "intermediary_institution": "Any intermediary bank involved in the transaction",
                "ack_no_form_15ca": "The acknowledgment number on Form 15CA",
                "ack_no_form_15cb": "The acknowledgment number on Form 15CB",
                "account_to_be_debited_for_remittance": "The account from which the main amount will be taken",
                "account_to_be_debited_for_charges": "The account from which the fees will be taken",
                "remittance_account": "The account for the remittance",
                "invoice_number": "The identification number of the invoice",
                "invoice_date": "The date when the invoice was issued",
                "transport_document_number": "The number on the transport document",
                "transport_document_date": "The date on the transport document",
                "port_of_loading": "The port where goods were loaded",
                "port_of_discharge": "The port where goods will be unloaded",
                "on_board_date": "The date when goods were loaded on vessel",
                "remittance_information": "Details about the purpose of the remittance",
                "purpose_code": "The code indicating the purpose of the foreign exchange transaction",
                "form_15ca": "Details from the Form 15CA document",
                "customer_request_letter_date": "The date on the customer request letter",
                "payment_reference_drawer": "Reference information for the payment drawer",
                "goods_description": "Description of the goods on the invoice",
                "inco_terms": "International Commercial Terms on the invoice (e.g., FOB, CIF)",
                "goods_carrier": "The carrier/vessel transporting the goods",
                "goods_shipment_date": "The date when goods were shipped",
                "bullion": "Whether the invoice is for bullion (yes/no)",
                "bullion_customer": "The customer for bullion transaction",
                "bullion_delivery": "Delivery details for bullion",
                "bullion_weight": "Weight of bullion being transacted"
            }
            
            # Create field list with descriptions
            fields_with_descriptions = []
            for field in doc_fields:
                description = field_descriptions.get(field, "")
                fields_with_descriptions.append(f"- {field}: {description}")
            
            fields_list = "\n".join(fields_with_descriptions)
            
            prompt = f"""
            You are a specialized financial document analyzer with expertise in extracting information from {document_type.replace("_", " ").title()} documents.
            
            Extract the following fields from the document with maximum precision:
            {fields_list}
            
            For each field:
            1. Extract the exact value as it appears in the document
            2. If the text is unclear, make a reasonable approximation
            3. For dates, standardize to YYYY-MM-DD format where possible
            4. For monetary amounts, include both value and currency
            5. Assign a confidence score between 0.0 and 1.0 for each extraction
            6. If a field cannot be found or extracted, provide a reason why
            
            For each field, provide:
            - The extracted value
            - A confidence score (0.0-1.0)
            - The exact text segment from which you extracted the information
            - A reason if the field couldn't be extracted
            
            Provide your analysis in the following strict JSON structure only:
            {{
                "extracted_fields": [
                    {{
                        "field_name": "field_name",
                        "value": "extracted value",
                        "confidence": 0.XX,
                        "source_text": "text from document",
                        "reason": "reason if field couldn't be extracted properly"
                    }}
                ],
                "metadata": {{
                    "analysis_timestamp": "ISO timestamp",
                    "overall_confidence": 0.XX
                }}
            }}
            
            Return ONLY valid JSON with no explanations or additional text.
            """
            
            extraction_response = model.generate_content([prompt, text])
            json_str = extraction_response.text.strip()
            
            # Clean up potential JSON formatting
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
                
            extraction_result = json.loads(json_str.strip())
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error during field extraction: {str(e)}")
            return {
                "extracted_fields": [],
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "overall_confidence": 0.0,
                    "error": str(e)
                }
            }
            
    @staticmethod
    def process_multimodal_document(file_data: bytes, file_type: str) -> Dict[str, Any]:
        """Process a document using Vertex AI's multimodal capabilities."""
        try:
            # Initialize Vertex AI model
            model = GenerativeModel("gemini-1.5-flash-002", safety_settings=safety_settings)
            
            result = {}
            
            # Create a Vertex AI Part from the file data
            file_part = Part.from_data(data=file_data, mime_type=file_type)
            
            # For images, process directly
            if file_type.lower() in ["image/jpeg", "image/jpg", "image/png", "image/tiff"]:
                # First, classify the document type using the image directly
                classification_prompt = """
                You are a document classification expert. Examine this document image and determine which type of financial document it is.
                
                Classify the document into EXACTLY ONE of these categories:
                - customer_request_letter
                - form_15ca
                - form_15cb
                - form_a2
                - invoice
                - transport_document
                - fdd_stationery
                - unknown
                
                Return ONLY a JSON with the document_type and confidence, like:
                {"document_type": "category_name", "confidence": 0.X}
                """
                
                classification_response = model.generate_content([
                    classification_prompt,
                    file_part
                ])
                
                json_str = classification_response.text.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                    
                classification = json.loads(json_str.strip())
                document_type = classification["document_type"]
                confidence = classification["confidence"]
                
                # Now extract text and fields based on the document type
                doc_fields = DOCUMENT_FIELDS.get(document_type, [])
                fields_str = ", ".join(doc_fields)
                
                extraction_prompt = f"""
                This is a {document_type.replace('_', ' ')} document. 
                Extract all text content AND the following fields: {fields_str}.
                
                For each field, provide the value and a confidence score between 0.0 and 1.0.
                Format your response as a JSON with two parts:
                1. "full_text": The complete text from the document
                2. "extracted_fields": An array of objects with field_name, value, and confidence
                """
                
                extraction_response = model.generate_content([
                    extraction_prompt,
                    file_part
                ])
                
                extraction_json_str = extraction_response.text.strip()
                if extraction_json_str.startswith("```json"):
                    extraction_json_str = extraction_json_str[7:]
                if extraction_json_str.endswith("```"):
                    extraction_json_str = extraction_json_str[:-3]
                
                extraction_result = json.loads(extraction_json_str.strip())
                
                # Combine the results
                result = {
                    "document_type": document_type,
                    "confidence": confidence,
                    "text": extraction_result.get("full_text", ""),
                    "extracted_fields": extraction_result.get("extracted_fields", []),
                    "pages": [{"page_num": 1, "text": extraction_result.get("full_text", "")}]
                }
                
            elif file_type.lower() in ["application/pdf", "pdf"]:
                # For PDFs, we still need to convert to images and process page by page
                images = convert_from_bytes(file_data)
                full_text = ""
                all_fields = []
                document_type_votes = {}
                
                for i, image in enumerate(images):
                    # Convert PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="PNG")
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Create Vertex AI Part for this image
                    page_part = Part.from_data(data=img_bytes, mime_type="image/png")
                    
                    # If it's the first page, classify the document
                    if i == 0:
                        classification_prompt = """
                        You are a document classification expert. Examine this document image and determine which type of financial document it is.
                        
                        Classify the document into EXACTLY ONE of these categories:
                        - customer_request_letter
                        - form_15ca
                        - form_15cb
                        - form_a2
                        - invoice
                        - transport_document
                        - fdd_stationery
                        - unknown
                        
                        Return ONLY a JSON with the document_type and confidence, like:
                        {"document_type": "category_name", "confidence": 0.X}
                        """
                        
                        classification_response = model.generate_content([
                            classification_prompt,
                            page_part
                        ])
                        
                        json_str = classification_response.text.strip()
                        if json_str.startswith("```json"):
                            json_str = json_str[7:]
                        if json_str.endswith("```"):
                            json_str = json_str[:-3]
                            
                        classification = json.loads(json_str.strip())
                        document_type = classification["document_type"]
                        document_type_votes[document_type] = classification["confidence"]
                    
                    # Extract text from each page
                    text_prompt = "Extract all text from this document image. Return only the extracted text."
                    text_response = model.generate_content([
                        text_prompt,
                        page_part
                    ])
                    page_text = text_response.text.strip()
                    full_text += page_text + "\n\n"
                    
                # Determine final document type (use the one with highest confidence)
                document_type = max(document_type_votes.items(), key=lambda x: x[1])[0] if document_type_votes else "unknown"
                confidence = document_type_votes.get(document_type, 0.0)
                
                # Now extract fields based on the document type and the full text
                doc_fields = DOCUMENT_FIELDS.get(document_type, [])
                if doc_fields and full_text:
                    fields_str = ", ".join(doc_fields)
                    
                    extraction_prompt = f"""
                    This is a {document_type.replace('_', ' ')} document with the following text:
                    
                    {full_text[:4000]}  # Limit text if too long
                    
                    Extract the following fields: {fields_str}.
                    
                    For each field, provide the value and a confidence score between 0.0 and 1.0.
                    Format your response as a JSON with an array of extracted_fields objects with field_name, value, and confidence.
                    """
                    
                    extraction_response = model.generate_content(extraction_prompt)
                    
                    extraction_json_str = extraction_response.text.strip()
                    if extraction_json_str.startswith("```json"):
                        extraction_json_str = extraction_json_str[7:]
                    if extraction_json_str.endswith("```"):
                        extraction_json_str = extraction_json_str[:-3]
                    
                    extraction_result = json.loads(extraction_json_str.strip())
                    all_fields = extraction_result.get("extracted_fields", [])
                
                # Combine the results
                result = {
                    "document_type": document_type,
                    "confidence": confidence,
                    "text": full_text,
                    "extracted_fields": all_fields,
                    "pages": [{"page_num": i+1, "text": page_text} for i, page_text in enumerate(full_text.split("\n\n"))]
                }
                
            else:
                logger.warning(f"Unsupported file type for Vertex AI processing: {file_type}")
                result = {
                    "error": f"Unsupported file type: {file_type}",
                    "text": "",
                    "pages": [],
                    "document_type": "unknown",
                    "confidence": 0.0,
                    "extracted_fields": []
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Error during multimodal document processing: {str(e)}")
            return {
                "error": str(e),
                "text": "",
                "pages": [],
                "document_type": "unknown",
                "confidence": 0.0,
                "extracted_fields": []
            }
def process_zip_files(file_contents: List[bytes], file_names: List[str], job_id: str):
    """Process multiple zip files and generate Excel report using Gemini's multimodal capabilities."""

    logger.info(f"Starting process_zip_files for job {job_id}")
    logger.info(f"Number of files: {len(file_contents)}")
    logger.info(f"File names: {file_names}")

    try:
        # Create a temp directory for this job
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store results for each folder
        folder_results = {}
        total_files = 0
        processed_files = 0
        
        # Process each zip file
        for zip_content, zip_name in zip(file_contents, file_names):
            
            # Extract the zip file to temp directory
            zip_dir = os.path.join(temp_dir, os.path.splitext(zip_name)[0])
            os.makedirs(zip_dir, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(zip_dir)
            
            # Process each folder in the zip
            for root, dirs, files in os.walk(zip_dir):
                # Skip the root directory
                if root == zip_dir:
                    continue
                
                # Get folder name (relative to zip)
                rel_path = os.path.relpath(root, zip_dir)
                folder_name = rel_path
                
                # Skip if there are no files
                if not files:
                    continue
                
                # Initialize folder results if not already present
                if folder_name not in folder_results:
                    folder_results[folder_name] = []
                
                # Track document types in this folder
                document_counts = Counter()
                
                # First pass: process all documents with multimodal approach
                for file in files:
                    if file.startswith('.') or file.startswith('~'):
                        continue  # Skip hidden files
                    
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    try:
                        # Get file extension
                        _, ext = os.path.splitext(file)
                        
                        # Only process supported file types
                        if ext.lower() not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                            continue
                        
                        # Determine file type
                        file_type = {
                            '.pdf': 'application/pdf',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.tiff': 'image/tiff',
                            '.tif': 'image/tiff'
                        }.get(ext.lower(), 'application/octet-stream')
                        
                        # Read file
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Process document with Gemini's multimodal capabilities
                        result = DocumentProcessor.process_multimodal_document(file_data, file_type)
                        
                        # Extract results
                        doc_type = result.get("document_type", "unknown")
                        confidence = result.get("confidence", 0.0)
                        
                        # Count document types
                        if doc_type != "unknown":
                            document_counts[doc_type] += 1
                        
                        # Create a unique file ID for tracking
                        file_info = {
                            "id": str(uuid.uuid4()),
                            "job_id": job_id,
                            "file_name": file, 
                            "folder_name": folder_name, 
                            "file_path": file_path, 
                            "document_type": doc_type, 
                            "confidence": confidence, 
                            "processing_status": "processed"
                        }
                        
                        # Store extracted fields
                        file_info["extracted_fields"] = []
                        for field in result.get("extracted_fields", []):
                            field_info = {
                                "field_name": field.get("field_name", ""),
                                "value": field.get("value", ""),
                                "confidence": field.get("confidence", 0.0),
                                "reason": field.get("reason", "")
                            }
                            file_info["extracted_fields"].append(field_info)
                            
                            # Add to folder results
                            folder_results[folder_name].append({
                                "filepath": file_path,
                                "field_name": field_info["field_name"],
                                "value": field_info["value"],
                                "confidence": field_info["confidence"],
                                "reason": field_info["reason"]
                            })
                        
                        processed_files += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                
                # Determine the dominant document type
                dominant_type = document_counts.most_common(1)
                if dominant_type:
                    dominant_doc_type = dominant_type[0][0]
                    logger.info(f"Folder {folder_name}: Dominant document type is {dominant_doc_type}")
        
        # Generate Excel report
        excel_path = os.path.join(output_dir, f"extraction_results_{job_id}.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Similar to original implementation, but using folder_results instead of database queries
            
            for folder_name, results in folder_results.items():
                if not results:
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame()
                
                # Group by filepath
                filepath_groups = {}
                
                for item in results:
                    filepath = item["filepath"]
                    if filepath not in filepath_groups:
                        filepath_groups[filepath] = {
                            "filepath": filepath,
                        }
                    
                    # Add field name, value, confidence
                    if "field_name" in item:
                        field_name = item["field_name"]
                        filepath_groups[filepath][field_name] = item["value"]
                        filepath_groups[filepath][f"{field_name}_conf"] = item["confidence"]
                        
                        if item.get("reason"):
                            filepath_groups[filepath][f"{field_name}_reason"] = item["reason"]
                
                # Convert to DataFrame
                if filepath_groups:
                    df = pd.DataFrame(list(filepath_groups.values()))
                    
                    # Reorder columns to put field and confidence side by side
                    cols = ["filepath"]
                    for field in FIELDS:
                        field_name = field["name"]
                        if field_name in df.columns:
                            cols.append(field_name)
                            cols.append(f"{field_name}_conf")
                            if f"{field_name}_reason" in df.columns:
                                cols.append(f"{field_name}_reason")
                    
                    # Use only columns that exist in the DataFrame
                    cols = [col for col in cols if col in df.columns]
                    if cols:  # Only reindex if columns exist
                        df = df[cols]
                
                # Create a sanitized sheet name
                sheet_name = re.sub(r'[\\/*?[\]:]', '_', folder_name)
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:28] + '...'
                
                # Write to Excel
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Update job status in-memory
        processed_jobs[job_id] = {
            "status": "completed",
            "start_time": time.time(),
            "end_time": time.time(),
            "total_files": total_files,
            "processed_files": processed_files,
            "output_file_path": excel_path
        }
        
        logger.info(f"Job {job_id} completed. Output file: {excel_path}")
        return excel_path
        
    except Exception as e:
        logger.error(f"Error processing zip files: {str(e)}")
        
        # Update job status to failed
        processed_jobs[job_id] = {
            "status": "failed",
            "start_time": time.time(),
            "end_time": time.time(),
            "total_files": 0,
            "processed_files": 0,
            "error_message": str(e)
        }
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up temp directory: {str(cleanup_error)}")
        
        raise e
    finally:
        # Clean up temp directory after a delay
        def delayed_cleanup():
            time.sleep(3600)  # Keep files for 1 hour
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error during delayed cleanup: {str(e)}")
        
        # Start cleanup in background
        import threading
        threading.Thread(target=delayed_cleanup).start()

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...)
):
    """
    Upload zip files containing folders of documents.
    Each zip file can contain multiple folders, and each folder will be analyzed separately.
    The dominant document type in each folder will be determined, and fields will be extracted accordingly.
    Results will be provided in an Excel file with one sheet per folder.
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Read the file contents into memory before background processing
        file_contents = []
        file_names = []
        for file in files:
            content = await file.read()
            file_contents.append(content)
            file_names.append(file.filename)
        
        # Initialize job status in-memory
        processed_jobs[job_id] = {
            "status": "processing",
            "start_time": time.time(),
            "total_files": 0,
            "processed_files": 0,
            "job_id": job_id
        }
        
        # Define a wrapper function to handle exceptions and update the job status
        def process_wrapper(job_id, file_contents, file_names):
            try:
                return process_zip_files(file_contents, file_names, job_id)
            except Exception as e:
                # Log the full traceback
                logger.error(f"Error processing job {job_id}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Update job status to failed
                processed_jobs[job_id] = {
                    "status": "failed",
                    "end_time": time.time(),
                    "job_id": job_id,
                    "error_message": str(e)
                }
                
                # Re-raise to update the future's exception
                raise
        
        # Submit task to thread pool executor
        future = executor.submit(
            process_wrapper, 
            job_id,
            file_contents, 
            file_names
        )
        
        # Add a callback to handle completion
        def on_complete(future):
            try:
                # Remove the task from active tasks
                active_tasks.pop(job_id, None)
                # If the task completed successfully, the result is already handled in process_zip_files
                if future.exception():
                    logger.error(f"Task for job {job_id} failed: {future.exception()}")
            except Exception as e:
                logger.error(f"Error in on_complete callback: {str(e)}")
        
        future.add_done_callback(on_complete)
        
        # Store the future for tracking
        active_tasks[job_id] = future
        
        return {
            "status": "processing",
            "job_id": job_id,
            "message": "Files uploaded successfully. Processing started with Gemini's multimodal capabilities.",
            "files": file_names
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    try:
        # Try to fetch job from processed_jobs
        job = processed_jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
        # Create a copy to avoid modifying the original
        job_dict = job.copy()
        
        # Convert timestamps to ISO format if they are numeric
        for key in ["start_time", "end_time"]:
            if isinstance(job_dict.get(key), (int, float)):
                job_dict[key] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(job_dict[key]))
        
        return job_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
   
@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download the results of a completed job."""
    try:
        # Fetch job from processed_jobs
        job = processed_jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
        if job.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed. Current status: {job.get('status', 'unknown')}"
            )
        
        output_path = job.get("output_file_path")
        
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Output file not found")
        
        return FileResponse(
            path=output_path,
            filename=f"extraction_results_{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/stats")
async def get_stats():
    """Get aggregate statistics for the dashboard."""
    try:
        # Gather job stats from in-memory processed_jobs
        job_stats = {}
        total_recognized_files = 0
        total_files = 0
        
        for job in processed_jobs.values():
            # Count job statuses
            status = job.get("status", "unknown")
            job_stats[status] = job_stats.get(status, 0) + 1
            
            # Count files if available
            total_files += job.get("total_files", 0)
        
        # Compute processing times for last 10 completed jobs
        processing_times = []
        completed_jobs = [
            job for job in processed_jobs.values() 
            if job.get("status") == "completed" and 
               job.get("start_time") and 
               job.get("end_time")
        ]
        
        # Sort by end time and take last 10
        completed_jobs.sort(key=lambda x: x.get('end_time', 0), reverse=True)
        
        for job in completed_jobs[:10]:
            duration = job.get('end_time', 0) - job.get('start_time', 0)
            processing_times.append({
                "job_id": job.get("job_id", ""),
                "duration": duration
            })
        
        return {
            "jobs": job_stats,
            "files": {
                "total": total_files,
                "recognized": total_recognized_files  # Note: This would require tracking recognized files
            },
            "processing_times": processing_times
        }
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "total_active_jobs": len(active_tasks),
            "total_processed_jobs": len(processed_jobs)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)