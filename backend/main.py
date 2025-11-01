import os
import re
import json
import math
import logging
from typing import List, Dict, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging for better error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Pydantic Models
class FilterRequest(BaseModel):
    city: Optional[str] = None
    bhk: Optional[int] = None
    budget_max_lakhs: Optional[float] = None
    readiness: Optional[str] = None
    locality: Optional[str] = None


class ChatRequest(BaseModel):
    query: str


# Initialize OpenAI Client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables!")
    logger.warning("The /chat endpoint will not work until you set the API key.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")


# Singleton DataService Class
class DataService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if self._initialized:
            return
        
        self._initialized = True
        self.master_df = None
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # Load and process data on initialization
        self._load_and_process_data()
    
    def _load_csv_with_encoding(self, filename: str) -> pd.DataFrame:
        """Load CSV with fallback encoding handling."""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Loaded {filename} with UTF-8 encoding")
            return df
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 failed for {filename}, trying latin1...")
            df = pd.read_csv(filepath, encoding='latin1')
            logger.info(f"Loaded {filename} with latin1 encoding")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise Exception(f"Error loading {filename}: {str(e)}")
    
    def _clean_price(self, price_str) -> Optional[float]:
        """Convert price to lakhs, handling various formats including raw Rupees."""
        if pd.isna(price_str):
            return None
        
        # Handle numeric input directly (raw Rupees in database)
        if isinstance(price_str, (int, float)):
            # If it's a large number, assume it's in Rupees and convert to Lakhs
            if price_str > 100000:
                return price_str / 100000
            else:
                # Small number, assume already in Lakhs
                return price_str
        
        price_str = str(price_str).strip()
        
        # Match patterns like "1.2 Cr", "90 L", "1.5Cr", etc.
        cr_match = re.search(r'([\d.]+)\s*Cr', price_str, re.IGNORECASE)
        if cr_match:
            return float(cr_match.group(1)) * 100  # Convert Crores to Lakhs
        
        l_match = re.search(r'([\d.]+)\s*L', price_str, re.IGNORECASE)
        if l_match:
            return float(l_match.group(1))  # Already in Lakhs
        
        # Try to parse as plain number
        try:
            num = float(price_str)
            # CRITICAL FIX: If number is large, assume it's in Rupees, not Lakhs
            if num > 100000:  # e.g., 100,001 Rupees
                return num / 100000  # Convert Rupees to Lakhs
            else:
                return num  # Assumes it's already in Lakhs (e.g., 80.5, 90)
        except ValueError:
            return None
    
    def _extract_bhk_numeric(self, bhk_str) -> Optional[int]:
        """Extract numeric integer BHK value, handling decimals, Studio, and type field."""
        if pd.isna(bhk_str):
            return None
        
        bhk_str = str(bhk_str).strip()
        
        # Handle "Studio" or "1RK" explicitly
        if "studio" in bhk_str.lower() or "1rk" in bhk_str.lower() or bhk_str.lower() == "rk":
            return 0  # 0 BHK for Studio/RK
        
        # Match patterns like "3 BHK", "1.5 BHK", "2BHK"
        match = re.search(r'^([\d.]+)\s*BHK', bhk_str, re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                # Check if the number is a whole number (integer)
                if num.is_integer():
                    return int(num)
                else:
                    # It's a float like 1.5. Since our filters are int-based,
                    # we can't match it exactly. Return None.
                    return None
            except ValueError:
                return None
        
        # Fallback: Try to extract first digit from string (for cases like "4BHK" without space)
        digit_match = re.search(r'(\d+)', bhk_str)
        if digit_match:
            return int(digit_match.group(1))
        
        return None
    
    def _clean_for_json(self, data):
        """Recursively clean data structure to ensure JSON serialization."""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif isinstance(data, float):
            if pd.isna(data) or math.isnan(data):
                return None
            return data
        elif pd.isna(data):
            return None
        else:
            return data
    
    def _load_and_process_data(self):
        """Load all CSVs, merge them, and clean the data."""
        try:
            logger.info("="*50)
            logger.info("Loading CSV files...")
            logger.info("="*50)
            
            # Load all CSV files
            project_df = self._load_csv_with_encoding('project.csv')
            logger.info(f"Columns for project.csv: {project_df.columns.to_list()}")
            logger.info(f"Shape: {project_df.shape}")
            
            project_address_df = self._load_csv_with_encoding('ProjectAddress.csv')
            logger.info(f"Columns for ProjectAddress.csv: {project_address_df.columns.to_list()}")
            logger.info(f"Shape: {project_address_df.shape}")
            
            project_config_df = self._load_csv_with_encoding('ProjectConfiguration.csv')
            logger.info(f"Columns for ProjectConfiguration.csv: {project_config_df.columns.to_list()}")
            logger.info(f"Shape: {project_config_df.shape}")
            
            project_config_variant_df = self._load_csv_with_encoding('ProjectConfigurationVariant.csv')
            logger.info(f"Columns for ProjectConfigurationVariant.csv: {project_config_variant_df.columns.to_list()}")
            logger.info(f"Shape: {project_config_variant_df.shape}")
            
            logger.info("="*50)
            logger.info("Merging DataFrames...")
            logger.info("="*50)
            
            # Step 1: Merge project with ProjectAddress
            project_merged = pd.merge(
                project_df,
                project_address_df,
                left_on='id',
                right_on='projectId',
                how='left'
            )
            logger.info(f"Merged project + ProjectAddress: {len(project_merged)} rows")
            
            # Step 2: Merge ProjectConfiguration with ProjectConfigurationVariant
            config_merged = pd.merge(
                project_config_df,
                project_config_variant_df,
                left_on='id',
                right_on='configurationId',
                how='left'
            )
            logger.info(f"Merged ProjectConfiguration + ProjectConfigurationVariant: {len(config_merged)} rows")
            
            # Step 3: Merge the two results
            self.master_df = pd.merge(
                project_merged,
                config_merged,
                left_on='id_x',
                right_on='projectId',
                how='inner'
            )
            logger.info(f"Final merged DataFrame: {len(self.master_df)} rows")
            
            logger.info("="*50)
            logger.info("Cleaning data...")
            logger.info("="*50)
            
            # Clean Price column
            if 'price' in self.master_df.columns:
                self.master_df['price_in_lakhs'] = self.master_df['price'].apply(self._clean_price)
                logger.info("Created 'price_in_lakhs' column")
                
                # Debug: Show sample of price conversions
                sample = self.master_df[['price', 'price_in_lakhs']].head(3)
                logger.info(f"Sample price conversions:\n{sample}")
            else:
                logger.warning("'price' column not found")
                self.master_df['price_in_lakhs'] = None
            
            # Clean BHK column - try customBHK first, fallback to type
            if 'customBHK' in self.master_df.columns:
                self.master_df['bhk_numeric'] = self.master_df['customBHK'].apply(self._extract_bhk_numeric)
                
                # Fallback: If customBHK didn't extract, try type column
                if 'type' in self.master_df.columns:
                    mask = self.master_df['bhk_numeric'].isna()
                    self.master_df.loc[mask, 'bhk_numeric'] = self.master_df.loc[mask, 'type'].apply(self._extract_bhk_numeric)
                
                logger.info("Created 'bhk_numeric' column")
                
                # Debug: Show sample of BHK extractions
                sample = self.master_df[['customBHK', 'type', 'bhk_numeric']].head(3)
                logger.info(f"Sample BHK extractions:\n{sample}")
            else:
                logger.warning("'customBHK' column not found")
                self.master_df['bhk_numeric'] = None
            
            logger.info("="*50)
            logger.info(f"Data loading complete! Total records: {len(self.master_df)}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error during data loading: {str(e)}", exc_info=True)
            raise
    
    def get_filtered_properties(self, filters: FilterRequest) -> List[Dict]:
        """Filter properties based on the provided filters."""
        if self.master_df is None or self.master_df.empty:
            logger.warning("Master DataFrame is empty or None")
            return []
        
        try:
            # Start with the full dataset
            filtered_df = self.master_df.copy()
            
            # Apply city filter using fullAddress column
            if filters.city is not None:
                filtered_df = filtered_df[
                    filtered_df['fullAddress'].str.contains(
                        filters.city,
                        case=False,
                        na=False
                    )
                ]
                logger.info(f"After city filter ({filters.city}): {len(filtered_df)} rows")
            
            # Apply locality filter (also uses fullAddress)
            if filters.locality is not None:
                filtered_df = filtered_df[
                    filtered_df['fullAddress'].str.contains(
                        filters.locality,
                        case=False,
                        na=False
                    )
                ]
                logger.info(f"After locality filter ({filters.locality}): {len(filtered_df)} rows")
            
            # Apply BHK filter
            if filters.bhk is not None:
                filtered_df = filtered_df[filtered_df['bhk_numeric'] == filters.bhk]
                logger.info(f"After BHK filter ({filters.bhk}): {len(filtered_df)} rows")
            
            # Apply budget filter (maximum price)
            if filters.budget_max_lakhs is not None:
                filtered_df = filtered_df[
                    (filtered_df['price_in_lakhs'].notna()) & 
                    (filtered_df['price_in_lakhs'] <= filters.budget_max_lakhs)
                ]
                logger.info(f"After budget filter (<= {filters.budget_max_lakhs}L): {len(filtered_df)} rows")
            
            # Apply readiness filter (using the 'status' column) - FIXED VERSION
            if filters.readiness is not None and 'status' in filtered_df.columns:
                readiness_query = filters.readiness.lower()
                if readiness_query == "ready to move":
                    # Match 'READY' or 'MOVE'
                    filtered_df = filtered_df[
                        filtered_df['status'].str.contains("READY|MOVE", case=False, na=False)
                    ]
                elif readiness_query == "under construction":
                    # Match 'CONSTRUCTION'
                    filtered_df = filtered_df[
                        filtered_df['status'].str.contains("CONSTRUCTION", case=False, na=False)
                    ]
                logger.info(f"After readiness filter ({filters.readiness}): {len(filtered_df)} rows")
            
            # ROBUST NaN FIX: Convert DataFrame to object dtype and replace NaN with None
            # This ensures all NaN values are properly converted before JSON serialization
            filtered_df = filtered_df.astype(object).where(pd.notnull(filtered_df), None)
            
            # Convert to list of dictionaries
            result_list = filtered_df.to_dict('records')
            
            # ADDITIONAL SAFETY: Clean any remaining NaN values using our helper
            result_list = self._clean_for_json(result_list)
            
            logger.info(f"Final filtered results: {len(result_list)} properties")
            return result_list
            
        except Exception as e:
            logger.error(f"Error in get_filtered_properties: {str(e)}", exc_info=True)
            raise


# AI Query Parser Function
async def parse_query_with_ai(query: str) -> FilterRequest:
    """
    Use OpenAI to parse natural language query into structured filters.
    """
    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    
    system_prompt = """You are a real estate search assistant that extracts structured filters from natural language queries.

Your task is to analyze the user's query and extract the following information:
- city: The city name (e.g., "Mumbai", "Bangalore", "Delhi")
- bhk: Number of bedrooms as an integer (e.g., 2, 3, 4). Use 0 for Studio/RK apartments.
- budget_max_lakhs: Maximum budget in lakhs (convert Crores to lakhs: 1 Cr = 100 L)
- readiness: Property readiness status - either "Ready to Move" or "Under Construction"
- locality: Specific area/locality within a city (e.g., "Andheri", "Whitefield", "Dwarka")

Important rules:
1. Only extract information that is explicitly mentioned in the query
2. If something is not mentioned, set it to null
3. For budget, always convert to lakhs (e.g., "1.5 Cr" = 150, "80 lakhs" = 80)
4. For Studio/RK apartments, use bhk: 0
5. For queries asking for "cheapest" or "lowest price", do NOT set a budget_max_lakhs
6. Respond ONLY with a valid JSON object, no other text
7. Use lowercase for city and locality names

Example inputs and outputs:
- "3 BHK in Mumbai under 2 crore" → {"city": "mumbai", "bhk": 3, "budget_max_lakhs": 200, "readiness": null, "locality": null}
- "Ready to move 2BHK in Bangalore Whitefield" → {"city": "bangalore", "bhk": 2, "budget_max_lakhs": null, "readiness": "Ready to Move", "locality": "whitefield"}
- "Studio apartment in Delhi" → {"city": "delhi", "bhk": 0, "budget_max_lakhs": null, "readiness": null, "locality": null}
- "what are the cheapest properties you have?" → {"city": null, "bhk": null, "budget_max_lakhs": null, "readiness": null, "locality": null}

Respond ONLY with the JSON object, nothing else."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse the JSON response
        json_str = response.choices[0].message.content
        parsed_data = json.loads(json_str)
        
        logger.info(f"Parsed query '{query}' to filters: {parsed_data}")
        
        # Create and return FilterRequest
        return FilterRequest(
            city=parsed_data.get("city"),
            bhk=parsed_data.get("bhk"),
            budget_max_lakhs=parsed_data.get("budget_max_lakhs"),
            readiness=parsed_data.get("readiness"),
            locality=parsed_data.get("locality")
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse AI response as JSON: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error calling OpenAI API: {str(e)}"
        )


# AI Summary Generator Function
async def generate_summary_with_ai(query: str, filters: FilterRequest, results: List[Dict]) -> str:
    """
    Generate a natural language summary of search results using AI.
    The summary is grounded in the actual data returned from the search.
    """
    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    
    # Handle case with no results
    if len(results) == 0:
        # Build a descriptive message about what was searched
        search_criteria = []
        if filters.bhk is not None:
            if filters.bhk == 0:
                search_criteria.append("Studio/RK")
            else:
                search_criteria.append(f"{filters.bhk} BHK")
        if filters.city:
            search_criteria.append(f"in {filters.city.title()}")
        if filters.locality:
            search_criteria.append(f"{filters.locality.title()}")
        if filters.budget_max_lakhs:
            budget_cr = filters.budget_max_lakhs / 100
            search_criteria.append(f"under ₹{budget_cr:.2f} Cr")
        if filters.readiness:
            search_criteria.append(f"({filters.readiness})")
        
        criteria_text = " ".join(search_criteria) if search_criteria else "your search criteria"
        
        system_prompt = f"""You are a helpful real estate assistant. The user searched for properties but no results were found.

User's query: "{query}"
Search criteria: {criteria_text}

Write a empathetic, helpful 2-3 sentence response that:
1. Acknowledges that no properties matched their criteria
2. Suggests they might try adjusting their filters (budget, location, or BHK)
3. Remains positive and encouraging

Keep it conversational and friendly."""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the no-results message."}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating no-results summary: {str(e)}", exc_info=True)
            # Fallback message if AI fails
            return f"I couldn't find any properties matching {criteria_text}. Try adjusting your budget, location, or BHK preferences to see more options."
    
    # Handle case with results - create grounded summary
    else:
        # Extract key information from results for data context
        # Limit to first 10 properties to avoid token limits
        sample_results = results[:10]
        
        # Create a simplified data context with essential information
        data_summary = []
        for idx, prop in enumerate(sample_results, 1):
            # Clean all values to avoid NaN issues
            prop_info = {
                "property": idx,
                "name": prop.get("projectName") or prop.get("name") or "Unknown",
                "location": prop.get("fullAddress") or "Unknown",
                "bhk": prop.get("customBHK") or prop.get("type") or "Unknown",
                "price": str(prop.get('price') or 'Unknown'),
                "price_lakhs": prop.get("price_in_lakhs") if prop.get("price_in_lakhs") and not pd.isna(prop.get("price_in_lakhs")) else None,
                "readiness": prop.get("status") or "Unknown",
                "area": prop.get("carpetArea") or "Unknown"
            }
            data_summary.append(prop_info)
        
        # Convert to JSON string for the prompt
        data_context = json.dumps(data_summary, indent=2)
        
        # Calculate some aggregate statistics - filter out None and NaN values
        total_count = len(results)
        prices = [r.get("price_in_lakhs") for r in results 
                 if r.get("price_in_lakhs") is not None and not pd.isna(r.get("price_in_lakhs"))]
        avg_price_lakhs = sum(prices) / len(prices) if prices else None
        min_price_lakhs = min(prices) if prices else None
        max_price_lakhs = max(prices) if prices else None
        
        # Build statistics context
        stats_context = f"""
Total properties found: {total_count}
"""
        if avg_price_lakhs:
            stats_context += f"Average price: ₹{avg_price_lakhs/100:.2f} Cr\n"
        if min_price_lakhs and max_price_lakhs:
            stats_context += f"Price range: ₹{min_price_lakhs/100:.2f} Cr - ₹{max_price_lakhs/100:.2f} Cr\n"
        
        system_prompt = f"""You are a real estate assistant providing a summary of property search results.

User's original query: "{query}"

PROPERTY DATA (first 10 of {total_count} results):
{data_context}

STATISTICS:
{stats_context}

Your task:
Write a natural, conversational 2-4 sentence summary that:
1. Confirms how many properties were found
2. Highlights key insights from the data (e.g., price range, popular locations, common configurations)
3. Mentions 1-2 specific property examples if relevant
4. Is based ONLY on the data provided above - do not make up information

Keep it concise, informative, and helpful. Use natural language, not technical jargon."""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the summary based on the data provided."}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary with AI: {str(e)}", exc_info=True)
            # Fallback summary if AI fails
            price_info = ""
            if avg_price_lakhs:
                price_info = f" with an average price of ₹{avg_price_lakhs/100:.2f} Cr"
            return f"I found {total_count} properties matching your search{price_info}. The results include various options across different locations and configurations."


# Initialize FastAPI app
app = FastAPI(
    title="Property Search API",
    description="AI-powered backend API for property search and filtering",
    version="2.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DataService (singleton - loads data once)
logger.info("🚀 Initializing DataService...")
try:
    data_service = DataService()
    logger.info("✓ DataService initialized successfully!")
except Exception as e:
    logger.error(f"✗ Failed to initialize DataService: {str(e)}", exc_info=True)
    data_service = None


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint to check if backend is running."""
    if data_service and data_service.master_df is not None:
        return {
            "message": "Backend is running and data is loaded.",
            "total_records": len(data_service.master_df),
            "status": "healthy",
            "openai_configured": openai_client is not None
        }
    else:
        return {
            "message": "Backend is running but data failed to load.",
            "status": "error",
            "openai_configured": openai_client is not None
        }


@app.post("/chat")
async def chat_search(request: ChatRequest):
    """
    AI-powered natural language property search with intelligent summaries.
    
    Send a natural language query and get:
    - Parsed filters
    - Filtered property results
    - AI-generated summary grounded in the data
    
    Example queries:
    - "3 BHK in Mumbai under 2 crore"
    - "Ready to move 2BHK in Bangalore"
    - "Studio apartment in Pune under 50 lakhs"
    - "What are the cheapest properties you have?"
    """
    if data_service is None:
        raise HTTPException(
            status_code=500,
            detail="DataService not initialized. Data failed to load."
        )
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Step 1: Parse the natural language query using AI
        filters = await parse_query_with_ai(request.query)
        
        # Step 2: Get filtered properties from the database
        results = data_service.get_filtered_properties(filters)
        
        # Step 3: Generate AI summary grounded in the actual results
        summary = await generate_summary_with_ai(request.query, filters, results)
        
        # Step 4: Return comprehensive response
        return {
            "success": True,
            "query": request.query,
            "summary": summary,
            "parsed_filters": {
                "city": filters.city,
                "bhk": filters.bhk,
                "budget_max_lakhs": filters.budget_max_lakhs,
                "readiness": filters.readiness,
                "locality": filters.locality
            },
            "count": len(results),
            "results": results
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Unexpected error in chat_search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing your search: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "data_loaded": data_service is not None and data_service.master_df is not None,
        "openai_configured": openai_client is not None
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )