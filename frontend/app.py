import streamlit as st
import requests
from typing import Dict, List, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

BACKEND_URL = "http://127.0.0.1:8000/chat"
BACKEND_HEALTH_URL = "http://127.0.0.1:8000/health"

# Page configuration
st.set_page_config(
    page_title="Property Chatbot 🤖",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================================

st.markdown("""
<style>
    /* Main chat container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Property card styling */
    .property-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .property-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    
    .property-price {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4CAF50;
        margin: 0.5rem 0;
    }
    
    .property-detail {
        display: inline-block;
        background: #f0f0f0;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.3rem 0.3rem 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .property-status {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-ready {
        background: #d4edda;
        color: #155724;
    }
    
    .status-construction {
        background: #fff3cd;
        color: #856404;
    }
    
    .property-address {
        color: #666;
        font-size: 0.95rem;
        margin: 0.5rem 0;
        line-height: 1.5;
    }
    
    /* Header styling */
    .chat-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_backend_health() -> bool:
    """Check if the backend is running and healthy."""
    try:
        response = requests.get(BACKEND_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def format_price(price_lakhs: Optional[float]) -> str:
    """
    Format price in lakhs to a readable string.
    Backend returns price_in_lakhs as float (e.g., 150.0 = ₹150 L = ₹1.5 Cr)
    """
    if price_lakhs is None or price_lakhs == 0:
        return "Price not available"
    
    if price_lakhs >= 100:
        crores = price_lakhs / 100
        return f"₹{crores:.2f} Cr"
    else:
        return f"₹{price_lakhs:.2f} L"


def get_status_class(status: str) -> str:
    """Get CSS class for property status."""
    if not status:
        return "status-construction"
    
    status_upper = status.upper()
    if "READY" in status_upper or "MOVE" in status_upper:
        return "status-ready"
    return "status-construction"


def format_status(status: str) -> str:
    """Format status text for display."""
    if not status:
        return "Status Unknown"
    
    status_map = {
        "READY_TO_MOVE": "✅ Ready to Move",
        "UNDER_CONSTRUCTION": "🏗️ Under Construction",
        "READY TO MOVE": "✅ Ready to Move",
        "UNDER CONSTRUCTION": "🏗️ Under Construction",
    }
    
    # Check direct match
    status_upper = status.upper()
    for key, value in status_map.items():
        if key in status_upper:
            return value
    
    # Default: clean up the status text
    return status.replace("_", " ").title()


def render_property_card(property_data: Dict, index: int):
    """
    Render a single property card with all details.
    Uses exact field names from backend's merged DataFrame.
    """
    
    # Extract property details from backend response
    # Backend fields: projectName, fullAddress, customBHK, type, price_in_lakhs, status, 
    # carpetArea, bathrooms, balcony, furnishedType, possessionDate
    
    project_name = property_data.get('projectName', 'Unnamed Project')
    full_address = property_data.get('fullAddress', 'Address not available')
    
    # BHK - customBHK is the primary field
    bhk = property_data.get('customBHK') or property_data.get('type', 'N/A')
    
    # Price - backend returns price_in_lakhs (cleaned float)
    price_lakhs = property_data.get('price_in_lakhs')
    
    # Status field
    status = property_data.get('status', '')
    
    # Additional fields
    carpet_area = property_data.get('carpetArea')
    bathrooms = property_data.get('bathrooms')
    balcony = property_data.get('balcony')
    furnished = property_data.get('furnishedType', '')
    possession_date = property_data.get('possessionDate')
    
    # Format price
    price_str = format_price(price_lakhs)
    
    # Create expander with property name
    with st.expander(f"🏠 {project_name} - {price_str}", expanded=(index < 2)):
        
        # Price (large and prominent)
        st.markdown(f'<div class="property-price">{price_str}</div>', unsafe_allow_html=True)
        
        # Status badge
        status_class = get_status_class(status)
        status_text = format_status(status)
        st.markdown(
            f'<span class="property-status {status_class}">{status_text}</span>',
            unsafe_allow_html=True
        )
        
        # Property configuration in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if bhk and str(bhk) != 'N/A':
                st.markdown(f"**🛏️ Configuration**")
                st.write(bhk)
        
        with col2:
            if carpet_area:
                st.markdown(f"**📐 Carpet Area**")
                # Handle both numeric and string formats
                area_str = str(carpet_area)
                if not any(unit in area_str.lower() for unit in ['sq', 'ft', 'm']):
                    area_str = f"{area_str} sq.ft"
                st.write(area_str)
        
        with col3:
            if bathrooms:
                st.markdown(f"**🚿 Bathrooms**")
                st.write(f"{bathrooms}")
        
        # Additional details
        st.markdown("---")
        
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            if balcony is not None:
                st.markdown(f"**🌅 Balconies:** {balcony}")
            if furnished and furnished != 'N/A':
                furnished_clean = str(furnished).replace('_', ' ').title()
                st.markdown(f"**🪑 Furnished:** {furnished_clean}")
        
        with details_col2:
            if possession_date and str(possession_date).lower() not in ['null', 'none']:
                # Clean up possession date
                poss_str = str(possession_date)
                if 'T' in poss_str:
                    poss_str = poss_str.split('T')[0]
                st.markdown(f"**📅 Possession:** {poss_str}")
        
        # Address (full width)
        st.markdown("---")
        st.markdown(f"**📍 Address:**")
        st.write(full_address)


def send_query_to_backend(query: str) -> Optional[Dict]:
    """Send query to backend and return response."""
    try:
        response = requests.post(
            BACKEND_URL,
            json={"query": query},
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Backend error (Status {response.status_code}). Please try again.")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The backend is taking too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the backend. Please ensure it's running at http://127.0.0.1:8000")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic."""
    
    # Header
    st.markdown("""
        <div class="chat-header">
            <h1>🏠 Property Chatbot</h1>
            <p style="color: #666; font-size: 1.1rem;">
                Ask me anything about properties in natural language!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check backend health (non-blocking)
    backend_healthy = check_backend_health()
    if not backend_healthy:
        st.warning("⚠️ Backend is not responding. Please start the FastAPI server at `http://127.0.0.1:8000`")
        with st.expander("ℹ️ How to start the backend"):
            st.code("python main.py", language="bash")
            st.caption("Or use: `uvicorn main:app --reload`")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar with examples and controls
    with st.sidebar:
        st.markdown("### 💡 Example Queries")
        st.markdown("""
        - "3 BHK in Mumbai under 2 crore"
        - "Ready to move 2 BHK in Bangalore"
        - "Studio apartment in Delhi"
        - "Properties in Pune under 1 crore"
        - "2 BHK in Andheri under 150 lakhs"
        - "Show me all properties"
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This chatbot helps you find properties using natural language queries.
        
        **Powered by:**
        - FastAPI Backend
        - OpenAI GPT-4o-mini
        - Streamlit UI
        """)
        
        st.markdown("---")
        st.caption("🎓 Company Assignment Project")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # If this is an assistant message with properties, render them
            if message["role"] == "assistant" and "properties" in message:
                properties = message["properties"]
                
                if properties and len(properties) > 0:
                    st.markdown(f"---")
                    
                    for idx, property_data in enumerate(properties):
                        render_property_card(property_data, idx)
                elif "properties" in message and len(message["properties"]) == 0:
                    st.info("No properties found matching your criteria.")
    
    # Chat input
    user_query = st.chat_input("Ask about properties..." if backend_healthy else "Backend offline - start server to chat")
    
    if user_query:
        # Check backend before processing
        if not backend_healthy:
            st.error("🔌 Backend is not running. Please start the server first.")
            return
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Get response from backend
        with st.spinner("🔍 Searching for properties..."):
            response_data = send_query_to_backend(user_query)
        
        if response_data:
            # Extract data from backend response
            # Backend returns: {success, query, summary, parsed_filters, count, results}
            success = response_data.get('success', True)
            summary = response_data.get('summary', '')
            results = response_data.get('results', [])
            count = response_data.get('count', len(results))
            parsed_filters = response_data.get('parsed_filters', {})
            
            # Display assistant response
            with st.chat_message("assistant"):
                # Show AI-generated summary
                if summary:
                    st.write(summary)
                else:
                    if count > 0:
                        st.write(f"I found {count} {'property' if count == 1 else 'properties'} matching your criteria.")
                    else:
                        st.write("Let me search for properties matching your criteria.")
                
                # Show filters applied (informative debug info)
                if parsed_filters and any(v is not None for v in parsed_filters.values()):
                    filter_parts = []
                    
                    if parsed_filters.get('city'):
                        filter_parts.append(f"📍 {parsed_filters['city'].title()}")
                    
                    if parsed_filters.get('locality'):
                        filter_parts.append(f"🏘️ {parsed_filters['locality'].title()}")
                    
                    bhk_val = parsed_filters.get('bhk')
                    if bhk_val is not None:
                        if bhk_val == 0:
                            filter_parts.append(f"🛏️ Studio/RK")
                        else:
                            filter_parts.append(f"🛏️ {bhk_val} BHK")
                    
                    if parsed_filters.get('budget_max_lakhs'):
                        filter_parts.append(f"💰 Under {format_price(parsed_filters['budget_max_lakhs'])}")
                    
                    if parsed_filters.get('readiness'):
                        filter_parts.append(f"📊 {parsed_filters['readiness']}")
                    
                    if filter_parts:
                        st.info("**Filters applied:** " + " | ".join(filter_parts))
                
                # Render property cards
                if results and len(results) > 0:
                    st.markdown(f"---")
                    
                    for idx, property_data in enumerate(results):
                        render_property_card(property_data, idx)
                elif count == 0 or not results:
                    st.warning("⚠️ No properties found matching your criteria. Try adjusting your search!")
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": summary if summary else f"Found {count} properties.",
                "properties": results  # Store properties for re-rendering
            })
        else:
            # Error already displayed by send_query_to_backend
            with st.chat_message("assistant"):
                st.error("Sorry, I couldn't process your request. Please try again.")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sorry, something went wrong. Please try again."
            })


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()