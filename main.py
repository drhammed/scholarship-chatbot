import streamlit as st
import os
from dotenv import load_dotenv
from scholarship_bot import ScholarshipBot, ConversationState
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Scholarship Chatbot by drhammed",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Scholarship Chatbot by drhammed")
st.write("Hello! I'm your friendly chatbot. I'm here to help answer your questions regarding scholarships and funding for students, and provide information. I'm also super fast! Let's start!")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_options = [
        "llama3-70b-8192",
        "llama3-8b-8192", 
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview"
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)
    
    # API Keys (from secrets or environment)
    try:
        groq_api_key = st.secrets["api_keys"]["GROQ_API_KEY"]
        tavily_api_key = st.secrets["api_keys"]["TAVILY_API_KEY"]
        st.success("✅ API Keys Loaded")
    except Exception as e:
        # Try to get from environment variables (for development)
        groq_api_key = os.getenv("GROQ_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not groq_api_key or not tavily_api_key:
            st.error("❌ API Keys Not Found")
            st.error("Please configure GROQ_API_KEY and TAVILY_API_KEY in Streamlit secrets or environment variables")
            st.stop()
        else:
            st.success("✅ API Keys Loaded from Environment")
    
    # Control buttons
    st.header("Controls")
    if st.button("🔄 Reset Conversation"):
        if 'scholarship_bot' in st.session_state:
            st.session_state.scholarship_bot.reset_conversation()
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📋 Show Profile"):
        if 'scholarship_bot' in st.session_state:
            profile = st.session_state.scholarship_bot.user_profile
            st.write("**Current Profile:**")
            st.write(f"Field of Study: {profile.field_of_study}")
            st.write(f"Education Level: {profile.education_level}")
            st.write(f"Location: {profile.location}")
            st.write(f"Citizenship: {profile.citizenship}")
            st.write(f"GPA: {profile.gpa}")
            st.write(f"Financial Need: {profile.financial_need}")
            st.write(f"Research Interests: {', '.join(profile.research_interests)}")
            st.write(f"Career Goals: {profile.career_goals}")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'scholarship_bot' not in st.session_state:
    try:
        st.session_state.scholarship_bot = ScholarshipBot(
            groq_api_key=groq_api_key,
            tavily_api_key=tavily_api_key,
            model_name=selected_model
        )
    except Exception as e:
        st.error(f"Failed to initialize Scholarship Bot: {str(e)}")
        st.stop()

# Display conversation state
bot = st.session_state.scholarship_bot
state_colors = {
    ConversationState.PROFILING: "🔍",
    ConversationState.SEARCHING: "🔎",
    ConversationState.RESPONDING: "💬",
    ConversationState.COMPLETE: "✅"
}

st.info(f"**Current Phase:** {state_colors.get(bot.state, '❓')} {bot.state.value.title()}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize with welcome message
if len(st.session_state.messages) == 0:
    welcome_msg = """
    👋 Welcome to the Scholarship Guidance System!
    
    I'm your AI-powered scholarship advisor with three specialized agents:
    - 🔍 **Profiler Agent**: Collects your academic and personal information
    - 🔎 **Research Agent**: Searches for scholarships using live web data
    - 💬 **Response Agent**: Provides structured recommendations and guidance
    
    Let's start by getting to know you better. What field of study are you interested in?
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    with st.chat_message("assistant"):
        st.markdown(welcome_msg)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                # Process the message through the scholarship bot
                response = bot.process_message(prompt)
                
                # Display response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Error processing your request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Display system info in expander
with st.expander("ℹ️ System Information"):
    st.write("**Model:** ", selected_model)
    st.write("**Conversation State:** ", bot.state.value)
    st.write("**Profile Complete:** ", bot.user_profile.is_complete())
    st.write("**Pending Confirmation:** ", bot.pending_confirmation)
    st.write("**Messages in History:** ", len(st.session_state.messages))
    
    if bot.search_results:
        st.write("**Search Results Available:** ✅")
    else:
        st.write("**Search Results Available:** ❌")

# Footer
st.markdown("---")
st.markdown("🎓 **Scholarship Guidance System** - Powered by Multi-Agent AI Architecture")
