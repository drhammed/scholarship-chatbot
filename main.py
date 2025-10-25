import streamlit as st
import os
from dotenv import load_dotenv
from scholarship_bot import ScholarshipBot
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Scholarship Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🎓 Scholarship Chatbot")

# Sidebar for model selection and configuration
with st.sidebar:
    st.header("Settings")

    # Model Selection
    st.subheader("Model Provider")
    model_provider = st.selectbox(
        "Choose Provider:",
        ["Ollama (Cloud)", "Groq"]
    )

    if model_provider == "Ollama (Cloud)":
        ollama_models = [
            'gpt-oss:20b',
            'gpt-oss:120b'
        ]
        selected_model = st.selectbox("Model:", ollama_models)
        use_ollama = True
    else:  # Groq
        groq_models = [
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'groq/compound-mini',
            'llama3-70b-8192',
            'llama3-8b-8192'
        ]
        selected_model = st.selectbox("Model:", groq_models)
        use_ollama = False

    # API Keys
    try:
        groq_api_key = st.secrets["api_keys"]["GROQ_API_KEY"]
        tavily_api_key = st.secrets["api_keys"]["TAVILY_API_KEY"]
        ollama_api_key = st.secrets["api_keys"].get("OLLAMA_API_KEY", "")
        st.success("API Keys Loaded")
    except Exception as e:
        # Try to get from environment variables (for development)
        groq_api_key = os.getenv("GROQ_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        ollama_api_key = os.getenv("OLLAMA_API_KEY", "")

        if not tavily_api_key:
            st.error("TAVILY_API_KEY Not Found")
            st.error("Please configure TAVILY_API_KEY in Streamlit secrets or environment variables")
            st.stop()

        if use_ollama and not ollama_api_key:
            st.error("OLLAMA_API_KEY Not Found")
            st.error("Please configure OLLAMA_API_KEY in your .env file or Streamlit secrets")
            st.stop()

        if not use_ollama and not groq_api_key:
            st.error("GROQ_API_KEY Not Found")
            st.error("Please configure GROQ_API_KEY in your .env file or Streamlit secrets")
            st.stop()

        st.success("API Keys Loaded")

    st.divider()

    # Control buttons
    st.subheader("Controls")
    if st.button("Reset Conversation", use_container_width=True):
        if 'scholarship_bot' in st.session_state:
            st.session_state.scholarship_bot.reset_conversation()
        st.session_state.messages = []
        st.rerun()

    if st.button("View My Profile", use_container_width=True):
        if 'scholarship_bot' in st.session_state:
            profile = st.session_state.scholarship_bot.user_profile
            with st.expander("Your Profile Information", expanded=True):
                st.write(f"**Field of Study:** {profile.field_of_study or 'Not set'}")
                st.write(f"**Education Level:** {profile.education_level or 'Not set'}")
                st.write(f"**Location:** {profile.location or 'Not set'}")
                st.write(f"**Citizenship:** {profile.citizenship or 'Not set'}")
                st.write(f"**GPA:** {profile.gpa if (isinstance(profile.gpa, (int, float)) and profile.gpa > 0) or (isinstance(profile.gpa, str) and profile.gpa.strip()) else 'Not set'}")
                st.write(f"**Financial Need:** {profile.financial_need or 'Not set'}")
                st.write(f"**Research Interests:** {', '.join(profile.research_interests) if profile.research_interests else 'Not set'}")
                st.write(f"**Career Goals:** {profile.career_goals or 'Not set'}")
        else:
            st.info("No profile data yet. Start chatting to build your profile!")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize or update bot when model changes
if 'scholarship_bot' not in st.session_state or st.session_state.get('current_model') != selected_model:
    try:
        st.session_state.scholarship_bot = ScholarshipBot(
            groq_api_key=groq_api_key if not use_ollama else None,
            ollama_api_key=ollama_api_key if use_ollama else None,
            tavily_api_key=tavily_api_key,
            model_name=selected_model,
            use_ollama=use_ollama
        )
        st.session_state.current_model = selected_model
    except Exception as e:
        st.error(f"Failed to initialize Scholarship Bot: {str(e)}")
        st.stop()

# Get bot instance
bot = st.session_state.scholarship_bot

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize with welcome message
if len(st.session_state.messages) == 0:
    welcome_msg = """Hello! I'm here to help you find scholarships that match your profile.

To get started, please tell me about your academic background. You can share multiple details in one message (field of study, education level, location, citizenship, GPA, etc.) or we can go step by step.

What would you like to share about yourself?"""
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
        with st.spinner("Thinking..."):
            try:
                # Process the message through the scholarship bot
                response = bot.process_message(prompt)

                # Display response
                st.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}\n\nPlease try again or rephrase your question."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(" **Tip:** Use the sidebar to switch between different AI models and view your profile.")
