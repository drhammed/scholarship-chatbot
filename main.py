import streamlit as st
#from langchain_voyageai import VoyageAIEmbeddings
import os
#import json
#import boto3
from dotenv import load_dotenv
#from urllib.parse import urlparse
#from pinecone import Pinecone
#import pinecone
from langchain_openai import ChatOpenAI
import openai
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
#import time
#import re
import warnings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.prompts import SystemMessagePromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import uuid
from datetime import datetime, timedelta

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set up Streamlit app
st.set_page_config(page_title="Scholarship Chatbot by drhammed", layout="wide")
st.title("Scholarship Chatbot by drhammed")
st.write("Hello! I'm your friendly chatbot. I'm here to help answer your questions regarding scholarships and funding for students, and provide information. I'm also super fast! Let's start!")

# Load environment variables from .env file
load_dotenv()

#For Streamlit & AWS
#OpenAI API key
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
#Groq API KEY
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]

#For Heroku & Local deployment
#OPENAI_API_KEY = os.getenv("My_OpenAI_API_key")
#GROQ_API_KEY = os.getenv("My_Groq_API_key")

# Model selection
model_options = ["llama3-70b-8192", "llama3-8b-8192","llama-3.2-1b-preview", "llama-3.2-3b-preview"]
selected_model = st.sidebar.selectbox("Select a model", model_options)

# Initialize selected model

def get_model(selected_model):
        if selected_model == "llama3-8b-8192":
            return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
        elif selected_model == "llama3-70b-8192":
            return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
        elif selected_model == "llama-3.2-1b-preview":
            return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.2-1b-preview", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
        elif selected_model == "llama-3.2-3b-preview":
            return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.2-3b-preview", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
        else:
            raise ValueError("Invalid model selected")

llm_mod = get_model(selected_model)


system_prompt = """
<rules>
  Scholarship_PROMPT1: Follow the instructions below to provide scholarship and funding information for users effectively and efficiently.

  1. Adhere to the defined user needs and context.
  2. Avoid repeating questions if the user has already provided the necessary information.
  3. Start each interaction by understanding the user's profile if not already provided.
  4. Summarize and confirm data before proceeding to the next steps.
  5. Offer detailed guidance on scholarships and application processes, including tips for personal statements and deadlines.
  6. Focus on continuous conversation, ensuring that the user's needs are fully addressed before ending the chat.
  7. Handle off-topic questions as follows:
     - If the user asks a question that is not related to scholarships, funding, fellowships, and other academic discussions (except greetings and compliments for you), respond with:
       "Sorry, but I'm here to assist you with scholarship, funding, and related information. If you have any questions related to these topics, please feel free to ask!"
  8. Maintain context awareness throughout the conversation to avoid redundant inquiries.

</rules>

<answer_operator>
  <prompt_metadata>
    Type: Scholarship Information Provider
    Purpose: Offer Comprehensive Scholarship Guidance
    Paradigm: Context-Aware Assistance
    Constraints: Relevance, Efficiency, Continuity
    Objective: User-Centered Support
  </prompt_metadata>
  
  <core>
    - Inquiry: Collect relevant information about the user's academic background, field of study, and educational level.
    - Confirmation: Summarize and confirm the collected data before proceeding to provide scholarship details.
    - Information: Identify and present scholarships that match the user's profile, including eligibility criteria, deadlines, and application processes.
    - Guidance: Offer detailed advice on how to apply for scholarships, including tips for writing personal statements and preparing for interviews.
    - Continuity: Maintain a continuous, context-aware conversation, ensuring the user’s needs are met without redundant inquiries.
    - Action Confirmation: Confirm each step with the user before proceeding to the next phase of guidance or application support.
    - Completion: Provide a clear summary of the identified scholarships and the next steps for the user to follow.
  
  </core>
  
  <think>
    Consider the user's needs and context at each step to ensure relevance and efficiency.
  </think>
  
  <expand>
    Explore additional funding opportunities and provide comprehensive support throughout the application process.
  </expand>
  
  <loop>
    while(true) {{
      gather_user_information();
      confirm_data_with_user();
      if(user_confirms_data()) {{ 
        present_scholarship_options();
        provide_detailed_guidance();
      }}
      break_if_user_satisfied();
    }}
  </loop>
  
  <verify>
    Ensure that each step is context-aware, avoids redundancy, and meets the user's needs.
  </verify>
  
  <mission>
    Provide efficient, context-aware scholarship and funding information, guiding users through the entire process without redundant questions or unnecessary repetition.
  </mission>
</answer_operator>

Scholarship_PROMPT2:
What did you do?
Did you use the <answer_operator>? Y/N
Answer the above question with Y or N at each output.

"""



# Initialize the conversation memory
#conversational_memory_length = 100
#memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Create a conversation chain
conversation = ConversationChain(
    llm=llm_mod,
    memory=memory,
    prompt=ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}")
    ])
)


# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'sessions' not in st.session_state:
    st.session_state['sessions'] = {}

if 'current_session_name' not in st.session_state:
    st.session_state['current_session_name'] = None

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

if 'conversation_state' not in st.session_state:
    st.session_state['conversation_state'] = "start"

# Function to get today's date in a readable format
def get_readable_date():
    return datetime.now().strftime("%Y-%m-%d")

# Function to generate a summary of the user's first query
def generate_summary(user_input):
    summary_prompt = f"Summarize this query in a few words: {user_input}"
    summary_response = llm_mod.predict(summary_prompt)
    return summary_response.strip()

# Function to generate a unique session name based on the summary of the user's first query
def generate_session_name(user_input):
    summary = generate_summary(user_input)
    return summary

# Function to save the current session
def save_current_session():
    if st.session_state['current_session_name'] and len(st.session_state['messages']) > 1:
        st.session_state['sessions'][st.session_state['current_session_name']] = {
            'date': get_readable_date(),
            'messages': st.session_state['messages'].copy()
        }

# Function to display chat sessions in the sidebar
def display_chat_sessions():
    st.sidebar.header("Chat History")
    today = get_readable_date()
    yesterday = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    sessions = sorted(st.session_state['sessions'].items(), key=lambda x: x[1]['date'], reverse=True)
    
    current_day = ""
    for session_name, session_info in sessions:
        session_day = session_info['date']
        if session_day != current_day:
            if session_day == today:
                st.sidebar.subheader("Today")
            elif session_day == yesterday:
                st.sidebar.subheader("Yesterday")
            else:
                st.sidebar.subheader(session_day)
            current_day = session_day
        if st.sidebar.button(session_name):
            st.session_state['messages'] = session_info['messages']

# Display saved chat sessions in the sidebar
display_chat_sessions()

# Display chat messages from history.
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

def clear_input():
    st.session_state.user_input = ''

if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

user_question = st.chat_input("You: ")

if user_question:
    st.session_state.user_input = user_question

    # Set session name based on the summary of the first user input
    if st.session_state.current_session_name is None:
        st.session_state.current_session_name = generate_session_name(st.session_state.user_input)
    
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": st.session_state.user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(st.session_state.user_input)
        
    if st.session_state.conversation_state == "start":
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ])

    elif st.session_state.conversation_state == "awaiting_confirmation":
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
            HumanMessagePromptTemplate.from_template("The user has confirmed the scholarships. Proceed with application guidance."),
        ])

    with st.spinner("Thinking..."):
        try:
            response = conversation.predict(human_input=st.session_state.user_input)
            if "Do you confirm the above data?" in response:
                st.session_state.conversation_state = "awaiting_confirmation"
            elif "Proceeding with detailed guidance" in response:
                st.session_state.conversation_state = "providing_guidance"
            else:
                st.session_state.conversation_state = "start"
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            response = "Sorry, I'm having trouble processing your request right now. Please try again later."

    # Add bot response to chat history
    st.session_state['messages'].append({"role": "assistant", "content": response})
    
    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    clear_input() # Clear the input field
    
    # Save the current session automatically
    save_current_session()
