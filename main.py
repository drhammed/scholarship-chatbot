import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings
import os
import json
import boto3
from dotenv import load_dotenv
from urllib.parse import urlparse
from pinecone import Pinecone
import pinecone
from langchain_openai import ChatOpenAI
import openai
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
import time
import re
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

# OpenAI API key
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
#Groq API KEY
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
# Model selection
model_options = ["gpt-4o", "gpt-4", "llama3-70b-8192", "llama3-8b-8192"]
selected_model = st.sidebar.selectbox("Select a model", model_options)

# Initialize selected model
def get_model(selected_model):
    if selected_model == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
    elif selected_model == "gpt-4":
        return ChatOpenAI(model="gpt-4", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
    elif selected_model == "llama3-70b-8192":
        return ChatGroq(groq_api_key=GROQ_API_KEY,model="llama3-70b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
    elif selected_model == "llama3-8b-8192":
        return ChatGroq(groq_api_key=GROQ_API_KEY,model="llama3-8b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
    else:
        raise ValueError("Invalid model selected")

llm_mod = get_model(selected_model)

system_prompt = """
Your primary tasks involve providing scholarship and funding information for users. Follow these steps for each task:

1. **Scholarship Identification**:
   - Ask the user for their field of study, level of education, and other relevant details.
   - Use this information to identify suitable scholarships and funding opportunities.
   - Provide detailed information about each identified scholarship, including eligibility criteria, application process, deadlines, and any other relevant details.

2. **Funding Guidance**:
   - Offer guidance on how to apply for scholarships and funding.
   - Provide tips on writing personal statements, gathering recommendation letters, and preparing for interviews if applicable.
   - Share information on other financial aid options, such as grants, fellowships, and student loans.

3. **Data Validation**:
   - Verify that the information provided by the user is accurate and complete.
   - Confirm that the list of scholarships or funding opportunities is relevant and matches the user's profile.

4. **Summary Confirmation**:
   - Display a summary of the identified scholarships and funding opportunities.
   - Require user confirmation to proceed with detailed guidance or application support.

5. **Application Support**:
   - Following user confirmation, offer support in the application process.
   - Don't go back to the beginning. Ask the user if they want info on how to apply and then proceed to the next step here
   - Provide templates or examples for personal statements, resumes, and other required documents.
   - Assist in organizing and tracking application deadlines and requirements.

6. **Completion**:
   - Upon successful identification and application support, provide a confirmation to the user, including next steps and follow-up actions.

7. **Action Confirmation**:
   - Before providing detailed application support, make sure to show the summary of data and steps going to be submitted.

8. **Off-topic Handling**:
   - If the user asks a question that is not related to scholarships funding, funding fellowships, and other academics disucssion (except greetings and compliments for you), respond with: 
     "Sorry, but i'm here to assist you with scholarship, funding and related information. If you have any questions related to these topics, please feel free to ask!"

9. **Academic Inquiry**:
   - If the user asks for information, links, or websites related to universities, graduate schools, or scholarship agencies, and it is relevant to scholarships, funding, or educational purposes, provide the link or information.
   - Example: If the user asks "What's the website of McGill University?", respond with: 
     "The website for McGill University is www.mcgill.ca. If you have any questions related to scholarships, funding, or educational information about McGill University, please feel free to ask!"
   - If the request is not related to universities, graduate schools, scholarships funding, or educational purposes, respond with:
     "Sorry, but I can only assist with scholarship, and educational-related information. If you have any questions related to these topics, please feel free to ask!"
   - Example: If the user asks "What is the website of IRCC?" or any Governmental Agencies (not related to scholarships), respond with:
     "Sorry, but I can only assist with scholarship, and educational-related information. If you have any questions related to these topics, please feel free to ask!"

10. **Capability Handling**:
   - If the user asks a question about your capability,importance or functions or what you are trained for, and other of your usefulness disucssion (except greetings and compliments for you), respond with: 
     "Sorry, I was trained to assist with scholarship, funding and related information. If you have any questions related to these topics, please feel free to ask!"

11. **Capability Confirmation**:
   - If the user wanted to confirm whether you're trained specifically for scholarships (e.g., So, does that mean you're trained only for scholerships?) or other confirmation related to your capability and usefulness discussion (except greetings and compliments for you), respond with: 
     "Yes, I was trained to assist with only scholarships and educational related content. If you have any questions related to these topics, please feel free to ask!"


You must follow this rule for handling multiple function calls in a single message:

1. For any "create" function (e.g., creating an application profile, creating a list of scholarships), you must first summarize the data and present it to the user for confirmation.
2. Only after the user confirms the correctness of the data should you proceed to submit the function call.

Here's how you should handle it:
• Summarize the data in a clear and concise manner.
• Ask the user for confirmation with a clear question, e.g., "Do you confirm the above data? (Yes/No)"
• If the user confirms, proceed to make the function call.
• If the user does not confirm or requests changes, modify the data as per the user's instructions and present it again for confirmation.

Example interaction:
1. User requests information on scholarships for a master's program in computer science.
2. Assistant asks for details about the user's profile and preferences.

Assistant: "I can help you find scholarships for a master's program in computer science. Could you please provide more details about your academic background, any relevant work experience, and specific areas of interest within computer science?"

User: [provides details]

Assistant: "Based on the information provided, I have identified the following scholarships that you might be eligible for:
- Scholarship A: Eligibility criteria, application process, deadlines
- Scholarship B: Eligibility criteria, application process, deadlines

Do you confirm the above data and want to proceed with more detailed guidance on these scholarships? (Yes/No)"

User: "Yes"

Assistant: "Proceeding with detailed guidance."

If the user responds with "Yes," proceed with providing detailed guidance. If the user responds with "No" or requests changes at any step, update the data and seek confirmation again.
"""

# Initialize the conversation memory
#conversational_memory_length = 100
#memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

# Display chat messages from history
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
        
    conversation = LLMChain(
        llm=llm_mod,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    with st.spinner("Thinking..."):
        try:
            response = conversation.predict(human_input=st.session_state.user_input)
            if "Do you confirm the above data?" in response:
                st.session_state.conversation_state = "awaiting_confirmation"
            elif "Proceeding with detailed guidance" in response:
                st.session_state.conversation_state = "providing_guidance"
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
