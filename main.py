#pip install langchain_voyageai
#!pip install langchain_openai
#!pip install langchain_pinecone
#pip install groq
#!pip install langchain_groq

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



def make_clickable_links(text):
    # Find all URLs in the text and convert them to clickable hyperlinks
    #url_pattern = re.compile(r'(https?://\S+)')
    #return url_pattern.sub(r'<a href="\1" target="_blank">\1</a>', text)
    url_pattern = re.compile(r'(https?://[^\s\)\]]+)')
    return url_pattern.sub(r'<a href="\1" target="_blank">\1</a>', text)

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # Set up Streamlit interface
    st.title("Scholarship Chatbot by drhammed")
    st.write("Hello! I'm your friendly chatbot. I'm here to help answer your questions regarding scholarships and funding for students, and provide information. I'm also super fast! Let's start!")

    # Get Groq API key from environment variable
    GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
    model = 'llama3-70b-8192'

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0.02)

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
   - Provide templates or examples for personal statements, resumes, and other required documents.
   - Assist in organizing and tracking application deadlines and requirements.

6. **Completion**:
   - Upon successful identification and application support, provide a confirmation to the user, including next steps and follow-up actions.

7. **Action Confirmation**:
   - Before providing detailed application support, make sure to show the summary of data and steps going to be submitted.

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

Please ensure this process is followed for all guidance and support calls.
"""
    conversational_memory_length = 5

    memory = ConversationBufferWindowMemory(k=conversational_memory_length,
                                            memory_key="chat_history",
                                            return_messages=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.write("### Chat History")
    for sender, message in st.session_state.chat_history:
        if sender == "User":
            st.markdown(f"<div style='color: blue;'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            message_with_links = make_clickable_links(message)
            st.markdown(f"<div style='color: green;'><strong>{sender}:</strong> {message_with_links}</div>", unsafe_allow_html=True)

    user_question = st.text_input("Ask a question:", key="user_input")

    if st.button("Send"):
        if user_question:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ])

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=False,
                memory=memory,
            )

            with st.spinner("Thinking..."):
                try:
                    response = conversation.predict(human_input=user_question)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response = "Sorry, I'm having trouble processing your request right now. Please try again later."
            st.session_state.chat_history.append(("User", user_question))
            st.session_state.chat_history.append(("Chatbot", response))
            st.experimental_rerun()
            st.rerun()

if __name__ == "__main__":
    main()