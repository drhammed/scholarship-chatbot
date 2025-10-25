import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from tavily import TavilyClient


class ConversationState(Enum):
    PROFILING = "profiling"
    SEARCHING = "searching"
    RESPONDING = "responding"
    COMPLETE = "complete"


@dataclass
class UserProfile:
    field_of_study: str = ""
    education_level: str = ""
    gpa: float = 0.0
    location: str = ""
    citizenship: str = ""
    financial_need: str = ""
    extracurriculars: Optional[List[str]] = None
    research_interests: Optional[List[str]] = None
    career_goals: str = ""
    
    def __post_init__(self):
        if self.extracurriculars is None:
            self.extracurriculars = []
        if self.research_interests is None:
            self.research_interests = []
    
    def is_complete(self) -> bool:
        required_fields = [self.field_of_study, self.education_level, self.location, self.citizenship]
        return all(field.strip() for field in required_fields)
    
    def to_search_context(self) -> str:
        return f"""
        Field of Study: {self.field_of_study}
        Education Level: {self.education_level}
        Location: {self.location}
        Citizenship: {self.citizenship}
        GPA: {self.gpa if self.gpa > 0 else 'Not specified'}
        Financial Need: {self.financial_need}
        Research Interests: {', '.join(self.research_interests) if self.research_interests else 'Not specified'}
        Career Goals: {self.career_goals}
        """


class ScholarshipBot:
    def __init__(self, tavily_api_key: str, model_name: str = 'llama3-70b-8192',
                 groq_api_key: Optional[str] = None, ollama_api_key: Optional[str] = None,
                 use_ollama: bool = False):
        # Initialize APIs
        self.tavily_api_key = tavily_api_key
        self.use_ollama = use_ollama

        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required")

        # Initialize LLM based on provider
        if use_ollama:
            # Ollama Cloud requires API key for authentication
            if not ollama_api_key:
                raise ValueError("OLLAMA_API_KEY is required when using Ollama Cloud")

            self.llm = ChatOllama(  # type: ignore
                model=model_name,
                temperature=0.02,
                base_url="https://ollama.com",
                client_kwargs={
                    "headers": {
                        "Authorization": f"Bearer {ollama_api_key}"
                    }
                }
            )
        else:
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is required when using Groq")
            self.llm = ChatGroq(  # type: ignore[call-arg]
                api_key=groq_api_key,
                model=model_name,
                temperature=0.02
            )

        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)

        # Initialize conversation state
        self.state = ConversationState.PROFILING
        self.user_profile = UserProfile()
        self.search_results = []
        self.pending_confirmation = None

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )

    def profiler_agent(self, user_input: str) -> str:
        """Agent 1: Profiles the user and extracts relevant information"""
        
        system_prompt = f"""
        You are a Profiler Agent for a scholarship guidance system. Your job is to gather complete user information.
        
        Current user profile:
        {self.user_profile.to_search_context()}
        
        IMPORTANT RULES:
        1. Ask ONE question at a time to gather missing information
        2. Extract and update profile information from user responses
        3. Required fields: field_of_study, education_level, location, citizenship
        4. Optional but helpful: GPA, financial_need, research_interests, career_goals, extracurriculars
        5. Be conversational and friendly
        6. When asking about citizenship, clarify: "What is your citizenship/nationality? This is crucial as scholarships have specific eligibility requirements based on citizenship."
        7. Once you have the required information, say "PROFILE_COMPLETE" to proceed to search
        
        Focus on gathering the most important missing information first. Emphasize that citizenship information is critical for finding eligible scholarships.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])
        
        conversation = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False,
            memory=self.memory
        )
        
        response = conversation.predict(human_input=user_input)
        
        # Extract information from user input
        self._extract_profile_info(user_input)
        
        if "PROFILE_COMPLETE" in response or self.user_profile.is_complete():
            self.state = ConversationState.SEARCHING
            return f"{response}\n\nGreat! I have enough information. Let me search for relevant scholarships for you..."
        
        return response

    def research_agent(self, query: str) -> Dict[str, Any]:
        """Agent 2: Performs live web search using Tavily"""
        
        # Construct citizenship-specific search queries
        citizenship_query = f"""
        scholarships grants funding for {self.user_profile.citizenship} citizens 
        {self.user_profile.field_of_study} {self.user_profile.education_level} 
        international students 2024 2025
        """
        
        # Also search for location-specific opportunities
        location_query = f"""
        scholarships {self.user_profile.location} university 
        {self.user_profile.field_of_study} {self.user_profile.education_level}
        {self.user_profile.citizenship} international students
        """
        
        try:
            # Primary search focused on citizenship eligibility
            citizenship_results = self.tavily_client.search(
                query=citizenship_query,
                search_depth="advanced",
                max_results=6,
                include_answer=True,
                include_raw_content=False
            )
            
            # Secondary search for institution-specific scholarships
            location_results = self.tavily_client.search(
                query=location_query,
                search_depth="advanced", 
                max_results=4,
                include_answer=True,
                include_raw_content=False
            )
            
            # Search for application tips
            tips_query = f"scholarship application tips {self.user_profile.field_of_study} personal statement {self.user_profile.citizenship}"
            tips_results = self.tavily_client.search(
                query=tips_query,
                search_depth="basic",
                max_results=3,
                include_answer=True
            )
            
            # Combine results with source tracking
            all_results = {
                'citizenship_scholarships': citizenship_results,
                'location_scholarships': location_results,
                'application_tips': tips_results,
                'user_profile': self.user_profile.to_search_context()
            }
            
            self.search_results = all_results
            return all_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return {"error": str(e)}

    def response_agent(self, search_data: Dict[str, Any]) -> str:
        """Agent 3: Synthesizes search results and provides structured response"""
        
        system_prompt = """
        You are a Response Agent for a scholarship guidance system. Your job is to synthesize search results 
        and provide comprehensive, actionable scholarship guidance.
        
        CRITICAL REQUIREMENTS:
        1. **CITIZENSHIP ELIGIBILITY FIRST**: Only recommend scholarships that explicitly allow the user's citizenship/nationality
        2. **SOURCE ATTRIBUTION**: Always include the source URL for each scholarship mentioned using format: [Source: URL]
        3. **VERIFICATION NOTE**: Always remind users to verify eligibility on the official website
        
        RESPONSE STRUCTURE:
        1. **🎯 Scholarships for [User's Citizenship] Citizens** (3-5 scholarships they can actually apply for)
        2. **📋 Application Guidance** (specific tips for their profile and citizenship)
        3. **⏰ Next Steps** (concrete action items with deadlines)
        4. **🔗 Additional Resources** (relevant links with sources)
        
        FORMATTING RULES:
        - Each scholarship must include: Name, Amount (if available), Deadline, Eligibility, Application process
        - Include source URL for each scholarship: [Source: website.com]
        - Use clear headers and bullet points
        - Emphasize citizenship-specific eligibility criteria
        - Always ask for confirmation before proceeding with detailed application support
        
        AVOID:
        - Recommending scholarships limited to US citizens unless user is from US
        - Generic advice without considering user's specific citizenship
        - Scholarships without clear source attribution
        """
        
        search_context = json.dumps(search_data, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template(
                "Based on this search data and user profile, provide comprehensive scholarship guidance:\n\n{search_context}"
            )
        ])
        
        conversation = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False
        )
        
        response = conversation.predict(search_context=search_context)
        
        # Set up confirmation for next steps
        self.pending_confirmation = "application_support"
        
        return response + "\n\n" + "Would you like me to provide detailed application support for any of these scholarships? (Yes/No)"

    def _extract_profile_info(self, user_input: str):
        """Extract profile information from user input using LLM"""
        
        extraction_prompt = f"""
        Extract profile information from this user input: "{user_input}"
        
        Current profile: {self.user_profile.to_search_context()}
        
        Return ONLY a JSON object with any new information found. Use these exact keys:
        - field_of_study
        - education_level  
        - gpa
        - location
        - citizenship
        - financial_need
        - research_interests (array)
        - career_goals
        - extracurriculars (array)
        
        If no new information is found, return empty JSON {{}}.
        """
        
        try:
            conversation = LLMChain(
                llm=self.llm,
                prompt=ChatPromptTemplate.from_messages([
                    SystemMessage(content=extraction_prompt)
                ]),
                verbose=False
            )

            result = conversation.predict(input="extract")
            
            # Try to parse JSON and update profile
            try:
                extracted_data = json.loads(result.strip())
                for key, value in extracted_data.items():
                    if hasattr(self.user_profile, key) and value:
                        if isinstance(value, list):
                            current_list = getattr(self.user_profile, key) or []
                            updated_list = list(set(current_list + value))
                            setattr(self.user_profile, key, updated_list)
                        else:
                            setattr(self.user_profile, key, value)
            except json.JSONDecodeError:
                pass  # Continue without extraction if JSON parsing fails
                
        except Exception as e:
            print(f"Profile extraction error: {e}")

    def _provide_application_support(self) -> str:
        """Provides detailed application support"""
        
        support_prompt = f"""
        Provide detailed application support for scholarships based on this user profile:
        {self.user_profile.to_search_context()}
        
        And these search results:
        {json.dumps(self.search_results, indent=2)}
        
        Provide:
        1. **Application Timeline** - When to start, key deadlines
        2. **Document Preparation** - What documents are needed
        3. **Personal Statement Tips** - Specific advice for their field/citizenship
        4. **Recommendation Letters** - Who to ask and what to provide them
        5. **Interview Preparation** - If applicable
        6. **Common Mistakes to Avoid** - Specific to their profile
        7. **Follow-up Actions** - What to do after applying
        
        Make it actionable and specific to their situation.
        """
        
        conversation = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content=support_prompt)
            ]),
            verbose=False
        )

        response = conversation.predict(input="provide_support")
        self.state = ConversationState.COMPLETE
        
        return response

    def process_message(self, user_input: str) -> str:
        """Main message processing orchestrator"""
        
        user_input_lower = user_input.strip().lower()
        
        # Handle confirmations
        if self.pending_confirmation:
            if user_input_lower in ['yes', 'y', 'ok', 'sure']:
                if self.pending_confirmation == "application_support":
                    self.pending_confirmation = None
                    return self._provide_application_support()
            elif user_input_lower in ['no', 'n', 'not now']:
                self.pending_confirmation = None
                return "No problem! Feel free to ask if you need help with anything else regarding scholarships."
        
        # Route to appropriate agent based on state
        if self.state == ConversationState.PROFILING:
            return self.profiler_agent(user_input)
        
        elif self.state == ConversationState.SEARCHING:
            try:
                search_results = self.research_agent(user_input)
                if isinstance(search_results, dict) and "error" in search_results:
                    return f"I encountered an error while searching: {search_results['error']}. Please try again."
                
                self.state = ConversationState.RESPONDING
                return self.response_agent(search_results)
                
            except Exception as e:
                return f"I encountered an error while searching for scholarships: {str(e)}. Please try again."
        
        elif self.state == ConversationState.RESPONDING:
            # Handle additional questions or requests
            return self._handle_followup_questions(user_input)
        
        elif self.state == ConversationState.COMPLETE:
            return self._handle_followup_questions(user_input)
        
        return "I'm here to help you find scholarships! Let's start by getting to know your academic background."

    def _handle_followup_questions(self, user_input: str) -> str:
        """Handle follow-up questions after initial scholarship search"""
        
        followup_prompt = f"""
        The user has asked a follow-up question about scholarships: "{user_input}"
        
        User Profile:
        {self.user_profile.to_search_context()}
        
        Previous Search Results:
        {json.dumps(self.search_results, indent=2)}
        
        Provide a helpful response. If they're asking for more specific information about a scholarship,
        application process, or need additional guidance, provide detailed assistance.
        If they want to search for different scholarships, indicate that you can help with that too.
        """
        
        conversation = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content=followup_prompt)
            ]),
            verbose=False
        )

        return conversation.predict(input="followup")

    def reset_conversation(self):
        """Reset the conversation state"""
        self.state = ConversationState.PROFILING
        self.user_profile = UserProfile()
        self.search_results = []
        self.pending_confirmation = None
        self.memory.clear()
