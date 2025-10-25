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
    gpa: Any = ""  # Can be str, float, or empty
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
        gpa_display = 'Not specified'
        if isinstance(self.gpa, (int, float)) and self.gpa > 0:
            gpa_display = str(self.gpa)
        elif isinstance(self.gpa, str) and self.gpa.strip():
            gpa_display = self.gpa

        return f"""
        Field of Study: {self.field_of_study}
        Education Level: {self.education_level}
        Location: {self.location}
        Citizenship: {self.citizenship}
        GPA: {gpa_display}
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

        # Extract information from user input FIRST
        self._extract_profile_info(user_input)

        # Check if profile is complete after extraction
        if self.user_profile.is_complete():
            self.state = ConversationState.SEARCHING
            return "Great! I have enough information. Let me search for relevant scholarships for you..."

        # Check if this is just a greeting without profile info
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        user_input_lower = user_input.lower().strip()
        is_greeting = any(greeting in user_input_lower for greeting in greeting_words) and len(user_input.split()) < 10

        system_prompt = f"""
        You are a Profiler Agent for a scholarship guidance system using ReAct (Reasoning + Acting) methodology.

        **REASONING PROCESS (Chain of Thought)**:
        1. **OBSERVE**: Check what profile information the user has already provided
        2. **ANALYZE**: Identify which required fields are still missing
        3. **PRIORITIZE**: Determine the most critical missing field
        4. **ACT**: Ask for ONE specific piece of information OR confirm profile is complete

        Current user profile:
        {self.user_profile.to_search_context()}

        **REQUIRED FIELDS** (must collect all):
        - citizenship (HIGHEST priority - needed for eligibility)
        - field_of_study
        - education_level
        - location

        **OPTIONAL BUT HELPFUL**:
        - GPA
        - financial_need
        - research_interests
        - career_goals

        **RULES**:
        1. If user just greeted you (e.g., "Hi", "Hello") → Respond warmly and ask for their academic background
        2. If profile information was provided → Acknowledge it briefly and ask for ONE missing required field
        3. ONLY ask for ONE piece of information at a time
        4. Be conversational, friendly, and concise
        5. Prioritize missing fields: citizenship > field_of_study > education_level > location
        6. If you have all required information, respond with exactly: "PROFILE_COMPLETE"
        7. DO NOT ask for information already provided in the current profile

        **RESPONSE STYLE**:
        - Keep responses short and focused
        - If greeting: Welcome them and ask about their academic background
        - If partial info: "Thanks! I see you're studying [field]. Could you tell me..."
        - If complete: "PROFILE_COMPLETE"

        **EXAMPLES**:
        User: "Hi"
        You: "Hello! I'd be happy to help you find scholarships. To get started, could you tell me about your academic background and citizenship?"

        User: "I'm studying Computer Science"
        You: "Great! Computer Science is an excellent field. What is your citizenship/nationality? This is crucial for finding eligible scholarships."
        """

        if is_greeting:
            return "Hello! I'd be happy to help you find scholarships. To get started, could you tell me about your academic background (field of study, degree level) and citizenship?"

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

        # Check again after LLM response
        if "PROFILE_COMPLETE" in response:
            self.state = ConversationState.SEARCHING
            return "Great! I have enough information. Let me search for relevant scholarships for you..."

        return response

    def research_agent(self, query: str) -> Dict[str, Any]:
        """Agent 2: Performs live web search using Tavily"""

        from datetime import datetime
        current_year = datetime.now().year
        next_year = current_year + 1

        # Determine target location for studies (could be different from current location)
        study_location = self.user_profile.location
        if 'United States' in study_location or 'USA' in study_location or 'US' in study_location:
            location_keywords = "USA United States American universities"
        elif 'Canada' in study_location or 'Canadian' in study_location:
            location_keywords = "Canada Canadian universities"
        elif 'United Kingdom' in study_location or 'UK' in study_location:
            location_keywords = "UK United Kingdom British universities"
        else:
            location_keywords = f"{study_location} universities"

        # Construct citizenship-specific search queries with current year
        citizenship_query = f"""
        scholarships grants funding {self.user_profile.citizenship} citizens
        {self.user_profile.field_of_study} {self.user_profile.education_level}
        international students {current_year} {next_year} open deadline
        """

        # Location-specific opportunities with date relevance
        location_query = f"""
        scholarships {location_keywords}
        {self.user_profile.field_of_study} {self.user_profile.education_level}
        {self.user_profile.citizenship} international students
        {current_year} {next_year} application open deadline
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

            # Search for current open deadlines
            deadline_query = f"""
            {self.user_profile.field_of_study} scholarships {self.user_profile.citizenship}
            {location_keywords} deadline {current_year} {next_year} still open accepting applications
            """
            deadline_results = self.tavily_client.search(
                query=deadline_query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
                include_raw_content=False
            )

            # Combine results with source tracking
            all_results = {
                'citizenship_scholarships': citizenship_results,
                'location_scholarships': location_results,
                'current_deadlines': deadline_results,
                'user_profile': self.user_profile.to_search_context(),
                'search_date': datetime.now().strftime("%B %d, %Y"),
                'target_years': f"{current_year}-{next_year}"
            }

            self.search_results = all_results
            return all_results

        except Exception as e:
            print(f"Search error: {e}")
            return {"error": str(e)}

    def response_agent(self, search_data: Dict[str, Any]) -> str:
        """Agent 3: Synthesizes search results and provides structured response"""

        from datetime import datetime
        today_date = datetime.now().strftime("%B %d, %Y")

        system_prompt = f"""
        You are a Response Agent for a scholarship guidance system using ReAct (Reasoning + Acting) methodology.

        **CURRENT DATE**: {today_date}

        **REASONING PROCESS (Chain of Thought)**:
        1. **OBSERVE**: Review the user's profile data from the search context
        2. **ANALYZE**: Examine search results for citizenship-eligible scholarships
        3. **FILTER**: Remove scholarships with expired deadlines (before {today_date})
        4. **VERIFY**: Ensure each scholarship explicitly accepts the user's citizenship
        5. **SYNTHESIZE**: Create a personalized, actionable response
        6. **ACT**: Provide specific recommendations with sources

        **CRITICAL REQUIREMENTS**:
        1. **CITIZENSHIP ELIGIBILITY FIRST**: ONLY recommend scholarships that explicitly allow the user's citizenship/nationality
           - If user is Nigerian, DO NOT recommend US-only or country-specific scholarships unless Nigeria is explicitly mentioned
           - Look for: "international students", "Nigerian students", "African students", or specific country mentions
        2. **DATE VERIFICATION**: Today is {today_date} - ONLY include scholarships with future deadlines
        3. **SOURCE ATTRIBUTION**: Always include the source URL for EVERY scholarship: [Source: URL]
        4. **PROFILE AWARENESS**: Use the actual profile data provided - DO NOT ask for information already given
        5. **VERIFICATION NOTE**: Remind users to verify eligibility on official websites

        **RESPONSE STRUCTURE**:
        1. **Profile Acknowledgment** - Briefly confirm their background (citizenship, field, level)
        2. **Scholarships for [User's Citizenship] Citizens** (3-5 scholarships with future deadlines they can apply for)
           - Each must include: Name, Amount, Deadline, Eligibility, URL
        3. **Application Guidance** (specific tips for their profile and citizenship)
        4. **Next Steps** (concrete action items with deadlines)
        5. **Additional Resources** (relevant links with sources)

        **FORMATTING RULES**:
        - Use clear headers and bullet points
        - Include source URL for each scholarship: [Source: website.com]
        - Emphasize citizenship-specific eligibility criteria
        - Show deadlines in a clear format

        **CRITICAL - AVOID THESE MISTAKES**:
        - DO NOT ask for information already in the user profile (field of study, citizenship, education level, location)
        - DO NOT recommend scholarships limited to specific countries unless user is from that country
        - DO NOT include scholarships with expired deadlines
        - DO NOT provide generic advice - make it specific to their citizenship and field
        - DO NOT recommend scholarships without clear source attribution
        """

        search_context = json.dumps(search_data, indent=2)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template(
                "Based on this search data and user profile, provide comprehensive scholarship guidance:\n\n{search_context}\n\nREMEMBER: Use the profile information provided. DO NOT ask for details already given."
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

        system_prompt = """
        You are an Information Extraction Agent. Your task is to extract scholarship-relevant profile information from user messages.

        REASONING PROCESS (Chain of Thought):
        1. Read the user input carefully
        2. Identify any profile-related information mentioned
        3. Map the information to the appropriate profile fields
        4. Return ONLY valid JSON

        EXTRACTION RULES:
        - field_of_study: Academic major/field (e.g., "Computer Science", "Engineering", "Business")
        - education_level: Degree level (e.g., "Bachelor's", "BSc", "Master's", "PhD", "Undergraduate", "Graduate")
        - gpa: Academic performance (e.g., "4.57/5.00", "3.8/4.0", "First Class")
        - location: Current/intended study location (e.g., "Nigeria", "USA", "University of Lagos")
        - citizenship: Nationality/citizenship (e.g., "Nigerian", "American", "Indian")
        - financial_need: Funding requirements (e.g., "fully-funded", "partial funding", "tuition only")
        - research_interests: Array of research topics (e.g., ["Natural Language Processing", "Machine Learning"])
        - career_goals: Professional aspirations
        - extracurriculars: Array of activities

        EXAMPLES:
        Input: "I'm a Nigerian with a BSc in Computer Science, CGPA 4.57/5.00"
        Output: {"citizenship": "Nigerian", "field_of_study": "Computer Science", "education_level": "Bachelor's", "gpa": "4.57/5.00", "location": "Nigeria"}

        Input: "I want fully-funded scholarships for NLP and AI research"
        Output: {"financial_need": "fully-funded", "research_interests": ["Natural Language Processing", "Artificial Intelligence"]}

        Return ONLY a valid JSON object with extracted fields. If nothing found, return {}
        """

        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Current profile:\n{current_profile}\n\nUser input to extract from:\n{user_input}"
                )
            ])

            conversation = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=False
            )

            result = conversation.predict(
                current_profile=self.user_profile.to_search_context(),
                user_input=user_input
            )

            # Clean up the result to extract JSON
            result = result.strip()
            # Remove markdown code blocks if present
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            # Try to parse JSON and update profile
            try:
                extracted_data = json.loads(result)
                if extracted_data:  # Only process if we got data
                    for key, value in extracted_data.items():
                        if hasattr(self.user_profile, key) and value:
                            if isinstance(value, list):
                                current_list = getattr(self.user_profile, key) or []
                                updated_list = list(set(current_list + value))
                                setattr(self.user_profile, key, updated_list)
                            else:
                                setattr(self.user_profile, key, value)
                    print(f"[DEBUG] Extracted profile data: {extracted_data}")
            except json.JSONDecodeError as je:
                print(f"JSON parsing error: {je}\nReceived: {result}")

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
            if user_input_lower in ['yes', 'y', 'ok', 'sure', 'yeah', 'yep']:
                if self.pending_confirmation == "application_support":
                    self.pending_confirmation = None
                    return self._provide_application_support()
            elif user_input_lower in ['no', 'n', 'not now', 'nope', 'nah']:
                self.pending_confirmation = None
                return "No problem! Feel free to ask if you need help with anything else regarding scholarships."
            else:
                # User is asking a different question instead of answering yes/no
                # Clear pending confirmation and route to follow-up handler
                self.pending_confirmation = None
                # Don't return here, let it fall through to state handling

        # Route to appropriate agent based on state
        if self.state == ConversationState.PROFILING:
            profiler_response = self.profiler_agent(user_input)

            # If profiling is complete, automatically proceed with search
            if self.state == ConversationState.SEARCHING:
                try:
                    search_results = self.research_agent(user_input)
                    if isinstance(search_results, dict) and "error" in search_results:
                        return f"{profiler_response}\n\nHowever, I encountered an error while searching: {search_results['error']}. Please try again."

                    self.state = ConversationState.RESPONDING
                    scholarship_response = self.response_agent(search_results)
                    return f"{profiler_response}\n\n{scholarship_response}"

                except Exception as e:
                    return f"{profiler_response}\n\nHowever, I encountered an error while searching for scholarships: {str(e)}. Please try again."

            return profiler_response

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

        from datetime import datetime
        today_date = datetime.now().strftime("%B %d, %Y")

        user_input_lower = user_input.lower()

        # Check if user is asking for a new search with different criteria
        new_search_keywords = [
            'us ', 'usa', 'canada', 'uk', 'europe', 'australia',
            'master', 'phd', 'doctoral', 'undergraduate', 'graduate',
            'different field', 'change my', 'instead of', 'looking for'
        ]

        # Check if user is refining their search location/criteria
        is_refining_search = any(keyword in user_input_lower for keyword in new_search_keywords)

        if is_refining_search:
            # Update profile if they're specifying new preferences
            if 'us' in user_input_lower or 'usa' in user_input_lower or 'united states' in user_input_lower:
                self.user_profile.location = 'United States'
            elif 'canada' in user_input_lower or 'canadian' in user_input_lower:
                self.user_profile.location = 'Canada'
            elif 'uk' in user_input_lower or 'united kingdom' in user_input_lower or 'britain' in user_input_lower:
                self.user_profile.location = 'United Kingdom'

            # Perform a new search with updated criteria
            try:
                search_results = self.research_agent(user_input)
                if isinstance(search_results, dict) and "error" in search_results:
                    return f"I'll search for scholarships with your updated criteria.\n\nHowever, I encountered an error: {search_results['error']}. Please try again."

                # Generate new response with updated search results
                return self.response_agent(search_results)

            except Exception as e:
                return f"I encountered an error while searching for scholarships with your updated criteria: {str(e)}. Please try again."

        # Otherwise, answer based on existing search results
        system_prompt = f"""
        You are a Response Agent for a scholarship guidance system using ReAct methodology. The user has asked a follow-up question.

        **CURRENT DATE**: {today_date}

        **REASONING PROCESS**:
        1. **OBSERVE**: Read the user's follow-up question carefully
        2. **ANALYZE**: Check what information from search results is relevant
        3. **FILTER**: Remove expired scholarships (deadline before {today_date})
        4. **VERIFY**: Ensure recommendations match their current requirements
        5. **ACT**: Provide specific, actionable response

        User Profile:
        {self.user_profile.to_search_context()}

        Previous Search Results:
        {json.dumps(self.search_results, indent=2)}

        **CRITICAL INSTRUCTIONS**:
        1. **DATE AWARENESS**: Today is {today_date} - ONLY recommend scholarships with future deadlines
        2. **LOCATION SPECIFICITY**: If they ask for US/Canada/UK scholarships, ONLY show scholarships for those locations
        3. **SOURCE ATTRIBUTION**: Always include source URLs for any scholarships mentioned
        4. **SPECIFIC ANSWERS**: Answer based on the search results - don't give generic advice
        5. **ADMIT LIMITATIONS**: If search results don't have what they're asking for, offer to do a new search

        **RESPONSE FORMAT**:
        - Use clear headers and bullet points
        - Include: Scholarship Name, Deadline, Eligibility, Source URL
        - For expired scholarships: State "This deadline has passed"
        - If no relevant results: "I don't have scholarships matching [criteria] in my current search. Would you like me to search specifically for [criteria]?"

        **AVOID**:
        - Recommending scholarships with past deadlines as if they're still available
        - Generic information not backed by search results
        - Scholarships that don't match their location/field requirements
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

        return conversation.predict(human_input=user_input)

    def reset_conversation(self):
        """Reset the conversation state"""
        self.state = ConversationState.PROFILING
        self.user_profile = UserProfile()
        self.search_results = []
        self.pending_confirmation = None
        self.memory.clear()
