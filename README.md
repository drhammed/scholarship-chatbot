# 🎓 Scholarship Chatbot

An intelligent, multi-agent scholarship guidance system that helps students discover and apply for scholarships tailored to their profiles. Built with Streamlit and powered by LangChain, this application uses a sophisticated three-agent architecture to provide personalized scholarship recommendations.

## ✨ Features

- **🤖 Multi-Agent Architecture**: Three specialized AI agents work together to provide comprehensive scholarship guidance
- **🔄 Model Flexibility**: Support for both Groq and Ollama Cloud models
- **🌐 Live Web Search**: Real-time scholarship search using Tavily API
- **💬 Conversational Interface**: Natural, friendly chat experience
- **📊 Profile Management**: Automatic user profile building and tracking
- **🎯 Personalized Results**: Citizenship-aware scholarship matching
- **📝 Context-Aware**: Maintains conversation history for relevant responses

## 🏗️ Architecture

This chatbot uses a **multi-agent architecture** with three specialized agents that work seamlessly behind the scenes:

### Agent 1: Profile Agent 🔍
**Role**: Gathers comprehensive user information through natural conversation

**Responsibilities**:
- Collects required information: field of study, education level, location, citizenship
- Extracts optional details: GPA, financial need, research interests, career goals
- Uses conversational prompts to make profile building feel natural
- Validates profile completeness before proceeding

**Why it matters**: Citizenship and location are critical for scholarship eligibility. This agent ensures we have accurate information before searching.

### Agent 2: Research Agent 🔎
**Role**: Performs intelligent, multi-query web searches for scholarships

**Responsibilities**:
- Constructs citizenship-specific search queries
- Performs location-based scholarship searches
- Searches for application tips and guidance
- Uses Tavily's advanced search with source attribution
- Combines results from multiple search strategies

**Why it matters**: Scholarships have complex eligibility requirements. Multiple search strategies ensure we find opportunities the user can actually apply for.

### Agent 3: Response Agent 💬
**Role**: Synthesizes search results into actionable guidance

**Responsibilities**:
- Filters scholarships by citizenship eligibility
- Provides structured recommendations with clear sections
- Includes source URLs for verification
- Offers application guidance and next steps
- Can provide detailed application support on request

**Why it matters**: Raw search results aren't enough. This agent transforms data into a clear action plan with verified sources.

## 🔄 Conversation Flow

```
User Arrives → Profile Agent (collects info) → Research Agent (searches web)
→ Response Agent (provides recommendations) → User Follow-up Questions (handled contextually)
```

The user never sees these transitions - they just experience a smooth, helpful conversation!

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- API Keys:
  - **Groq API Key** (for Groq models) - Get one [here](https://console.groq.com/keys)
  - **Ollama API Key** (for Ollama Cloud models) - Get one [here](https://ollama.com)
  - **Tavily API Key** (for web search) - Get one [here](https://tavily.com)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/scholarship-chatbot.git
cd scholarship-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OLLAMA_API_KEY=your_ollama_api_key_here
```

4. Run the application:
```bash
streamlit run main.py
```

## 📖 Usage

1. **Select Your Model**: Choose between Groq or Ollama Cloud in the sidebar
2. **Start Chatting**: Answer questions about your academic background
3. **Get Recommendations**: Receive personalized scholarship matches with sources
4. **Ask Follow-ups**: Request detailed application support or ask specific questions
5. **View Profile**: Check your saved profile information anytime
6. **Reset if Needed**: Start fresh with a new conversation

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI Framework**: LangChain
- **LLM Providers**:
  - Groq (llama-3, compound-mini, etc.)
  - Ollama Cloud (gpt-oss models)
- **Search Engine**: Tavily API
- **Memory**: ConversationBufferWindowMemory (10-message window)
- **State Management**: Python Enums and Dataclasses

## 📁 Project Structure

```
scholarship-chatbot/
├── main.py                 # Streamlit UI and main application logic
├── scholarship_bot.py      # Multi-agent bot implementation
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not in repo)
└── README.md              # This file
```

## 🔑 Key Design Decisions

1. **Multi-Agent vs Single Agent**: We use specialized agents because:
   - Each has a focused responsibility
   - Easier to debug and improve individual components
   - More predictable behavior at each conversation stage

2. **Citizenship-First Approach**: Many scholarship searches fail because they don't consider citizenship eligibility upfront. Our Research Agent specifically searches for citizenship-eligible scholarships.

3. **Source Attribution**: Every recommendation includes source URLs, allowing users to verify information and apply directly.

4. **Context Window**: 10-message buffer balances context retention with token efficiency.

5. **Model Flexibility**: Support for multiple providers ensures users can choose based on their preferences and API availability.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Search powered by [Tavily](https://tavily.com)
- LLM inference by [Groq](https://groq.com) and [Ollama](https://ollama.com)
- UI framework by [Streamlit](https://streamlit.io)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This chatbot provides scholarship information based on web searches. Always verify eligibility requirements and deadlines on official scholarship websites before applying.
