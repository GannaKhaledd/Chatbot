from config import api_key, persist_directory
from data_loading import load_data
from doc_creator import create_documents
from vector_store import initialize_vector_store
from memory import get_memory
from tools import get_tools
from prompts import get_context_prompt
from agent import setup_agent
from ui import setup_chat_ui

from groq import Groq
from langchain_groq import ChatGroq
import panel as pn

# Initialize API
client = Groq(api_key=api_key)
llm_model = "llama3-groq-70b-8192-tool-use-preview"
llm = ChatGroq(temperature=0.2, model=llm_model)

# Load data
data = load_data()
documents = create_documents(data)

# Vector store and memory
vector_store = initialize_vector_store(documents)
memory = get_memory()

# Tools and prompts
tools = get_tools(vector_store)
context_prompt = get_context_prompt()

# Agent setup
agent = setup_agent(llm, tools, memory, context_prompt)

# Chat UI
context = []
chat_ui = setup_chat_ui(agent, context)

# Use Panel to serve the UI
pn.serve(chat_ui, port=5000)  
