import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchResults
class State(TypedDict):
    keyword:str
    news_article:str
    summary:str

# Create a sidebar for the navbar
st.sidebar.title("Navigation")

# API Key Input
api_key = st.sidebar.text_input("Enter your API Key", type="password")
no = st.sidebar.text_input("No. of News Article", type="default")

# First Layer Dropdown
menu_options = ["openai", "anthropic", "google","groq"]
selected_menu = st.sidebar.selectbox("Main Menu", menu_options)

# Second Layer Dropdown based on the first selection
sub_menu_options = {
    "openai": ["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-4","gpt-3.5-turbo"],
    "google": ['gemini-1.5-flash','gemini-1.5-pro','gemini-1.0-pro'],
    "anthropic": ["claude-3-5-sonnet-20241022","claude-3-5-haiku-20241022","claude-3-opus-20240229","claude-3-sonnet-20240229","claude-3-haiku-20240307"],
    "groq":[ "distil-whisper-large-v3-en",
    "gemma2-9b-it",
    "gemma-7b-it",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "whisper-large-v3",
    "whisper-large-v3-turbo"]
}
selected_sub_menu = st.sidebar.selectbox(
    "Sub Menu",
    sub_menu_options[selected_menu]
)

# Store values in a variable for further use
navbar_config = {
    "api_key": api_key,
    "company": selected_menu,
    "model": selected_sub_menu
}



def news(state):
    search = DuckDuckGoSearchResults(num_results=no, source="news" ,output_format="list")
    li = search.invoke(state['keyword'])
    links = [i['link'] for i in li]
    loader = WebBaseLoader(links)
    docs = loader.load()
    news =""
    for doc in docs:
        news+=f"Title : {doc.metadata['title']}\n Article : {doc.page_content}"
    return {"news_article":news}


def LLM(api_key,company,model):
    if company == 'openai':
        return ChatOpenAI(api_key=api_key,
    model=model,
    temperature=0,
    max_tokens=None,
    timeout=None,

)
    if company == 'google':
        return ChatGoogleGenerativeAI(api_key=api_key,
    model=model,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
    if company == 'anthropic':
        return ChatAnthropic(api_key=api_key,
    model=model,
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)
    if company == 'groq':
        return ChatGroq(api_key=api_key,
    model=model,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
    

def chatbot(state):
    llm= LLM(navbar_config["api_key"],navbar_config["company"],navbar_config["model"])
    prompt = f"""You are an expert news summarizer combining AP and Reuters style clarity with digital-age brevity.

    Your task:
    1. Core Information:
       - Lead with the most newsworthy development
       - Include key stakeholders and their actions
       - Add critical numbers/data if relevant
       - Explain why this matters now
       - Mention immediate implications

    2. Style Guidelines:
       - Use strong, active verbs
       - Be specific, not general
       - Maintain journalistic objectivity
       - Make every word count
       - Explain technical terms if necessary

    Format: Create a single paragraph of 250-400 words that informs and engages.
    Pattern: [Major News] + [Key Details/Data] + [Why It Matters/What's Next]

    Focus on answering: What happened? Why is it significant? What's the impact?

    IMPORTANT: Provide ONLY the summary paragraph. Do not include any introductory phrases, 
    labels, or meta-text like "Here's a summary" or "In AP/Reuters style."
    Start directly with the news content.\nHere are the news article {state['news_article']}"""
    return {"summary":llm.invoke(prompt)}
def GraphBuilder():

    graph_builder =StateGraph(State)
    graph_builder.add_node("news", news)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "news")
    graph_builder.add_edge("news","chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()
topic = st.text_input("Enter news topic:", value="artificial intelligence")
if st.button("Process News", type="primary"):
    if topic and navbar_config and no:
        try:
            graph = GraphBuilder()
            final_summary = graph.invoke({"keyword":topic})
            st.header(f"📝 News Summary: {topic}")
            st.markdown(final_summary["summary"].content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a topic!")