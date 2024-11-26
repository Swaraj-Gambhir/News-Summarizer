import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from typing_extensions import TypedDict
from typing import List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel, Field
import requests
import time
def check_robot(link):
    url = link+"/robots.txt"
    response = requests.get(url)

    if response.status_code == 200:
        # print("robots.txt found!",link)
        return False
            # Prints the content of robots.txt
    else:
        # print("robots.txt not found!")
        return True
class State(TypedDict):
    keyword:list
    used_keywords:list
    news:list
    summary:list
    no_of_iterations: int
    next_node: str
    purpose: str
    links:list

class KeyWord(BaseModel):
    """Keywords that needs to be made fro query"""
    keyword:List[str] = Field(description="Keywords to research more about any given news. Give at max 3 keywords to research more")

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
    print("Chatbot")
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
    Start directly with the news content.\nHere are the news article {str(state['news'])}. Try to stick to the purpose which is as follow {state["purpose"]}"""
    nxt = "re"
    if len(state["summary"]) > int(state["no_of_iterations"] )-1:
        nxt = "end"
   
    li = state["summary"]
    li.append(llm.invoke(prompt).content)
    return {"summary":li,"next_node":nxt}
def keyword_generator(state):
    print("Keyword")
    llm= LLM(navbar_config["api_key"],navbar_config["company"],navbar_config["model"])
    structured_llm = llm.with_structured_output(KeyWord)
    prompt= f"As a news agent your task is to do deep dive research. In this specific task you need to make keywords taking this entire news report as a refer such that more insights can be derived for the {state["used_keywords"][0]}. THe purpose of task is as follow {state["purpose"]}. Help make more relevant search queries for news search engine based on the information at present, Remeber to not move away from purpose. - News {state["summary"][-1]}"
    return {"keyword":structured_llm.invoke(prompt).keyword}

def news_node(state):
    print("News")
    li2=[]
    news =[]
    links2=[]
    search = DuckDuckGoSearchResults(num_results=10, source="news" ,output_format="list")
    time.sleep(10)
    for keyword in state["keyword"]:
        li = search.invoke(keyword)
        li2.append(keyword)
        links = [i['link'] for i in li]
        
        for link in links:
            if check_robot(link) and len(links2) <=int(no) :
                links2.append(link)     
            if len(links2) > int(no):
                break
        if len(links2)>0:
            loader = WebBaseLoader(links2)
            docs = loader.load()
            for doc in docs:
                news.append({"content":f"Title : {doc.metadata['title']}\n Article : {doc.page_content}","keyword":keyword})
    return {"news":news,"keyword":[],"used_keywords":state["used_keywords"]+li2,"links":state["links"]+links2}
def route_edges(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    next_node = state["next_node"]
    if next_node == "re" :
        return "re"
    
    return "end"
def GraphBuilder():
    graph_builder =StateGraph(State)
    graph_builder.add_node("news_node", news_node)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("keyword_node",keyword_generator)
    graph_builder.add_edge(START, "news_node")
    graph_builder.add_edge("news_node","chatbot")
    graph_builder.add_conditional_edges("chatbot",route_edges,{"re": "keyword_node", "end": END})
    graph_builder.add_edge("keyword_node","news_node")
    return graph_builder.compile()
topic = st.text_input("Enter news topic:", value="artificial intelligence")
purpose = st.text_input("Enter the purpose of research:", value="artificial intelligence")
n = st.text_input("Enter number of iterations:", value="2")
if st.button("Process News", type="primary"):
    if topic and navbar_config and no:
        try:
            graph = GraphBuilder()
            final_summary = graph.invoke({"keyword":[topic],"summary":[],"no_of_iterations":n,"used_keywords":[],"links":[],"purpose":purpose})
            st.header(f"📝 News Summary: {topic}")
            for ind,summ in enumerate(final_summary["summary"]):
                st.markdown(ind+1)
                st.markdown(summ)
            st.markdown(str(final_summary["used_keywords"]))
            st.markdown(str(final_summary["links"]))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a topic!")