from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from langgraph.graph import END, MessageGraph
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from utils import LLMRuntimePromptError,base_nodes,base_return_router,print_result
from langchain_core.output_parsers import StrOutputParser
from colorama import Fore
model_call = ChatCohere(temperature=0.00001,max_tokens=15000,cohere_api_key="????????")
model_prompt = """
    No Formality, No Apologies , be Brief and precise
    You take any request , from poetry , to code
    You are the entry point in an agent system , you never do the job , you delegate it to the planner agent
    You Communicate with the planner agent to give him the task and the context , you never do the task yourself or ask from the planner details about the task
    You only always ask the user for more details and context and removes ambiguity and clarify the task itself
    You Always answer with this format if the conditions are met , only once at a time : 
    For Asking Details from the user : Use prefix <clarification> . You End the entire discussion with </clarification>
    For Delegating the task + context to the planner agent :  <planner_call> , add the task and the context / user inputs you got with it . You End the entire discussion with </planner_call>
    For Answering the user Finally once the planner finish the job :  Use prefix <answer> . You End the discussion with </answer>
    You Can Repeat the process until you have a clear task and context , send commands to the planner , and finish with the <answer> 
    """
str_parser = StrOutputParser()
def get_input(all_states):
    inp = input("You: ")
    return {
        "error": None,
        "next_action": "get_assistant_call",
        "assistant_history": [HumanMessage(inp)]
    }
def extract_action(content):
    if("<planner_call>" in content):
        task = content.split("<planner_call>")[1].split("</planner_call>")[0]
        return task
    return None
def get_assistant_call(all_states):
    state = all_states["assistant_history"]
    prompt = ChatPromptTemplate.from_messages([
        ("system",model_prompt),
        MessagesPlaceholder("chat_history")
    ])   
    chain = prompt | model_call | str_parser
    result = chain.invoke({"chat_history": state})
    print_result(result,"Assistant: ")
    task = extract_action(result)
    print_result(task,"Assistant Task: ",Fore.CYAN)
    if(task != None):
        return {
            "error": None,
            "next_action": "get_planner_call",
            "planner_history": [HumanMessage("Assitant Task: "+task)]
        }
    return {
        "error": None,
        "next_action": "get_input",
        "assistant_history": [AIMessage(result)]
    }
        

def assitant_router(all_states):
    # print("NEXT : "+all_states["next_action"])
    return all_states["next_action"]

def link_graph(graph):
    graph.add_node("get_assistant_call",get_assistant_call)
    graph.add_node("get_input",get_input)
    graph.set_entry_point("get_input")
def link_conditional_edges(graph):
    graph.add_conditional_edges("get_input",assitant_router,base_nodes)
    graph.add_conditional_edges("get_assistant_call",assitant_router,base_nodes)