from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from langgraph.graph import END, MessageGraph
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from utils import LLMRuntimePromptError,base_nodes,base_return_router
from langchain_cohere import ChatCohere
from colorama import Fore
from utils import print_result

model_call = ChatCohere(temperature=0.00001,max_tokens=15000,cohere_api_key="????????")
model_prompt = """
    You are the Planner, The Brain in an agent system workflow , You Receive Tasks From the assistant
    You Collaborate with the searcher_agent and task_executor agents ,You do not pocess knowledge except for reasoning , you query them for data and wait for them before proceeding
    You Never infer knowledge or hallucinate informations , you do not have factual information , for factual information you always ask the researcher agent repeatidly
    You Specialise in reasoning in order to solve a task Step By Step , each step is atomic and simple in a divide and conquer manner
    When given a task , you enter a reasoning phase , you always answer with this format, only once at a time and never together : 
        - NEXT STEP : <searcher_call> One atomic information you need to search at a time, never search for more than on thing at a time ! if you want information about A & B , Search for A , Once you have an answer , Search for B , Never A And B Together , ends the discussion with </searcher_call>
        - NEXT STEP : <task_execution> Actions to execute With The input and the expected result and how relevant it should be , to be executed by the agents , Only One Atomic Action at a time , never more than one at a time !, ends the discussion with </task_execution>
        - NEXT STEP : <final_answer> The final answer to the task and to the reasoning procedure once all information and steps are executed , ends with </final_answer>
    No Apologies , No Formalities , No Extra Verbosity
    NEXT STEP : 
    """
            
def extract_action(content: str):
    if content.startswith("NEXT STEP : <searcher_call>"):
        return "get_task_execution_call"
    if content.startswith("NEXT STEP : <task_execution>"):
        return "get_task_execution_call"
    if content.startswith("NEXT STEP : <final_answer>"):
        return "get_assistant_call"
    return None
def get_planner_call(all_states):
    state = all_states["planner_history"]
    prompt = ChatPromptTemplate.from_messages([
        ("system",model_prompt),
        ("user","Task : Determine the Hottest City between London And Tunis Yesterday in celcius and convert it to fahreneit"),
        ("assistant","NEXT STEP : <searcher_call>Weather for Tunis Yesterday in celcius </searcher_call>"),
        ("user","26.5 Degrees Celsius"),
        ("assistant","NEXT STEP : <searcher_call>Weather for London Yesterday in celcius </searcher_call>"),
        ("user","-10 Celsius"),
        ("assistant","NEXT STEP : <task_execution> Compare the two temperatures : 26.5 and -10 </task_execution>"),
        ("user","Result for calling eval_math with arguments : 26.5 > -10 is true"),
        ("assistant","NEXT STEP : <searcher_call> how to convert from Fahreneit to celcius formula </searcher_call>"),
        ("user"," °F = °C × (9/5) + 32"),
        ("assistant","NEXT STEP : <task_execution> Compute 25.6 × (9/5) + 32 </task_execution>"),
        ("user","78"),
        ("assistant","NEXT STEP : <final_answer> The hottest city is tunis with 26.5 celcius and converted to fahreneit is 78 </final_answer>"),
        MessagesPlaceholder("chat_history")
    ])   
    chain = prompt | model_call
    result = chain.invoke({"chat_history": state})
    task = extract_action(result.content)

    print_result(result.content,"Planner Agent: ",Fore.CYAN)

    
    if(task == None):
        return {
            "error": "Wrong Format , Please Respect the Format -- NEXT STEP : <action>",
            "next_action": "get_planner_call",
            "planner_history": [AIMessage(result.content),HumanMessage("Wrong Format , Please Respect the Format -- NEXT STEP : <action>")]
        }
    
    if task == "get_task_execution_call":
        return {
            "error": None,
            "next_action": "get_task_execution_call",
            "task_executor_history": [HumanMessage(result.content)],
            "planner_history": [AIMessage(result.content)]
        }
    if task == "get_assistant_call":
        return {
            "error": None,
            "next_action": task,
            "assistant_history": [HumanMessage("Planner Agent: "+result.content)],
            "planner_history": [AIMessage(result.content)]
        }
        

def planner_router(all_states):
    # print("NEXT : "+all_states["next_action"])
    return all_states["next_action"]

def link_graph(graph):
    graph.add_node("get_planner_call",get_planner_call)
def link_conditional_edges(graph):
    graph.add_conditional_edges("get_planner_call",planner_router,base_nodes)