from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
import json
from utils import LLMRuntimePromptError,base_nodes,base_return_router
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from utils import print_result
from colorama import Fore
import os
os.environ["TAVILY_API_KEY"] = "????????"
os.environ["OPENWEATHERMAP_API_KEY"] = "????????"
model_fn_call = ChatCohere(temperature=0.00001,max_tokens=15000,cohere_api_key="????????")
other_caller = ChatCohere(temperature=0.00001,max_tokens=15000,cohere_api_key="????????")

travily_search = TavilySearchResults()
weather = OpenWeatherMapAPIWrapper()

other_exec_prompt = """
    You a Smart emotional Assistant
    You are creative but try to answer factually and informatively
    No Apologies , No Formality , Be Brief and precise
    You Never ask the user for details or input 
    You Always answer to the question even if you have little input
"""

def eval_math_python(math_expression: int) -> str:
    """ Evaluate Simple Math expressions , + , - / etc , and > <
        It does not accept any variables or functions, only numbers and operators like 5**2 but not other functions like sin,cos,tan etc.
    """
    return str(eval(math_expression))

def get_weather_call(location: str) -> str:
    """Get the current weather information for a specified location. Only works for the current day
        Location is in the format of "City, Country" e.g. 'London, UK'
    """
    return weather.run(location)

def other_exec(query: str) -> str:
    """
    Process Any Information and generate data that other tools cannot process 
    Used when no other tools are relevant or are not available
    Accepts as Argument a query, Example : - query : what is the capital of france ? , Context: i'm searching for a hotel there given that etc
    the query should contain the context and all the related data required to solve the query
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",other_exec_prompt),
        ("user","Query : {input}")
    ])   
    chain = prompt | other_caller
    result = chain.invoke({"input": query})
    return result.content

def web_search_call(query: str) -> str:
    travily_search_results = travily_search.invoke(query)
    return str(travily_search_results)

fn_tools = [convert_to_openai_tool(eval_math_python),convert_to_openai_tool(other_exec),convert_to_openai_tool(travily_search),convert_to_openai_tool(get_weather_call)]

model_fn_system_prompt = """
            You are a smart function calling AI model. 
            You are provided with function signatures within <tools></tools> XML tags. 
            You always call One Function to assist the user with it's query
            Don't make assumptions about what values to plug into functions. Instead, use the given data and transform it to the correct format depending on the tool
            Here are the available tools: 
            <tools> {tools} </tools>
            For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows And En Explentation of the argument and function choice:
            Function Choice and how it is relevant : 
            Args Choice and how it is relevant : 
            <tool_call>
            {{"arguments": <args-dict>, "name": <function-name>}}
            </tool_call>
            """
                        
def extract_tool_name_and_args(content):
    if("<tool_call>\n" in content and "\n</tool_call>" in content):
        json_data = content.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
        return json.loads(json_data)
    raise LLMRuntimePromptError('Bad Format , Make sure the message has the correct json within <tool_call>\n{{"arguments": <args-dict>, "name": <function-name>}}\n</tool_call>')

def get_tool_call(all_states):
    state = all_states["task_executor_history"]
    prompt = ChatPromptTemplate.from_messages([
        ("system",model_fn_system_prompt),
        MessagesPlaceholder("chat_history")
    ])   
    chain = prompt | model_fn_call
    result = chain.invoke({"chat_history": state,"tools": fn_tools})
    print_result(result.content,"Tool Executor : ",Fore.CYAN)
    try:
        fn = extract_tool_name_and_args(result.content)
        print_result(str(fn),"Function Extracted From Tool : ",Fore.RED)

        return {
            "fn": fn,
            "task_executor_history": [AIMessage(result.content)],
            "error": None
        }
    except LLMRuntimePromptError as e:
        print(e)
        return {
            "error": e.message,
            "fn": None,
            "task_executor_history": [AIMessage(result.content),HumanMessage("Error: "+e.message)]
        }   
    except BaseException as e:
        print(e)
        return {
            "error": e,
            "fn": None,
            "task_executor_history": [AIMessage(result.content),HumanMessage("Error occured while calling the tool : "+str(e))]
        } 
        

def fn_caller(all_states):
    isError = all_states["error"] != None
    fn = all_states["fn"]
    if(isError):
        # print("NEXT !: ","get_task_execution_call")
        return "get_task_execution_call"
    if(fn != None):
        # print("NEXT IS A FUNCTION CALL : ",fn)
        return fn["name"]
    # print("NEXT IS get_planner_call")
    return "get_planner_call"

def evaluate_other_exec(all_states):
    fn = all_states["fn"]
    try:
        result = other_exec(**fn["arguments"])
        print_result(result,"Other Exec Result :  ",Fore.BLUE)
        return {
            "error": None,
            "planner_history": [HumanMessage("Result :  "+result)],
            "fn": None
        }    
    except BaseException as e:
        return {
            "error": e,
            "task_executor_history": [HumanMessage("Error occured while calling the tool other_exec: "+str(e))]
        }
            
def evaluate_math_simple_expr(all_states):
    fn = all_states["fn"]
    try:
        result = eval_math_python(**fn["arguments"])
        print_result(result,"Math Python Result :  ",Fore.BLUE)
        return {
            "error": None,
            "planner_history": [HumanMessage("Result :  "+result)],
            "fn": None
        }    
    except BaseException as e:
        return {
            "error": e,
            "task_executor_history": [HumanMessage("Error occured while calling the tool eval_math_python: "+str(e))]
        }    


def evaluate_web_search(all_states):
    fn = all_states["fn"]
    try:
        result = web_search_call(**fn["arguments"])
        print_result(result,"Web Search Result :  ",Fore.BLUE)
        return {
            "error": None,
            "planner_history": [HumanMessage("Result :  "+result)],
            "fn": None
        }    
    except BaseException as e:
        return {
            "error": e,
            "task_executor_history": [HumanMessage("Error occured while calling the tool web_search: "+str(e))]
        }   
        
def evaluate_weather_call(all_states):
    fn = all_states["fn"]
    try:
        result = get_weather_call(**fn["arguments"])
        print_result(result,"Weather Result : ",Fore.BLUE)
        return {
            "error": None,
            "planner_history": [HumanMessage("Result :  "+result)],
            "fn": None
        }    
    except BaseException as e:
        return {
            "error": e,
            "task_executor_history": [HumanMessage("Error occured while calling the tool get_weather_call: "+str(e))]
        } 

def link_graph(graph):
    graph.add_node("get_task_execution_call",get_tool_call)
    graph.add_node("eval_math_python",evaluate_math_simple_expr)
    graph.add_node("other_exec",evaluate_other_exec)
    graph.add_node("tavily_search_results_json",evaluate_web_search)
    graph.add_node("get_weather_call",evaluate_weather_call)
    
def link_conditional_edges(graph):
    graph.add_conditional_edges("get_task_execution_call",fn_caller,
        {
            **base_nodes,
            "eval_math_python": "eval_math_python",
            "other_exec": "other_exec",
            "tavily_search_results_json": "tavily_search_results_json",
            "get_weather_call": "get_weather_call"
        }
    )
    graph.add_conditional_edges("eval_math_python",fn_caller,base_nodes)    
    graph.add_conditional_edges("other_exec",fn_caller,base_nodes)    
    graph.add_conditional_edges("tavily_search_results_json",fn_caller,base_nodes)
    graph.add_conditional_edges("get_weather_call",fn_caller,base_nodes)