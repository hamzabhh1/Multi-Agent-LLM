from colorama import  Back, Fore

class LLMRuntimePromptError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

def base_return_router(all_states):
    return all_states["last_caller"]
base_nodes = {
    "get_input": "get_input",
    "get_task_execution_call": "get_task_execution_call",
    "get_planner_call": "get_planner_call",
    "get_assistant_call": "get_assistant_call"
}

def print_result(text,prefix,color = Fore.WHITE):
    print(Fore.GREEN + "-------------------------------------------------------------------")
    print(Fore.RESET + color + prefix,flush=True)
    if(text != None):
        print(color + text)
    else:
        print(color + "None")
    print(Fore.GREEN + "-------------------------------------------------------------------")
    print(Fore.RESET)
    return text