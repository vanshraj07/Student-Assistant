import os
from typing import TypedDict, Annotated, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


@tool
def add_subject(subject: str) -> str:
    """Adds a subject to the list of subjects."""
    return f"Successfully added '{subject}'. The user should be notified of this success."


@tool
def delete_subject(subject: str) -> str:
    """Deletes a subject from the list of subjects."""
    return f"Successfully deleted '{subject}'. The user should be notified of this success."

tools = [add_subject, delete_subject]


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    subjects: List[str]



def call_model(state: GraphState):
    print("---AGENT---")
    messages = state['messages']
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)



def update_state_from_tool_calls(state: GraphState) -> GraphState:
    print("---UPDATING SUBJECTS LIST---")
    last_message = state['messages'][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return state

    current_subjects = state.get('subjects', [])
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        subject = tool_args.get('subject')
        
        if not subject:
            continue

        if tool_name == 'add_subject':
            if subject not in current_subjects:
                print(f"Adding '{subject}' to the list.")
                current_subjects.append(subject)
        elif tool_name == 'delete_subject':
            if subject in current_subjects:
                print(f"Deleting '{subject}' from the list.")
                current_subjects.remove(subject)

    
    return {"subjects": current_subjects}



def should_continue(state: GraphState):
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue_to_tools"
    else:
        return "end_conversation"


workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("agent", call_model)
workflow.add_node("update_state", update_state_from_tool_calls)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.set_entry_point("agent")

# Add the conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue_to_tools": "update_state", 
        "end_conversation": END,
    },
)


workflow.add_edge("update_state", "tools")
workflow.add_edge("tools", "agent") 


graph = workflow.compile()

print("Graph compiled! You can now interact with it.")