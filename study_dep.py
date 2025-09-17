import streamlit as st
import os
from typing import List, TypedDict, Annotated, Optional, Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd


# ---------------------- Load Environment Variables ----------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    st.error("❌ MONGO_URI is not set in .env file")
    st.stop()

# ---------------------- MongoDB Setup ----------------------
client = MongoClient(MONGO_URI)
db = client["student_assistant"]
subjects_collection = db["subjects"]

# ---------- DB Utility Functions ----------
def add_subject_db(subject: str):
    subjects_collection.update_one(
        {"name": subject}, {"$set": {"name": subject}}, upsert=True
    )

def delete_subject_db(subject: str):
    subjects_collection.delete_one({"name": subject})

def list_subjects_db() -> List[str]:
    return [doc["name"] for doc in subjects_collection.find()]

def add_topic_db(subject: str, topic: str):
    """Add topic with completion status (default: False)"""
    subjects_collection.update_one(
        {"name": subject},
        {"$addToSet": {"topics": {"name": str(topic), "completed": False}}},
        upsert=True,
    )

def list_topics_db(subject: str) -> List[Dict]:
    """Returns list of topic dictionaries with name and completion status"""
    doc = subjects_collection.find_one({"name": subject})
    if not doc:
        return []
    
    topics = doc.get("topics", [])
    # Handle both old string format and new dict format
    normalized_topics = []
    for t in topics:
        if isinstance(t, str):
            normalized_topics.append({"name": t, "completed": False})
        elif isinstance(t, dict):
            normalized_topics.append(t)
    return normalized_topics

def find_subject_by_topic_db(topic: str) -> str | None:
    # Search in both old and new format
    doc = subjects_collection.find_one({
        "$or": [
            {"topics": topic},  # old string format
            {"topics.name": topic}  # new dict format
        ]
    })
    return doc["name"] if doc else None

def mark_topic_complete_db(subject: str, topic: str):
    """Mark a topic as complete"""
    subjects_collection.update_one(
        {"name": subject, "topics.name": topic},
        {"$set": {"topics.$.completed": True}}
    )

def mark_topic_incomplete_db(subject: str, topic: str):
    """Mark a topic as incomplete"""
    subjects_collection.update_one(
        {"name": subject, "topics.name": topic},
        {"$set": {"topics.$.completed": False}}
    )

def get_incomplete_topics_db(subject: str = None) -> Dict[str, List[str]]:
    """Get all incomplete topics, optionally filtered by subject"""
    result = {}
    
    if subject:
        subjects = [subject]
    else:
        subjects = list_subjects_db()
    
    for subj in subjects:
        topics = list_topics_db(subj)
        incomplete = [t["name"] for t in topics if not t.get("completed", False)]
        if incomplete:
            result[subj] = incomplete
    
    return result

def get_completed_topics_db(subject: str = None) -> Dict[str, List[str]]:
    """Get all completed topics, optionally filtered by subject"""
    result = {}
    
    if subject:
        subjects = [subject]
    else:
        subjects = list_subjects_db()
    
    for subj in subjects:
        topics = list_topics_db(subj)
        completed = [t["name"] for t in topics if t.get("completed", False)]
        if completed:
            result[subj] = completed
    
    return result

def get_topic_completion_status_db(subject: str, topic: str) -> bool:
    """Get completion status of a specific topic"""
    topics = list_topics_db(subject)
    for t in topics:
        if t.get("name") == topic:
            return t.get("completed", False)
    return False

def add_topics_to_all_subjects_db(topics: List[str]):
    subjects = list_subjects_db()
    for subject in subjects:
        for topic in topics:
            subjects_collection.update_one(
                {"name": subject},
                {"$addToSet": {"topics": {"name": str(topic), "completed": False}}},
                upsert=True,
            )

def delete_topic_from_all_subjects_db(topic: str):
    # Case 1: plain string
    subjects_collection.update_many(
        {},
        {"$pull": {"topics": topic}}
    )
    # Case 2: dict format
    subjects_collection.update_many(
        {},
        {"$pull": {"topics": {"name": topic}}}
    )

def delete_topic_from_subject_db(subject: str, topic: str):
    subjects_collection.update_one(
        {"name": subject},
        {"$pull": {"topics": {"name": topic}}}
    )

def delete_topic_from_multiple_subjects_db(subjects: List[str], topic: str):
    subjects_collection.update_many(
        {"name": {"$in": subjects}},
        {"$pull": {"topics": {"name": topic}}}
    )


# ---------------------- Tools ----------------------
@tool
def add_subject(subject: str) -> str:
    """Adds a subject to the list of subjects."""
    add_subject_db(subject)
    return f"✅ Successfully added '{subject}'."

@tool
def delete_subject(subject: str) -> str:
    """Deletes a subject from the list of subjects."""
    delete_subject_db(subject)
    return f"🗑️ Successfully deleted '{subject}'."

@tool
def list_subjects() -> str:
    """Lists all subjects."""
    subjects = list_subjects_db()
    if not subjects:
        return "📂 No subjects available."
    return "📚 Current subjects: " + ", ".join(subjects)

@tool
def add_topic(subject: str, topic: str) -> str:
    """Adds a topic under a subject."""
    add_topic_db(subject, topic)
    return f"✅ Successfully added topic '{topic}' under '{subject}'."

@tool
def list_topics(subject: str) -> str:
    """Lists all topics under a subject with their completion status."""
    topics = list_topics_db(subject)
    if not topics:
        return f"📂 No topics found under '{subject}'."
    
    topic_list = []
    for t in topics:
        name = t.get("name", str(t))
        status = "✅" if t.get("completed", False) else "❌"
        topic_list.append(f"{name} {status}")
    
    return f"📚 Topics under {subject}: " + ", ".join(topic_list)

@tool
def mark_topic_complete(subject: str, topic: str) -> str:
    """Marks a topic as complete/done."""
    mark_topic_complete_db(subject, topic)
    return f"✅ Topic '{topic}' in '{subject}' marked as complete!"

@tool
def mark_topic_incomplete(subject: str, topic: str) -> str:
    """Marks a topic as incomplete/not done."""
    mark_topic_incomplete_db(subject, topic)
    return f"❌ Topic '{topic}' in '{subject}' marked as incomplete."

@tool
def get_incomplete_topics(subject: str = None) -> str:
    """Gets all incomplete/pending topics. Can filter by subject."""
    incomplete = get_incomplete_topics_db(subject)
    
    if not incomplete:
        if subject:
            return f"🎉 All topics in '{subject}' are complete!"
        else:
            return "🎉 All topics are complete!"
    
    result = "📝 **Topics left to complete:**\n"
    for subj, topics in incomplete.items():
        result += f"\n**{subj}:**\n"
        for topic in topics:
            result += f"  - {topic} ❌\n"
    
    return result

@tool
def get_completed_topics(subject: str = None) -> str:
    """Gets all completed topics. Can filter by subject."""
    completed = get_completed_topics_db(subject)
    
    if not completed:
        if subject:
            return f"❌ No completed topics in '{subject}' yet."
        else:
            return "❌ No completed topics yet."
    
    result = "✅ **Completed topics:**\n"
    for subj, topics in completed.items():
        result += f"\n**{subj}:**\n"
        for topic in topics:
            result += f"  - {topic} ✅\n"
    
    return result

@tool
def get_topic_status(subject: str, topic: str) -> str:
    """Checks if a specific topic is complete or not."""
    is_complete = get_topic_completion_status_db(subject, topic)
    status = "complete ✅" if is_complete else "incomplete ❌"
    return f"Topic '{topic}' in '{subject}' is {status}"

@tool
def find_subject_by_topic(topic: str) -> str:
    """Finds which subject a topic belongs to."""
    subject = find_subject_by_topic_db(topic)
    if not subject:
        return f"❌ Topic '{topic}' not found in any subject."
    return f"📖 Topic '{topic}' belongs to subject '{subject}'."

@tool
def add_topics_to_all_subjects(topics: List[str]) -> str:
    """Adds multiple topics to all subjects."""
    add_topics_to_all_subjects_db(topics)
    return f"✅ Successfully added topics {', '.join(map(str, topics))} to all subjects."

@tool
def delete_topic_from_all_subjects(topic: str) -> str:
    """Deletes a topic from all subjects."""
    delete_topic_from_all_subjects_db(topic)
    return f"🗑️ Successfully deleted topic '{topic}' from all subjects."

@tool
def delete_topic_from_subject(subject: str, topic: str) -> str:
    """Deletes a topic from a specific subject."""
    delete_topic_from_subject_db(subject, topic)
    return f"🗑️ Successfully deleted topic '{topic}' from subject '{subject}'."

@tool
def delete_topic_from_multiple_subjects(subjects: List[str], topic: str) -> str:
    """Deletes a topic from multiple specific subjects."""
    delete_topic_from_multiple_subjects_db(subjects, topic)
    return f"🗑️ Successfully deleted topic '{topic}' from subjects: {', '.join(subjects)}."


# All tools
tools = [
    add_subject,
    delete_subject,
    list_subjects,
    add_topic,
    list_topics,
    find_subject_by_topic,
    add_topics_to_all_subjects,
    delete_topic_from_all_subjects,
    delete_topic_from_subject,
    delete_topic_from_multiple_subjects,
    mark_topic_complete,  # ✅ NEW
    mark_topic_incomplete,  # ✅ NEW
    get_incomplete_topics,  # ✅ NEW
    get_completed_topics,  # ✅ NEW
    get_topic_status,  # ✅ NEW
]


# ---------------------- Graph State ----------------------
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    subjects: List[str]
    file_data: Optional[Any]  # Can be a pandas DataFrame or string from a PDF


# ---------------------- Model ----------------------
def call_model(state: GraphState):
    messages = state["messages"]
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ---------------------- Update State ----------------------
def update_state_from_tool_calls(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or last_message.additional_kwargs.get("tool_calls", [])

    if not tool_calls:
        return state

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {}) or {}
        subject = tool_args.get("subject")
        subjects = tool_args.get("subjects")
        topic = tool_args.get("topic")
        topics = tool_args.get("topics")

        if tool_name == "add_subject" and subject:
            add_subject_db(subject)
        elif tool_name == "delete_subject" and subject:
            delete_subject_db(subject)
        elif tool_name == "add_topic" and subject and topic:
            add_topic_db(subject, topic)
        elif tool_name == "add_topics_to_all_subjects" and topics:
            add_topics_to_all_subjects_db(topics)
        elif tool_name == "delete_topic_from_all_subjects" and topic:
            delete_topic_from_all_subjects_db(topic)
        elif tool_name == "delete_topic_from_subject" and subject and topic:
            delete_topic_from_subject_db(subject, topic)
        elif tool_name == "delete_topic_from_multiple_subjects" and subjects and topic:
            delete_topic_from_multiple_subjects_db(subjects, topic)
        elif tool_name == "mark_topic_complete" and subject and topic:
            mark_topic_complete_db(subject, topic)
        elif tool_name == "mark_topic_incomplete" and subject and topic:
            mark_topic_incomplete_db(subject, topic)

    current_subjects = list_subjects_db()
    return {**state, "subjects": current_subjects}


# ---------------------- Control Flow ----------------------
def should_continue(state: GraphState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or last_message.additional_kwargs.get("tool_calls", [])
    return "continue_to_tools" if tool_calls else "end_conversation"


# ---------------------- Workflow ----------------------
workflow = StateGraph(GraphState)
workflow.add_node("agent", call_model)
workflow.add_node("update_state", update_state_from_tool_calls)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

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


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Student Assistant", page_icon="🎓")

# Add custom CSS for better visualization
st.markdown("""
<style>
    .completed-topic { color: green; font-weight: bold; }
    .incomplete-topic { color: red; }
    .progress-bar { background-color: #f0f0f0; border-radius: 10px; padding: 3px; }
    .progress-fill { background-color: #4CAF50; height: 20px; border-radius: 7px; }
</style>
""", unsafe_allow_html=True)

# Conversation state
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "subjects": list_subjects_db(),
    }

# Show conversation
for msg in st.session_state.graph_state["messages"]:
    role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
    st.markdown(f"**{role}:** {msg.content}")

# Enhanced sidebar with completion tracking
st.sidebar.header("📚 Subjects & Progress")
subjects = list_subjects_db()

if subjects:
    # Overall progress
    total_topics = 0
    completed_topics = 0
    
    for sub in subjects:
        topics = list_topics_db(sub)
        total_topics += len(topics)
        completed_topics += sum(1 for t in topics if t.get("completed", False))
    
    if total_topics > 0:
        progress = completed_topics / total_topics
        st.sidebar.progress(progress)
        st.sidebar.write(f"Overall Progress: {completed_topics}/{total_topics} ({progress*100:.1f}%)")
        st.sidebar.divider()
    
    # Subject-wise breakdown
    for sub in subjects:
        topics = list_topics_db(sub)
        if topics:
            completed = sum(1 for t in topics if t.get("completed", False))
            total = len(topics)
            
            st.sidebar.write(f"**{sub}** ({completed}/{total})")
            
            for t in topics:
                name = t.get("name", str(t))
                status = "✅" if t.get("completed", False) else "⭕"
                st.sidebar.write(f"  {status} {name}")
        else:
            st.sidebar.write(f"**{sub}** (no topics yet)")
    
    st.sidebar.divider()
    
    # Quick stats
    incomplete = get_incomplete_topics_db()
    total_incomplete = sum(len(topics) for topics in incomplete.values())
    if total_incomplete > 0:
        st.sidebar.info(f"📝 {total_incomplete} topics left to complete")
    else:
        st.sidebar.success("🎉 All topics completed!")
else:
    st.sidebar.write("No subjects available.")

def create_dataframe_tool(state: GraphState):
    """A tool that can answer questions about a pandas DataFrame."""
    df = state.get('file_data')
    if not isinstance(df, pd.DataFrame):
        return "No CSV data available to query."
    
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0) 
    agent_executor = create_pandas_dataframe_agent(llm, df, verbose=True)
    
    # Placeholder for integration
    pass