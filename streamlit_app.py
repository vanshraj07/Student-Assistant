import streamlit as st
from study_dep import graph, GraphState, HumanMessage
from typing import List, Dict
import json

class ChatbotUI:
    def __init__(self):
        self.initialize_session_state()
        self.setup_ui()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "subjects" not in st.session_state:
            st.session_state.subjects = []
        if "chat_state" not in st.session_state:
            st.session_state.chat_state = {"messages": [], "subjects": []}

    def setup_ui(self):
        """Setup the main UI components"""
        st.title("Study Department Chatbot")
        self.show_subject_list()
        self.show_chat_interface()

    def show_subject_list(self):
        """Display the current list of subjects"""
        with st.sidebar:
            st.header("Current Subjects")
            if st.session_state.subjects:
                for subject in st.session_state.subjects:
                    st.write(f"• {subject}")
            else:
                st.write("No subjects added yet")

    def show_chat_interface(self):
        """Display the chat interface and handle messages"""
        # Display chat history
        for message in st.session_state.messages:
            role = "assistant" if message.get("role") == "assistant" else "user"
            with st.chat_message(role):
                st.write(message.get("content"))

        # Chat input
        if prompt := st.chat_input("How can I help you manage your subjects?"):
            self.process_user_message(prompt)

    def process_user_message(self, prompt: str):
        """Process user input and generate response"""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Prepare state for graph
        current_state = {
            "messages": [HumanMessage(content=prompt)],
            "subjects": st.session_state.subjects
        }

        # Get response from graph
        result = graph.invoke(current_state)

        # Update session state
        if "subjects" in result:
            st.session_state.subjects = result["subjects"]

        # Extract and display assistant's response
        if result["messages"]:
            response = result["messages"][-1]
            content = response.content
            st.session_state.messages.append({"role": "assistant", "content": content})
            with st.chat_message("assistant"):
                st.write(content)

def main():
    st.set_page_config(
        page_title="Study Department Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    ChatbotUI()

if __name__ == "__main__":
    main()