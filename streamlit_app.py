import streamlit as st
from study_dep import graph, GraphState, HumanMessage
from typing import List, Dict, Any
import json
import pandas as pd
from io import StringIO

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
        if "file_data" not in st.session_state:
            st.session_state.file_data = None

    def setup_ui(self):
        """Setup the main UI components"""
        with st.sidebar:
            st.header("Controls")
            self.show_file_uploader()
            st.divider()
            self.show_subject_list()
        
        self.show_chat_interface()

    def show_file_uploader(self):
        """Display file uploader and process the file."""
        st.subheader("Upload File")
        uploaded_file = st.file_uploader(
            "Upload a CSV or PDF file for context",
            type=["csv", "pdf"]
        )
        if uploaded_file is not None:
            # Use a button to trigger processing to avoid reprocessing on every interaction
            if st.button("Process File"):
                self.process_uploaded_file(uploaded_file)

    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded file and store its data in session state."""
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    dataframe = pd.read_csv(stringio)
                    st.session_state.file_data = dataframe
                    st.success("CSV file processed successfully!")
                    st.dataframe(dataframe.head())
                elif uploaded_file.name.endswith('.pdf'):
                    # NOTE: You need to install a PDF reader library like 'PyMuPDF'
                    # pip install PyMuPDF
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        st.session_state.file_data = text
                        st.success("PDF file processed successfully!")
                        st.text_area("Extracted Text (first 500 chars)", text[:500], height=150)
                    except ImportError:
                        st.error("PyMuPDF is not installed. Please run 'pip install PyMuPDF' to process PDF files.")
                        st.session_state.file_data = None
                else:
                    st.error("Unsupported file type.")
                    st.session_state.file_data = None
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.session_state.file_data = None

    def show_subject_list(self):
        """Display the current list of subjects"""
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
            "subjects": st.session_state.subjects,
            "file_data": st.session_state.file_data
        }

        # Get response from graph
        with st.spinner("Thinking..."):
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
    st.markdown(
    """
    <div style="text-align: center; padding-top: 50px;">
        <h1 style="font-size: 3em;">📚 Study Department Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
    
    ChatbotUI()

if __name__ == "__main__":
    main()