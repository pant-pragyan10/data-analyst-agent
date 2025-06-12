import streamlit as st
import os
from data_analyst_agent import DataAnalystAgent
import base64
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = DataAnalystAgent()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("📊 Data Analyst Agent")
st.markdown("""
This application allows you to analyze various types of documents and get insights through natural language interaction.
Upload your file and start asking questions!
""")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your file",
    type=['csv', 'xlsx', 'xls', 'doc', 'docx', 'pdf', 'txt', 'png', 'jpg', 'jpeg', 'gif']
)

# Handle file upload
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load the file using our agent
    if st.session_state.agent.load_file(uploaded_file.name):
        st.session_state.data_loaded = True
        st.success(f"Successfully loaded {uploaded_file.name}")
        
        # Display data preview
        if st.session_state.agent.data_type == 'tabular':
            st.subheader("Data Preview")
            st.dataframe(st.session_state.agent.data.head())
        elif st.session_state.agent.data_type == 'text':
            st.subheader("Text Preview")
            st.text(st.session_state.agent.data[:500] + "...")
        elif st.session_state.agent.data_type == 'image':
            st.subheader("Image Preview")
            st.image(st.session_state.agent.data)
    else:
        st.error("Error loading file")
        st.session_state.data_loaded = False
    
    # Clean up the temporary file
    os.remove(uploaded_file.name)

# Chat interface
st.subheader("Ask Questions About Your Data")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "visualization" in message:
            st.image(message["visualization"])

# Chat input
if st.session_state.data_loaded:
    user_query = st.chat_input("Ask a question about your data...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Get analysis from agent
        with st.spinner("Analyzing..."):
            result = st.session_state.agent.analyze_data(user_query)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Add assistant response to chat history
                response = {"role": "assistant", "content": result["analysis"]}
                
                # Check if visualization is needed
                if "visualization" in result["analysis"].lower():
                    # Extract visualization parameters from the analysis
                    if st.session_state.agent.data_type == 'tabular':
                        # Create a simple visualization based on the data
                        plot = st.session_state.agent.create_visualization(
                            'histogram',
                            column=st.session_state.agent.data.columns[0]
                        )
                        if isinstance(plot, str) and not plot.startswith('Error'):
                            response["visualization"] = base64.b64decode(plot)
                
                st.session_state.chat_history.append(response)
                
                # Display the response
                with st.chat_message("assistant"):
                    st.write(response["content"])
                    if "visualization" in response:
                        st.image(response["visualization"])
else:
    st.info("Please upload a file to start analyzing data.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This Data Analyst Agent can:
    - Analyze various file types (CSV, Excel, Word, PDF, Text, Images)
    - Answer questions about your data
    - Create visualizations
    - Provide insights through natural language interaction
    """)
    
    st.header("Supported File Types")
    st.markdown("""
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - Word documents (.doc, .docx)
    - PDF files (.pdf)
    - Text files (.txt)
    - Image files (.png, .jpg, .jpeg, .gif)
    """)
    
    st.header("Example Questions")
    st.markdown("""
    - "What are the main trends in this data?"
    - "Show me the distribution of [column name]"
    - "What is the correlation between [column1] and [column2]?"
    - "Summarize the key findings"
    - "Create a visualization of [specific aspect]"
    """) 