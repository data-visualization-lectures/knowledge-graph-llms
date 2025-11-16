# Import necessary modules
import streamlit as st
import streamlit.components.v1 as components  # For embedding custom HTML
from generate_knowledge_graph import generate_knowledge_graph, export_graph_to_json, export_graph_to_csv

# Set up Streamlit page configuration
st.set_page_config(
    page_icon=None, 
    layout="wide",  # Use wide layout for better graph display
    initial_sidebar_state="auto", 
    menu_items=None
)

# Set the title of the app
st.title("Knowledge Graph From Text")

# Initialize session state for graph data
if "graph_documents" not in st.session_state:
    st.session_state.graph_documents = None
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None

# Sidebar section for user input method
st.sidebar.title("Input document")
input_method = st.sidebar.radio(
    "Choose an input method:",
    ["Upload txt", "Input text"],  # Options for uploading a file or manually inputting text
)

# Case 1: User chooses to upload a .txt file
if input_method == "Upload txt":
    # File uploader widget in the sidebar
    uploaded_file = st.sidebar.file_uploader(label="Upload file", type=["txt"])
    
    if uploaded_file is not None:
        # Read the uploaded file content and decode it as UTF-8 text
        text = uploaded_file.read().decode("utf-8")

        # Button to generate the knowledge graph
        if st.sidebar.button("Generate Knowledge Graph"):
            with st.spinner("Generating knowledge graph..."):
                # Call the function to generate the graph from the text
                net, graph_documents = generate_knowledge_graph(text)
                st.session_state.graph_documents = graph_documents
                st.success("Knowledge graph generated successfully!")

                # Save the graph to an HTML file
                output_file = "knowledge_graph.html"
                net.save_graph(output_file)

                # Read and store HTML in session state
                with open(output_file, 'r', encoding='utf-8') as HtmlFile:
                    st.session_state.graph_html = HtmlFile.read()

        # Display the graph if it exists in session state
        if st.session_state.graph_html is not None:
            components.html(st.session_state.graph_html, height=1000)

        # Display download buttons if graph data exists
        if st.session_state.graph_documents is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📥 Download Graph Data")

            col1, col2 = st.sidebar.columns(2)

            with col1:
                json_data = export_graph_to_json(st.session_state.graph_documents)
                st.download_button(
                    label="📄 JSON",
                    data=json_data,
                    file_name="knowledge_graph.json",
                    mime="application/json"
                )

            with col2:
                csv_data = export_graph_to_csv(st.session_state.graph_documents)
                st.download_button(
                    label="📊 CSV",
                    data=csv_data,
                    file_name="knowledge_graph.csv",
                    mime="text/csv"
                )

# Case 2: User chooses to directly input text
else:
    # Text area for manual input
    text = st.sidebar.text_area("Input text", height=300)

    if text:  # Check if the text area is not empty
        if st.sidebar.button("Generate Knowledge Graph"):
            with st.spinner("Generating knowledge graph..."):
                # Call the function to generate the graph from the input text
                net, graph_documents = generate_knowledge_graph(text)
                st.session_state.graph_documents = graph_documents
                st.success("Knowledge graph generated successfully!")

                # Save the graph to an HTML file
                output_file = "knowledge_graph.html"
                net.save_graph(output_file)

                # Read and store HTML in session state
                with open(output_file, 'r', encoding='utf-8') as HtmlFile:
                    st.session_state.graph_html = HtmlFile.read()

        # Display the graph if it exists in session state
        if st.session_state.graph_html is not None:
            components.html(st.session_state.graph_html, height=1000)

        # Display download buttons if graph data exists
        if st.session_state.graph_documents is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📥 Download Graph Data")

            col1, col2 = st.sidebar.columns(2)

            with col1:
                json_data = export_graph_to_json(st.session_state.graph_documents)
                st.download_button(
                    label="📄 JSON",
                    data=json_data,
                    file_name="knowledge_graph.json",
                    mime="application/json"
                )

            with col2:
                csv_data = export_graph_to_csv(st.session_state.graph_documents)
                st.download_button(
                    label="📊 CSV",
                    data=csv_data,
                    file_name="knowledge_graph.csv",
                    mime="text/csv"
                )