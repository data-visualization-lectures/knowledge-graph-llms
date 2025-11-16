# Import necessary modules
import streamlit as st
import streamlit.components.v1 as components  # For embedding custom HTML
from generate_knowledge_graph import generate_knowledge_graph, export_graph_to_json, export_graph_to_csv

# JavaScript to translate Streamlit components to Japanese
translate_script = """
<script>
function translateStreamlitUI() {
    // Wait for DOM to be fully loaded
    setTimeout(() => {
        // Translate file uploader text
        const labels = document.querySelectorAll('*');
        labels.forEach(el => {
            if (el.textContent.includes('Drag and drop file here')) {
                el.textContent = el.textContent.replace('Drag and drop file here', 'ファイルをドラッグ&ドロップしてください');
            }
            if (el.textContent.includes('Browse files')) {
                el.textContent = el.textContent.replace('Browse files', 'ファイルを選択');
            }
            if (el.textContent.includes('Limit 200MB per file')) {
                el.textContent = el.textContent.replace('Limit 200MB per file', 'ファイルサイズの制限: 200MB');
            }
        });
    }, 500);
}

// Run translation when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', translateStreamlitUI);
} else {
    translateStreamlitUI();
}

// Also run translation after every re-run
const observer = new MutationObserver(translateStreamlitUI);
observer.observe(document.body, { childList: true, subtree: true });
</script>
"""

# Inject the translation script
components.html(translate_script)

# Set up Streamlit page configuration
st.set_page_config(
    page_icon=None, 
    layout="wide",  # Use wide layout for better graph display
    initial_sidebar_state="auto", 
    menu_items=None
)

# Set the title of the app
st.title("テキストから知識グラフを生成")

# Initialize session state for graph data
if "graph_documents" not in st.session_state:
    st.session_state.graph_documents = None
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None

# Sidebar section for user input method
st.sidebar.title("ドキュメント入力")
input_method = st.sidebar.radio(
    "入力方法を選択:",
    ["ファイルをアップロード", "テキストを直接入力"],  # Options for uploading a file or manually inputting text
)

# Case 1: User chooses to upload a .txt file
if input_method == "ファイルをアップロード":
    # File uploader widget in the sidebar
    uploaded_file = st.sidebar.file_uploader(label="ファイルを選択", type=["txt"])
    
    if uploaded_file is not None:
        # Read the uploaded file content and decode it as UTF-8 text
        text = uploaded_file.read().decode("utf-8")

        # Button to generate the knowledge graph
        if st.sidebar.button("知識グラフを生成"):
            with st.spinner("知識グラフを生成中..."):
                # Call the function to generate the graph from the text
                net, graph_documents = generate_knowledge_graph(text)
                st.session_state.graph_documents = graph_documents
                st.success("知識グラフを生成しました！")

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
            st.sidebar.subheader("📥 グラフデータをダウンロード")

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
    text = st.sidebar.text_area("テキストを入力", height=300)

    if text:  # Check if the text area is not empty
        if st.sidebar.button("知識グラフを生成"):
            with st.spinner("知識グラフを生成中..."):
                # Call the function to generate the graph from the input text
                net, graph_documents = generate_knowledge_graph(text)
                st.session_state.graph_documents = graph_documents
                st.success("知識グラフを生成しました！")

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
            st.sidebar.subheader("📥 グラフデータをダウンロード")

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