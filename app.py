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
st.title("テキストから知識グラフを生成")

# Initialize session state for graph data
if "graph_documents" not in st.session_state:
    st.session_state.graph_documents = None
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None

# Initialize session state for API key validation
if "api_key_validated" not in st.session_state:
    st.session_state.api_key_validated = False
if "last_validated_key" not in st.session_state:
    st.session_state.last_validated_key = None

# Dynamic sidebar width based on graph generation state
sidebar_width = "200px" if st.session_state.graph_html is not None else "500px"

# Apply custom CSS for sidebar width
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        width: {sidebar_width} !important;
        min-width: {sidebar_width} !important;
        max-width: {sidebar_width} !important;
    }}
    [data-testid="stSidebarContent"] {{
        width: {sidebar_width} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Check if API key is available in environment variables
from dotenv import load_dotenv
import os

load_dotenv()
env_api_key = os.getenv("OPENAI_API_KEY")

# Sidebar section for API key input
st.sidebar.title("🔑 API設定")

if env_api_key:
    # If API key exists in environment, use it and skip user input
    st.sidebar.success("✅ 環境変数からAPIキーを読み込みました")
    api_key = env_api_key
    st.session_state.api_key_validated = True
else:
    # If no API key in environment, show input form
    api_key = st.sidebar.text_input(
        "OpenAI APIキーを入力",
        type="password",
        help="あなたのOpenAI APIキーを入力してください。キーは保存されません。"
    )
    
    # Reset validation if API key changed
    if api_key != st.session_state.last_validated_key:
        st.session_state.api_key_validated = False
        st.session_state.last_validated_key = None
    
    if not api_key:
        st.sidebar.warning("⚠️ APIキーを入力してください")
        st.info(
            "このツールを使用するには、OpenAI APIキーが必要です。\n\n"
            "APIキーは [OpenAI Platform](https://platform.openai.com/api-keys) で取得できます。\n\n"
            "**注意**: 入力されたAPIキーはセッション中のみ使用され、保存されません。"
        )
        st.stop()
    else:
        # Validate API key button
        if not st.session_state.api_key_validated:
            if st.sidebar.button("🔍 APIキーを検証", type="primary"):
                with st.spinner("APIキーを検証中..."):
                    try:
                        # Test API key with a simple call
                        from langchain_openai import ChatOpenAI
                        test_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=api_key)
                        # Make a minimal API call to verify the key
                        test_llm.invoke("test")
                        st.session_state.api_key_validated = True
                        st.session_state.last_validated_key = api_key
                        st.sidebar.success("✅ APIキーが検証されました")
                    except Exception as e:
                        st.sidebar.error(f"❌ APIキーが無効です: {str(e)}")
                        st.stop()
            else:
                st.sidebar.info("👆 APIキーを検証してください")
                st.stop()
        else:
            st.sidebar.success("✅ APIキーが検証されました")

st.sidebar.markdown("---")

# Sidebar section for prompt customization
st.sidebar.title("⚙️ プロンプト設定")
from generate_knowledge_graph import DEFAULT_PROMPT_TEMPLATE

with st.sidebar.expander("プロンプトをカスタマイズ", expanded=False):
    st.markdown("**プロンプトテンプレート**")
    st.caption("知識グラフ抽出に使用するプロンプトをカスタマイズできます。`{input}`はテキストが挿入される場所です。")
    
    custom_prompt = st.text_area(
        "プロンプトテンプレート",
        value=DEFAULT_PROMPT_TEMPLATE,
        height=300,
        help="プロンプトをカスタマイズして、抽出する情報を調整できます。",
        label_visibility="collapsed"
    )

st.sidebar.markdown("---")

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
                try:
                    # Call the function to generate the graph from the text
                    net, graph_documents = generate_knowledge_graph(text, api_key=api_key, prompt_template=custom_prompt)
                    st.session_state.graph_documents = graph_documents
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    st.stop()
                st.success("知識グラフを生成しました！")

                # Save the graph to an HTML file
                output_file = "knowledge_graph.html"
                net.save_graph(output_file)

                # Read and store HTML in session state
                with open(output_file, 'r', encoding='utf-8') as HtmlFile:
                    st.session_state.graph_html = HtmlFile.read()
                
                # Rerun to apply sidebar width change
                st.rerun()

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
                try:
                    # Call the function to generate the graph from the input text
                    net, graph_documents = generate_knowledge_graph(text, api_key=api_key, prompt_template=custom_prompt)
                    st.session_state.graph_documents = graph_documents
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    st.stop()
                st.success("知識グラフを生成しました！")

                # Save the graph to an HTML file
                output_file = "knowledge_graph.html"
                net.save_graph(output_file)

                # Read and store HTML in session state
                with open(output_file, 'r', encoding='utf-8') as HtmlFile:
                    st.session_state.graph_html = HtmlFile.read()
                
                # Rerun to apply sidebar width change
                st.rerun()

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