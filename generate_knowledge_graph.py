from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pyvis.network import Network

from dotenv import load_dotenv
import os
import asyncio
import json
import csv
from io import StringIO


# Load the .env file
load_dotenv()
# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Please set it in Streamlit Cloud Secrets or in your .env file."
    )

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create a custom prompt for Japanese relationship extraction
japanese_prompt = PromptTemplate.from_template(
    """以下のテキストから知識グラフの情報を抽出してください。
エンティティ（人物、組織、場所など）と、それらの間の関係性を日本語で抽出します。

関係性は以下の例のように、わかりやすい日本語で出力してください：
- 所属している
- 友人である
- 位置している
- 質問する
- 説明する
- 関連している

テキスト：
{input}

エンティティと関係性を抽出してください。"""
)

graph_transformer = LLMGraphTransformer(llm=llm, prompt=japanese_prompt)


# Extract graph data from input text
async def extract_graph_data(text):
    """
    Asynchronously extracts graph data from input text using a graph transformer.

    Args:
        text (str): Input text to be processed into graph format.

    Returns:
        list: A list of GraphDocument objects containing nodes and relationships.
    """
    documents = [Document(page_content=text)]
    graph_documents = await graph_transformer.aconvert_to_graph_documents(documents)
    return graph_documents


def visualize_graph(graph_documents):
    """
    Visualizes a knowledge graph using PyVis based on the extracted graph documents.

    Args:
        graph_documents (list): A list of GraphDocument objects with nodes and relationships.

    Returns:
        pyvis.network.Network: The visualized network graph object.
    """
    # Create network
    net = Network(height="1200px", width="100%", directed=True,
                      notebook=False, bgcolor="#222222", font_color="white", filter_menu=True, cdn_resources='remote') 

    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships

    # Build lookup for valid nodes
    node_dict = {node.id: node for node in nodes}
    
    # Filter out invalid edges and collect valid node IDs
    valid_edges = []
    valid_node_ids = set()
    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)
            valid_node_ids.update([rel.source.id, rel.target.id])

    # Track which nodes are part of any relationship
    connected_node_ids = set()
    for rel in relationships:
        connected_node_ids.add(rel.source.id)
        connected_node_ids.add(rel.target.id)

    # Add valid nodes to the graph
    for node_id in valid_node_ids:
        node = node_dict[node_id]
        try:
            net.add_node(node.id, label=node.id, title=node.type, group=node.type)
        except:
            continue  # Skip node if error occurs

    # Add valid edges to the graph
    for rel in valid_edges:
        try:
            net.add_edge(rel.source.id, rel.target.id, label=rel.type.lower())
        except:
            continue  # Skip edge if error occurs

    # Configure graph layout and physics
    net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
        }
    """)

    output_file = "knowledge_graph.html"
    try:
        net.save_graph(output_file)
        print(f"Graph saved to {os.path.abspath(output_file)}")
        return net
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None


def export_graph_to_json(graph_documents):
    """
    Exports graph data to JSON format.

    Args:
        graph_documents (list): A list of GraphDocument objects with nodes and relationships.

    Returns:
        str: JSON string containing nodes and relationships.
    """
    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships

    node_dict = {node.id: node for node in nodes}

    # Filter valid edges
    valid_edges = []
    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)

    # Build JSON structure
    graph_data = {
        "nodes": [
            {
                "id": node.id,
                "type": node.type
            }
            for node in nodes
        ],
        "relationships": [
            {
                "source": rel.source.id,
                "target": rel.target.id,
                "type": rel.type
            }
            for rel in valid_edges
        ]
    }

    return json.dumps(graph_data, ensure_ascii=False, indent=2)


def export_graph_to_csv(graph_documents):
    """
    Exports graph relationships to CSV format (edge list).

    Args:
        graph_documents (list): A list of GraphDocument objects with nodes and relationships.

    Returns:
        str: CSV string with source, target, and relationship type columns.
    """
    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships

    node_dict = {node.id: node for node in nodes}

    # Filter valid edges
    valid_edges = []
    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)

    # Build CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["source", "target", "relationship_type"])

    for rel in valid_edges:
        writer.writerow([rel.source.id, rel.target.id, rel.type])

    return output.getvalue()


def generate_knowledge_graph(text):
    """
    Generates and visualizes a knowledge graph from input text.

    This function runs the graph extraction asynchronously and then visualizes
    the resulting graph using PyVis.

    Args:
        text (str): Input text to convert into a knowledge graph.

    Returns:
        tuple: (pyvis.network.Network, list) - The visualized network graph object and graph_documents.
    """
    graph_documents = asyncio.run(extract_graph_data(text))
    net = visualize_graph(graph_documents)
    return net, graph_documents