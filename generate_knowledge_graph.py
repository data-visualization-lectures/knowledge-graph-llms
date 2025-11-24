from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from dotenv import load_dotenv
import os
import asyncio
import json
import csv
from io import StringIO


# Default prompt template for Japanese relationship extraction
DEFAULT_PROMPT_TEMPLATE = """以下のテキストから知識グラフの情報を抽出してください。
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


# Custom callback handler to log LLM interactions
class DebugCallbackHandler(BaseCallbackHandler):
    """Callback handler to log LLM inputs and outputs for debugging."""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when LLM starts processing."""
        print("\n" + "="*80)
        print("📤 LLMへのプロンプト送信")
        print("="*80)
        for i, prompt in enumerate(prompts, 1):
            print(f"\n【プロンプト {i}】")
            print(prompt)
        print("="*80)
    
    def on_llm_end(self, response, **kwargs):
        """Log when LLM finishes processing."""
        print("\n" + "="*80)
        print("📥 LLMからの応答")
        print("="*80)
        for i, generation in enumerate(response.generations, 1):
            for j, gen in enumerate(generation, 1):
                print(f"\n【応答 {i}-{j}】")
                print(gen.text)
        print("="*80 + "\n")


# Extract graph data from input text
async def extract_graph_data(text, graph_transformer, debug=False):
    """
    Asynchronously extracts graph data from input text using a graph transformer.

    Args:
        text (str): Input text to be processed into graph format.
        graph_transformer: LLMGraphTransformer instance to use for extraction.
        debug (bool): If True, returns debug information including raw LLM response.

    Returns:
        tuple: (graph_documents, debug_info) if debug=True, otherwise just graph_documents.
               debug_info is a dict containing raw LLM response and other debug data.
    """
    documents = [Document(page_content=text)]
    graph_documents = await graph_transformer.aconvert_to_graph_documents(documents)
    
    if debug:
        # Extract debug information
        debug_info = {
            "graph_documents": graph_documents,
            "num_nodes": len(graph_documents[0].nodes) if graph_documents else 0,
            "num_relationships": len(graph_documents[0].relationships) if graph_documents else 0,
            "nodes": [{"id": node.id, "type": node.type} for node in graph_documents[0].nodes] if graph_documents else [],
            "relationships": [
                {"source": rel.source.id, "target": rel.target.id, "type": rel.type} 
                for rel in graph_documents[0].relationships
            ] if graph_documents else []
        }
        return graph_documents, debug_info
    
    return graph_documents


def generate_cytoscape_elements(graph_documents):
    """
    Converts graph documents to Cytoscape.js elements JSON format.

    Args:
        graph_documents (list): A list of GraphDocument objects.

    Returns:
        list: A list of dictionaries representing nodes and edges for Cytoscape.js.
    """
    if not graph_documents:
        return []

    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships
    
    elements = []
    node_ids = set()

    import random

    # Add nodes
    for node in nodes:
        elements.append({
            "data": {
                "id": node.id,
                "label": node.id,
                "type": node.type
            },
            "position": {
                "x": random.uniform(100, 800),
                "y": random.uniform(100, 600)
            }
        })
        node_ids.add(node.id)

    # Add edges
    for rel in relationships:
        # Ensure both source and target nodes exist to avoid errors
        if rel.source.id in node_ids and rel.target.id in node_ids:
            elements.append({
                "data": {
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "label": rel.type
                }
            })
            
    return elements


def generate_cytoscape_html(elements):
    """
    Generates a complete HTML string with embedded Cytoscape.js graph.

    Args:
        elements (list): Cytoscape.js elements JSON.

    Returns:
        str: HTML string.
    """
    elements_json = json.dumps(elements, ensure_ascii=False)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <script src="https://unpkg.com/layout-base/layout-base.js"></script>
    <script src="https://unpkg.com/cose-base/cose-base.js"></script>
    <script src="https://unpkg.com/cytoscape-fcose/cytoscape-fcose.js"></script>
    <style>
        body {{
            font-family: sans-serif;
            margin: 0;
            padding: 0;
            background-color: #222222;
            color: #ffffff;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
        }}
        #cy {{
            width: 100%;
            height: 100%;
            display: block;
        }}
    </style>
</head>
<body>
    <div id="cy"></div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var elements = {elements_json};
            
            var cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: elements,
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'content': 'data(label)',
                            'font-size': '12px',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'background-color': '#555',
                            'text-outline-color': '#555',
                            'text-outline-width': '2px',
                            'color': '#fff',
                            'overlay-padding': '6px',
                            'z-index': '10',
                            'width': 'label',
                            'height': 'label',
                            'padding': '12px',
                            'shape': 'round-rectangle'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'curve-style': 'bezier',
                            'opacity': 0.8,
                            'line-color': '#bbb',
                            'width': 2,
                            'target-arrow-shape': 'triangle',
                            'target-arrow-color': '#bbb',
                            'label': 'data(label)',
                            'color': '#fff',
                            'font-size': '10px',
                            'text-rotation': '0',
                            'text-background-color': '#222222',
                            'text-background-opacity': 1,
                            'text-background-padding': '2px',
                            'text-background-shape': 'round-rectangle'
                        }}
                    }},
                    {{
                        selector: ':selected',
                        style: {{
                            'border-width': '6px',
                            'border-color': '#AAD8FF',
                            'border-opacity': '0.5',
                            'background-color': '#77828C',
                            'text-outline-color': '#77828C',
                            'line-color': '#AAD8FF',
                            'target-arrow-color': '#AAD8FF',
                            'source-arrow-color': '#AAD8FF'
                        }}
                    }}
                ],
                layout: {{
                    name: 'fcose',
                    quality: "default",
                    randomize: true,
                    animate: true,
                    animationDuration: 1000,
                    animationEasing: 'ease-in-out',
                    fit: true,
                    padding: 30,
                    nodeDimensionsIncludeLabels: true,
                    uniformNodeDimensions: false,
                    packComponents: true,
                    step: "all",
                    initialEnergyOnIncremental: 0.3,
                    samplingType: true,
                    sampleSize: 25,
                    nodeSeparation: 75,
                    piTol: 0.0000001,
                    nodeRepulsion: 4500,
                    idealEdgeLength: 50,
                    edgeElasticity: 0.45,
                    nestingFactor: 0.1,
                    gravity: 0.25,
                    numIter: 2500,
                    tile: true,
                    tilingPaddingVertical: 10,
                    tilingPaddingHorizontal: 10,
                    gravityRangeCompound: 1.5,
                    gravityCompound: 1.0,
                    gravityRange: 3.8,
                    initialEnergyOnIncremental: 0.3
                }}
            }});
        }});
    </script>
</body>
</html>
    """
    return html_content


def visualize_graph(graph_documents):
    """
    Visualizes a knowledge graph using Cytoscape.js based on the extracted graph documents.

    Args:
        graph_documents (list): A list of GraphDocument objects with nodes and relationships.

    Returns:
        object: A dummy object with a save_graph method to maintain compatibility, 
                or we can just handle the saving here.
    """
    # Generate Cytoscape elements
    elements = generate_cytoscape_elements(graph_documents)
    
    # Generate HTML
    html_content = generate_cytoscape_html(elements)
    
    # Define a helper class to mimic the PyVis Network object's save_graph method
    class CytoscapeGraph:
        def __init__(self, html_content):
            self.html_content = html_content
            
        def save_graph(self, filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.html_content)
            print(f"Graph saved to {os.path.abspath(filename)}")
            
    return CytoscapeGraph(html_content)


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


def generate_knowledge_graph(text, api_key=None, prompt_template=None):
    """
    Generates and visualizes a knowledge graph from input text.

    This function runs the graph extraction asynchronously and then visualizes
    the resulting graph using Cytoscape.js.

    Args:
        text (str): Input text to convert into a knowledge graph.
        api_key (str, optional): OpenAI API key. If not provided, reads from environment.
        prompt_template (str, optional): Custom prompt template. If not provided, uses DEFAULT_PROMPT_TEMPLATE.

    Returns:
        tuple: (CytoscapeGraph, list) - The graph object and graph_documents.
    """
    # APIキーの取得
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY が設定されていません。"
        )
    
    # LLMとtransformerの初期化（デバッグコールバック付き）
    debug_callback = DebugCallbackHandler()
    llm = ChatOpenAI(
        temperature=0, 
        model_name="gpt-4o", 
        api_key=api_key,
        callbacks=[debug_callback]
    )
    
    # Use custom prompt template or default
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_TEMPLATE
    
    # Create a custom prompt for Japanese relationship extraction
    japanese_prompt = PromptTemplate.from_template(prompt_template)
    
    graph_transformer = LLMGraphTransformer(llm=llm, prompt=japanese_prompt)
    
    # グラフデータの抽出と可視化
    print("\n" + "="*80)
    print("🚀 グラフデータ抽出を開始")
    print("="*80)
    graph_documents = asyncio.run(extract_graph_data(text, graph_transformer))
    
    # デバッグ情報をコンソールに出力
    print("\n" + "="*80)
    print("🔍 デバッグ情報: LLMから抽出されたグラフデータ")
    print("="*80)
    if graph_documents:
        print(f"📊 ノード数: {len(graph_documents[0].nodes)}")
        print(f"🔗 関係性数: {len(graph_documents[0].relationships)}")
        
        print("\n【ノード一覧】")
        for i, node in enumerate(graph_documents[0].nodes, 1):
            print(f"  {i}. ID: {node.id}, Type: {node.type}")
        
        print("\n【関係性一覧】")
        for i, rel in enumerate(graph_documents[0].relationships, 1):
            print(f"  {i}. {rel.source.id} --[{rel.type}]--> {rel.target.id}")
    else:
        print("⚠️ グラフデータが抽出されませんでした")
    print("="*80 + "\n")
    
    net = visualize_graph(graph_documents)
    return net, graph_documents