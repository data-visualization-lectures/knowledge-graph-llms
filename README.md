# Knowledge Graph Generator

  - LangChain + OpenAI でテキストから知識グラフを抽出し、Streamlit で可視化するアプリです（app.py）。サイドバーでテキスト入力または .txt アップロードを受け取り、「知識グラフを生成」押下でグラフを生成・表示します。
  - 抽出処理は generate_knowledge_graph.py で実装。ChatOpenAI（temperature 0、gpt-4o）を LLMGraphTransformer に渡し、日本語のカスタムプロンプトでエンティティと関係を抽出します。OPENAI_API_KEY を .env などで読み込みます。
  - 抽出結果（GraphDocument）を PyVis で可視化（向き付きグラフ、レイアウト設定あり）し、knowledge_graph.html に保存。Streamlit からその HTML を埋め込んで表示します。
  - 抽出済みグラフデータは JSON/CSV でダウンロード可能（export_graph_to_json / export_graph_to_csv in generate_knowledge_graph.py）。抽出時に存在しないノードを指すエッジは除外する簡易バリデーションも入っています。
  - 使い方：streamlit run app.py を実行 → 入力/アップロード → ボタンで生成 → グラフ表示・JSON/CSV ダウンロード。

  ## 日本語のカスタムプロンプト

   - 日本語カスタムプロンプトは generate_knowledge_graph.py 内の japanese_prompt で定義されています。内容は「入力テキストから知識グラフ情報を抽出し、エンティティ（人物・組織・場所など）とそれらの関係性を日本語で抽出する」指示です。関係性の例として「所属している」「友人である」「位置している」「質問する」「説明する」「関連している」を提示し、{input} にテキストを差し込み「エンティティと関係性を抽出してください」と締めています。

   ## AIモデル

 - 変更箇所は generate_knowledge_graph.py の ChatOpenAI(temperature=0, model_name="gpt-4o") です。この model_name を好みの OpenAI Chat モデル名に差し替えれば利用モデルを変えられます（例: gpt-4o-mini でコスト削減、gpt-4-turbo など）。
  - もし Azure OpenAI を使う場合は、ChatOpenAI に Azure 用の引数（azure_endpoint/api_version/azure_deployment など）を渡す形に変えます。
  - 異なるプロバイダ（Anthropic、Gemini など）を使いたい場合は LangChain の対応クラスに切り替える必要があります（例: ChatAnthropic など）。
  - モデル変更後は短いサンプルテキストでグラフ生成が問題なく動くかを試すのがおすすめです。


  ## Streamlit

- Streamlit 上で Python を書き、pyvis で生成した HTML を streamlit.components.v1.components.html でそのまま埋め込んでいるだけです。
  - 流れ: GPT でグラフ抽出→ PyVis が knowledge_graph.html を出力→ Streamlit がその HTML を読み込み iframe 的に表示。Python と HTML を“同時に”動かすのではなく、Python が生成した静的 HTML を埋め込んでいるイメージです。
 - そのまま Streamlit を使わない場合、Netlify/Vercel の “静的ホスティングだけ” では足りません。理由は、LLM呼び出しに API キーが必要で、ブラウザ側に埋め込むと漏洩するため、サーバー側の処理が必須だからです。
 - 実現パターン
          1. Vercel/Netlify Functions (サーバーレス) で Python/Node のエンドポイントを用意し、そこから OpenAI を呼び出し、JSON でノード/エッジを返す。フロントは静的 (HTML/JS) をホストし、JS で API を叩いてグラフを描画。
          2. フル JS に寄せるなら、LangChain JS + OpenAI (Node) をサーバーレス関数に置く。フロントは vis-network/cytoscape.js などで描画。
          3. Python サーバーを常時動かしたいなら、Render/Fly.io などの常駐系 PaaS が簡単（Vercel/Netlify は常駐プロセスが基本不可）。
  - 注意点
      - APIキーは必ずサーバー側で管理し、クライアントに埋め込まない。
      - コールドスタートが気になるなら Node 小さめの関数 + 短時間応答にするか、常駐 PaaS を検討。
      - グラフ描画はフロントの JS で完結するので、PyVis 生成をやめて JSON 返しにすればよい。
  - まとめ: Streamlit を外すなら「サーバーレス関数で LLM/グラフ抽出 → 静的フロントで可視化」の構成にすれば Netlify/Vercel でもデプロイ可能です。


  ## グラフ部分をJavaScriptにすることも簡単にできそう？

   - 仕組み的には簡単です。今は PyVis が HTML+JS を生成し、Streamlit がそれを埋め込んでいるだけなので、PyVis 部分を自前の JS 描画に置き換えれば OK です。
  - やり方の例
      2. もっとしっかりやるなら、Streamlit カスタムコンポーネント（React/JS）を作って、Python からデータを渡して描画。
  - 手間感
      - 既存の HTML 埋め込みを、外部 CDN の JS を使った簡易版に差し替えるだけなら数十行で動くはず。
      - レイアウトやインタラクションを細かくチューニングしたいなら、カスタムコンポーネント化が確実。
  - 次の一手の候補
      1. vis-network を CDN で読み込み、components.html に直接 JSON を渡して描画する最小実装を作る。
      2. cytoscape.js でスタイル・レイアウトを細かく制御する。
      3. React ベースのカスタムコンポーネント化で将来拡張に備える。