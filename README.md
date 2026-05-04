Embeddings Visualizer — compares how different transformer models place
sentences in their embedding space, projected to 3D via PCA.

Setup (one-time):
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
    pip install shiny sentence-transformers torch numpy scikit-learn plotly einops accelerate

Run:
    shiny run --reload app.py
