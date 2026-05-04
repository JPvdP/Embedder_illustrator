"""
Embeddings Visualizer — compares how different transformer models place
sentences in their embedding space, projected to 3D via PCA.

Setup (one-time):
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
    pip install shiny sentence-transformers torch numpy scikit-learn plotly einops accelerate

Run:
    shiny run --reload app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# --- Models to compare --------------------------------------------------------
# Two families:
#   * Classic encoder models (BERT-family, ~100M params): fast, small.
#   * LLM-based embedders: decoder LLMs (Qwen2, Mistral) finetuned for
#     sentence similarity. Larger, slower, top of MTEB.
MODELS: dict[str, dict] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "label": "MiniLM (encoder, 22M)",
        "trust_remote_code": False,
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "label": "MPNet (encoder, 110M)",
        "trust_remote_code": False,
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "label": "Multilingual MiniLM (encoder, 118M)",
        "trust_remote_code": False,
    },
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "label": "GTE-Qwen2 1.5B (LLM-based, ~3 GB, CPU OK)",
        "trust_remote_code": True,
    },
    # Uncomment if you have a GPU or a lot of patience:
    # "Alibaba-NLP/gte-Qwen2-7B-instruct": {
    #     "label": "GTE-Qwen2 7B (LLM-based, ~14 GB, GPU recommended)",
    #     "trust_remote_code": True,
    # },
    # "intfloat/e5-mistral-7b-instruct": {
    #     "label": "E5-Mistral 7B (LLM-based, ~14 GB, GPU recommended)",
    #     "trust_remote_code": False,
    # },
}

CHOICES = {model_id: cfg["label"] for model_id, cfg in MODELS.items()}

# --- Lazy model cache ---------------------------------------------------------
_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_id: str) -> SentenceTransformer:
    if model_id not in _model_cache:
        print(f"Loading {model_id} ... (first time may download several GB)",
              flush=True)
        cfg = MODELS[model_id]
        _model_cache[model_id] = SentenceTransformer(
            model_id,
            trust_remote_code=cfg["trust_remote_code"],
        )
    return _model_cache[model_id]


def embed_sentences(sentences: list[str], model_id: str) -> np.ndarray:
    return get_model(model_id).encode(sentences, convert_to_numpy=True)


def project_3d(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (3D coords, variance explained per PC) or None if too few rows."""
    if emb.shape[0] < 4:
        return None
    pca = PCA(n_components=3)
    coords = pca.fit_transform(emb)
    return coords, pca.explained_variance_ratio_


def make_plot(coords: np.ndarray, sentences: list[str], title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers+text",
                text=sentences,
                textposition="top center",
                hovertext=sentences,
                hoverinfo="text",
                marker=dict(size=6),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    return fig


# --- UI -----------------------------------------------------------------------
DEFAULT_SENTENCES = "\n".join([
    "The bank approved my loan.",
    "I sat by the river bank.",
    "The dog chased the ball.",
    "A puppy ran after the toy.",
    "Quantum physics is hard.",
    "I love eating pizza.",
])

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("Sentences"),
        ui.help_text("One sentence per line. At least 4 needed for 3D PCA."),
        ui.input_text_area(
            "sentences", label=None, value=DEFAULT_SENTENCES, rows=12,
            width="100%",
        ),
        ui.input_checkbox_group(
            "models",
            "Models to compare",
            choices=CHOICES,
            selected=list(CHOICES.keys())[:2],
        ),
        ui.input_action_button("go", "Embed and project", class_="btn-primary"),
        ui.hr(),
        ui.help_text(
            ui.tags.small(
                "What you see is a 3D PCA projection of high-dimensional ",
                "(384–1536D) embeddings. Compare ", ui.tags.em("relative"),
                " positions across models, not absolute coordinates — each ",
                "model has its own axes.",
            )
        ),
        width=380,
    ),
    ui.output_ui("plots"),
    title="Sentence Embeddings: Model Comparison",
)


# --- Server -------------------------------------------------------------------
def server(input: Inputs, output: Outputs, session: Session):

    @reactive.calc
    @reactive.event(input.go)
    def results():
        selected = input.models()
        if not selected:
            return {}

        sentences = [s.strip() for s in input.sentences().splitlines()]
        sentences = [s for s in sentences if s]

        if len(sentences) < 4:
            ui.notification_show(
                "Please enter at least 4 non-empty sentences.",
                type="warning", duration=5,
            )
            return {}

        out = {}
        with ui.Progress(min=0, max=len(selected)) as p:
            for i, model_id in enumerate(selected):
                p.set(i, message="Embedding...",
                      detail=MODELS[model_id]["label"])
                emb = embed_sentences(sentences, model_id)
                proj = project_3d(emb)
                out[model_id] = {
                    "sentences": sentences,
                    "coords": proj[0] if proj else None,
                    "var_explained": proj[1] if proj else None,
                    "dim": emb.shape[1],
                }
        return out

    @render.ui
    def plots():
        res = results()
        if not res:
            return ui.div()

        cards = []
        for i, (model_id, data) in enumerate(res.items()):
            label = MODELS[model_id]["label"]
            var_pct = 100 * float(np.sum(data["var_explained"]))
            header = ui.tags.div(
                ui.tags.strong(label),
                ui.tags.small(
                    f" — {data['dim']}D embeddings, "
                    f"PC1–3 explain {var_pct:.1f}% of variance"
                ),
            )
            cards.append(
                ui.card(
                    ui.card_header(header),
                    output_widget(f"plot_{i}"),
                )
            )

        # Two cards per row.
        return ui.layout_columns(*cards, col_widths=[6] * len(cards))

    # Pre-register render functions for up to N model slots. Shiny needs
    # these declared at server-init time, not dynamically inside `plots`.
    # We register the maximum possible (= number of available models).
    def make_renderer(slot_idx: int):
        @render_widget
        def _plot():
            res = results()
            items = list(res.items())
            if slot_idx >= len(items):
                return go.Figure()
            model_id, data = items[slot_idx]
            return make_plot(
                data["coords"], data["sentences"], MODELS[model_id]["label"]
            )
        return _plot

    for i in range(len(MODELS)):
        output(make_renderer(i), id=f"plot_{i}")


app = App(app_ui, server)
