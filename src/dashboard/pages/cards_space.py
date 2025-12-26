from dash import dcc, html, register_page, callback
from dash.dependencies import Input, Output
import plotly.express as px

FACTION_COLORS = {
    "Axiom": "#8B5A2B",   # marron
    "Bravos": "#C0392B",  # rouge
    "Lyra": "#E056FD",    # rose
    "Muna": "#27AE60",    # vert
    "Ordis": "#2980B9",   # bleu
    "Yzmir": "#8E44AD",   # violet
}

CARD_TYPE_SYMBOLS = {
    "Héros": "circle-open",
    "Personnage": "circle",
    "Sort": "cross",
    "Permanent d’Expédition": "square",
    "Repère Permanent": "diamond",
    "Permanent": "diamond"
}

from src.dashboard.data.card_data import load_cards

register_page(__name__, path="/cards-space", name="Espace des cartes")


layout = html.Div([
    html.H2("Espace des cartes"),

    html.Div([
        html.Label("Projection :"),
        dcc.RadioItems(
            id="cards-tsne-dim",
            options=[
                {"label": "2D", "value": 2},
                {"label": "3D", "value": 3},
            ],
            value=2,
            inline=True
        ),
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Label("Choisir une faction :"),
        dcc.Dropdown(
            id="cards-faction-dropdown",
            placeholder="Toutes factions"
        ),
    ], style={"width": "300px", "marginBottom": "15px"}),

    dcc.Graph(id="cards-tsne-graph")
])



@callback(
    Output("cards-faction-dropdown", "options"),
    Input("cards-tsne-dim", "value")
)
def update_faction_options(dim):
    df = load_cards(dim)
    factions = sorted(df["faction"].dropna().unique())
    return [{"label": f, "value": f} for f in factions]


@callback(
    Output("cards-tsne-graph", "figure"),
    Input("cards-tsne-dim", "value"),
    Input("cards-faction-dropdown", "value"),
)
def update_graph(dim, selected_faction):
    df = load_cards(dim)

    if selected_faction:
        df = df[df["faction"] == selected_faction]

    if dim == 2:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="faction",
            symbol="type",
            color_discrete_map=FACTION_COLORS,
            symbol_map=CARD_TYPE_SYMBOLS,
            hover_name="nom",
            title="t-SNE 2D des cartes"
        )
    else:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="faction",
            symbol="type",
            color_discrete_map=FACTION_COLORS,
            symbol_map=CARD_TYPE_SYMBOLS,
            hover_name="nom",
            title="t-SNE 3D des cartes"
        )

    fig.update_traces(marker=dict(size=6, opacity=0.85))
    fig.update_layout(height=700)

    return fig