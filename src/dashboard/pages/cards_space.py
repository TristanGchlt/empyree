from dash import dcc, html, register_page, callback
from dash.dependencies import Input, Output
import plotly.express as px

from src.dashboard.data.card_data import load_cards

register_page(__name__, path="/cards-space", name="Espace des cartes")

cards_df = load_cards()

layout = html.Div([
    html.H2("Espace des cartes"),

    html.Label("Choisir une faction :"),
    dcc.Dropdown(
        id="cards-faction-dropdown",
        options=[{"label": f, "value": f} for f in sorted(cards_df["faction"].unique())],
        value=None,
        multi=False,
        placeholder="Toutes factions"
    ),
    dcc.Graph(id="cards-tsne-graph")
])

FACTION_COLORS = {
    "Axiom": "#8B5A2B",   # marron
    "Bravos": "#C0392B",  # rouge
    "Lyra": "#E056FD",    # rose
    "Muna": "#27AE60",    # vert
    "Ordis": "#2980B9",   # bleu
    "Yzmir": "#8E44AD",   # violet
}

CARD_TYPE_SYMBOLS = {
    "Héros": "star",
    "Personnage": "circle",
    "Jeton Personnage": "circle",
    "Sort": "cross",
    "Permanent d’Expédition": "square",
    "Repère Permanent": "diamond",
    "Jeton Repère Permanent": "diamond"
}

@callback(
    Output("cards-tsne-graph", "figure"),
    Input("cards-faction-dropdown", "value")
)
def update_graph(selected_faction):
    df = cards_df if not selected_faction else cards_df[cards_df["faction"] == selected_faction]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="faction",
        symbol="type",
        symbol_map=CARD_TYPE_SYMBOLS,
        color_discrete_map=FACTION_COLORS,
        hover_name="nom",
        title="Représentation t-SNE des cartes"
    )
    fig.update_traces(
        marker=dict(size=7, opacity=0.85)
    )
    fig.update_layout(height=700)
    return fig