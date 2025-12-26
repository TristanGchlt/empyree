from dash import dcc, html, register_page, callback
from dash.dependencies import Input, Output
import plotly.express as px
from matplotlib import colors as mcolors
import pandas as pd

from src.dashboard.data.deck_data import load_decks

register_page(__name__, path="/decks-space", name="Espace des decks")

# Couleurs de base par faction
FACTION_COLORS = {
    "Axiom": "#8B5A2B",
    "Bravos": "#C0392B",
    "Lyra": "#E056FD",
    "Muna": "#27AE60",
    "Ordis": "#2980B9",
    "Yzmir": "#8E44AD",
}

layout = html.Div([
    html.H2("Espace des decks"),

    html.Div([
        html.Label("Projection :"),
        dcc.RadioItems(
            id="decks-tsne-dim",
            options=[{"label": "2D", "value": 2}, {"label": "3D", "value": 3}],
            value=2,
            inline=True
        ),
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Label("Choisir une faction :"),
        dcc.Dropdown(
            id="decks-faction-dropdown",
            placeholder="Toutes factions"
        ),
    ], style={"width": "300px", "marginBottom": "15px"}),

    dcc.Graph(id="decks-tsne-graph")
])

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def make_shades(base_color: str, n: int) -> list[str]:
    """
    Génère n nuances d'une couleur de base, du clair au foncé.
    n peut être indéterminé.
    """
    if n == 1:
        return [base_color]
    
    rgb = mcolors.to_rgb(base_color)
    min_factor = 0.6  # plus clair
    max_factor = 1.0  # plus foncé (la couleur de base)
    
    shades = []
    for i in range(n):
        factor = min_factor + (max_factor - min_factor) * i / (n - 1)
        shade = tuple(min(1, c * factor) for c in rgb)
        shades.append(mcolors.to_hex(shade))
    return shades

def assign_cluster_colors(df: pd.DataFrame, base_colors: dict) -> dict:
    """
    Crée un dictionnaire cluster -> couleur en fonction de la couleur de la faction.
    """
    color_map = {}
    for faction, base_color in base_colors.items():
        clusters = sorted(df.loc[df['faction'] == faction, 'cluster'].unique())
        n = len(clusters)
        if n == 0:
            continue
        shades = make_shades(base_color, n)  # renvoie une liste de couleurs
        for cluster, shade in zip(clusters, shades):
            color_map[cluster] = shade
    return color_map

@callback(
    Output("decks-faction-dropdown", "options"),
    Input("decks-tsne-dim", "value")
)
def update_faction_options(dim):
    df = load_decks(dim)
    factions = sorted(df["faction"].dropna().unique())
    return [{"label": f, "value": f} for f in factions]

@callback(
    Output("decks-tsne-graph", "figure"),
    Input("decks-tsne-dim", "value"),
    Input("decks-faction-dropdown", "value")
)
def update_graph(dim, selected_faction):
    df = load_decks(dim)
    if selected_faction:
        df = df[df["faction"] == selected_faction]

    # Assigner les couleurs cluster/faction
    color_map = assign_cluster_colors(df, FACTION_COLORS)

    if dim == 2:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            color_discrete_map=color_map,
            hover_name="deck_id",
            title="t-SNE 2D des decks"
        )
    else:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="cluster",
            color_discrete_map=color_map,
            hover_name="deck_id",
            title="t-SNE 3D des decks"
        )

    fig.update_traces(marker=dict(size=6, opacity=0.85))
    fig.update_layout(height=700)

    # Légende plus propre : regrouper par faction
    for faction, base_color in FACTION_COLORS.items():
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=base_color, size=8),
            name=faction
        )

    fig.update_traces(marker=dict(size=6, opacity=0.85))
    fig.update_layout(height=700, legend_title_text="Faction")
    return fig