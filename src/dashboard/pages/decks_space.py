from dash import dcc, html, register_page, callback
from dash.dependencies import Input, Output, State
import plotly.express as px

from src.dashboard.data.deck_data import load_decks
from src.dashboard.data.card_data import load_cards

register_page(__name__, path="/decks-space", name="Espace des decks")

HERO_COLORS = {
    # Axiom
    "Sierra": "#D7B899",
    "Treyst": "#A47148",
    "Subhash": "#6F3B1F",
    "Isaree": "#3B2416",

    # Bravos
    "Kojo": "#E53935",
    "Basira": "#D32F2F",
    "Atsadi": "#B71C1C",
    "Sol": "#7F0000",

    # Lyra
    "Nevenka": "#F8BBD0",
    "Fen": "#F06292",
    "Auraq": "#EC407A",
    "Nadir": "#AD1457",

    # Muna
    "Teija": "#81C784",
    "Rin": "#43A047",
    "Arjun": "#2E7D32",
    "Kauri": "#1B5E20",

    # Ordis
    "Sigismar": "#4FC3F7",
    "Gulrang": "#1E88E5",
    "Waru": "#006064",
    "Zhen": "#0D1B2A",

    # Yzmir
    "Akesha": "#CE93D8",
    "Afanas": "#8E24AA",
    "Lindiwe": "#6A1B9A",
    "Moyo": "#3E1B47",
}


CLUSTER_SYMBOLS = ["circle", "square", "diamond", "cross", "x"]

def build_cluster_symbol_map(df):
    """
    Associe un symbole par cluster intra-héros.
    Les clusters sont déjà uniques (ex: SIG1, SIG2, …)
    """
    symbol_map = {}
    clusters = sorted(df["cluster"].dropna().unique())

    for i, cluster in enumerate(clusters):
        symbol_map[cluster] = CLUSTER_SYMBOLS[i % len(CLUSTER_SYMBOLS)]

    return symbol_map


layout = html.Div(
    [
        html.H2("Espace des decks"),

        html.Div(
            [
                html.Label("Projection :"),
                dcc.RadioItems(
                    id="decks-umap-dim",
                    options=[
                        {"label": "2D", "value": 2},
                        {"label": "3D", "value": 3},
                    ],
                    value=2,
                    inline=True,
                ),
            ],
            style={"marginBottom": "10px"},
        ),

        html.Div([
            html.Label("Faction :"),
            dcc.Dropdown(
                id="decks-faction-dropdown",
                placeholder="Toutes les factions",
                clearable=True,
            ),
        ], style={"width": "250px", "marginBottom": "10px"}),

        html.Div([
            html.Label("Héros :"),
            dcc.Dropdown(
                id="decks-hero-dropdown",
                placeholder="Tous les héros",
                clearable=True,
            ),
        ], style={"width": "250px", "marginBottom": "15px"}),

        dcc.Checklist(
            id="show-clusters-toggle",
            options=[{"label": "Afficher les clusters", "value": "show"}],
            value=[],
            inline=True,
            style={"marginBottom": "10px"},
        ),

        dcc.Checklist(
            id="show-cards-toggle",
            options=[{"label": "Afficher les cartes", "value": "show"}],
            value=[],
            inline=True,
            style={"marginBottom": "10px"}
        ),

        dcc.Graph(id="decks-umap-graph"),
    ]
)

# -------------------------------------------------------------------
# Callback : options du dropdown faction
# -------------------------------------------------------------------

@callback(
    Output("decks-faction-dropdown", "options"),
    Input("decks-umap-dim", "value"),
)
def update_faction_options(dim):
    df = load_decks(dim)
    factions = sorted(df["faction"].dropna().unique())
    return [{"label": f, "value": f} for f in factions]

# -------------------------------------------------------------------
# Callback : options du dropdown héros
# -------------------------------------------------------------------

@callback(
    Output("decks-hero-dropdown", "options"),
    Input("decks-umap-dim", "value"),
    Input("decks-faction-dropdown", "value"),
)
def update_hero_options(dim, selected_faction):
    df = load_decks(dim)

    if selected_faction:
        df = df[df["faction"] == selected_faction]

    heroes = sorted(df["hero_name"].dropna().unique())
    return [{"label": h, "value": h} for h in heroes]

# -------------------------------------------------------------------
# Callback : Héros invalide
# -------------------------------------------------------------------

@callback(
    Output("decks-hero-dropdown", "value"),
    Input("decks-hero-dropdown", "options"),
    State("decks-hero-dropdown", "value"),
)
def reset_invalid_hero(hero_options, selected_hero):
    valid_values = {opt["value"] for opt in hero_options}
    if selected_hero in valid_values:
        return selected_hero
    return None

# -------------------------------------------------------------------
# Callback principal : figure
# -------------------------------------------------------------------

@callback(
    Output("decks-umap-graph", "figure"),
    Input("decks-umap-dim", "value"),
    Input("decks-faction-dropdown", "value"),
    Input("decks-hero-dropdown", "value"),
    Input("show-clusters-toggle", "value"),
    Input("show-cards-toggle", "value"),
)
def update_graph(dim, selected_faction, selected_hero, show_clusters, show_cards):

    # --------------------
    # Load & filter decks
    # --------------------
    df = load_decks(dim)

    if selected_faction:
        df = df[df["faction"] == selected_faction]

    if selected_hero:
        df = df[df["hero_name"] == selected_hero]

    show_clusters = "show" in show_clusters
    show_cards = "show" in show_cards

    scatter_fn = px.scatter if dim == 2 else px.scatter_3d

    # --------------------
    # Base scatter (DECKS)
    # --------------------
    scatter_kwargs = dict(
        data_frame=df,
        x="x",
        y="y",
        color="hero_name",
        color_discrete_map=HERO_COLORS,
        hover_name="hero_name",
        title="Espace des decks",
    )

    if dim == 3:
        scatter_kwargs["z"] = "z"

    if show_clusters:
        scatter_kwargs.update(
            symbol="cluster",
            symbol_map=build_cluster_symbol_map(df),
            hover_name="cluster",
            title="Espace des decks — clusters intra-héros",
        )

    fig = scatter_fn(**scatter_kwargs)

    fig.update_traces(marker=dict(size=6, opacity=0.85))

    # --------------------
    # Optional CARDS layer
    # --------------------
    if show_cards:
        cards_df = load_cards(dim)

        if selected_faction:
            cards_df = cards_df[cards_df["faction"] == selected_faction]

        if selected_hero:
            cards_df = cards_df[cards_df["hero_name"] == selected_hero]

        fig.add_scatter(
            x=cards_df["x"],
            y=cards_df["y"],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=4,
                color="rgba(120,120,120,0.5)",
            ),
            name="Cartes",
            hovertext=cards_df["nom"],
            hoverinfo="text",
            showlegend=True,
            **({"z": cards_df["z"]} if dim == 3 else {}),
        )

    fig.update_layout(
        height=700,
        legend_title_text="Héros",
    )

    return fig