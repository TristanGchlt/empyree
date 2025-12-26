import dash
from dash import html, dcc, Input, Output
import plotly.express as px

# ---------- Page registration ----------
dash.register_page(
    __name__,
    path="/cards",
    name="Espace des cartes",
)

# ---------- Data loading ----------
from src.dashboard.data.card_data import load_cards

df_cards = load_cards()

# simple projection (pour proto)
df_cards["x"] = df_cards[1]
df_cards["y"] = df_cards[2]

# ---------- Layout ----------
layout = html.Div(
    [
        html.H2("Espace des cartes"),

        html.Div(
            [
                html.Label("Faction"),
                dcc.Dropdown(
                    options=[
                        {"label": f, "value": f}
                        for f in sorted(df_cards["faction"].unique())
                    ],
                    value=df_cards["faction"].unique().tolist(),
                    multi=True,
                    id="cards-faction-filter",
                ),
            ],
            style={"width": "300px", "marginBottom": "20px"},
        ),

        dcc.Graph(id="cards-scatter"),
    ],
    style={"padding": "20px"},
)

# ---------- Callbacks ----------
@dash.callback(
    Output("cards-scatter", "figure"),
    Input("cards-faction-filter", "value"),
)
def update_cards_scatter(selected_factions):
    filtered = df_cards[df_cards["faction"].isin(selected_factions)]

    fig = px.scatter(
        filtered,
        x="x",
        y="y",
        color="Faction",
        hover_data=["Nom", "Type"],
        title="Projection 2D des cartes (card2vec)",
    )

    fig.update_layout(
        height=700,
        legend_title_text="Faction",
    )

    return fig