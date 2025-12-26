from dash import Dash, html, dcc
import dash

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        html.H1("Empyree – Exploration des embeddings", style={"textAlign": "center"}),

        html.Nav(
            [
                dcc.Link("🃏 Espace des cartes", href="/cards", style={"marginRight": "20px"}),
                # dcc.Link("🧩 Espace des decks", href="/decks"),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),

        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)