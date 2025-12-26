from dash import Dash, html, dcc
import dash


app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True
)


# from src.dashboard.pages import cards_space

# cards_space.register_callbacks(app)

app.layout = html.Div(
    [
        html.H1("Empyree – Exploration des embeddings", style={"textAlign": "center"}),

        html.Nav(
            [
                dcc.Link("🃏 Espace des cartes", href="/cards-space", style={"marginRight": "20px"}),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),

        dash.page_container,  # conteneur pour le contenu de la page active
    ]
)

# --- Lancement du serveur ---
if __name__ == "__main__":
    app.run(debug=True)