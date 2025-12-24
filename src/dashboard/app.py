from dash import Dash, dcc, html
import pandas as pd
import plotly.express as px

# Charger le dataset final
df = pd.read_csv("data/final/deck_dataset.csv")

# Exemple : scatter des deux premières dimensions des embeddings
fig = px.scatter(
    df,
    x="vector_0",
    y="vector_1",
    color="faction",
    hover_data=["deck_id", "cluster"]
)

# Créer l'app Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Card2Vec"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run(debug=True)