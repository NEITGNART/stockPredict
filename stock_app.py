import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from data_processing import get_dataset_df
from plot_graph import render_tab

app = dash.Dash()
server = app.server

coins = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "ada": "ADA-USD",
}


def predict_stock(coin_name):
    dataset, new_data = get_dataset_df(coin_name)
    q_80 = int(len(dataset) * 0.8)
    q_90 = int(len(dataset) * 0.9)

    train = dataset[0:q_80, :]
    valid = dataset[q_80: q_90, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = load_model("{}.h5".format(coin_name))

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    x_test = []
    for i in range(60, inputs.shape[0]):
        x_test.append(inputs[i - 60:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    closing_price = model.predict(x_test)
    closing_price = scaler.inverse_transform(closing_price)

    train = new_data[:q_80]
    valid = new_data[q_80: q_90]
    valid['Predictions'] = closing_price
    return train, valid


btc_train, btc_valid = predict_stock(coins["btc"])
eth_train, eth_valid = predict_stock(coins["eth"])
ada_train, ada_valid = predict_stock(coins["ada"])

app.layout = html.Div([
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs1", children=[
        render_tab(btc_train.index, btc_valid["Close"], btc_valid.index, btc_valid["Predictions"], label=coins["btc"]),
        render_tab(eth_train.index, eth_valid["Close"], eth_train.index, eth_valid["Predictions"], label=coins["eth"]),
        render_tab(ada_train.index, ada_valid["Close"], ada_train.index, ada_valid["Predictions"], label=coins["ada"]),
    ]),

])

if __name__ == '__main__':
    app.run_server(debug=True)
