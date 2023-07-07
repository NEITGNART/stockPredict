import pandas as pd


def get_dataset_df(coin_name):
    df_nse = pd.read_csv("{}.csv".format(coin_name))
    df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
    df_nse.index = df_nse['Date']

    data = df_nse.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data["Date"][i] = data['Date'][i]
        new_data["Close"][i] = data["Close"][i]

    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)
    return new_data.values, new_data

