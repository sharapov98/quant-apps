import pandas as pd
import matplotlib.pyplot as plt

class DataProcess():
    def __init__(self, data, d_value):
        self.data = data
        self.d_value = d_value
    # Names column in the dataframe
    dv_col = "Dollar Volume"

    def import_csv(self):
        self.series=pd.read_csv(self.data, index_col='Time', parse_dates=['Time'])
        self.series[self.dv_col] = self.series["VWAP"] * self.series["Volume"]
        return self.series

    # Create and cut dollar bars
    def dollar_bars(self, dv_col, d_value):
        t = self.series[self.dv_col]
        ts = 0
        idx = []
        for i, x in enumerate(t):
            ts += x
            if ts >= self.d_value:
                idx.append(i)
                ts = 0
                continue
        return idx

    def dollar_bar_df(self):
        idx = self.dollar_bars(self.series, self.d_value)
        return self.series.iloc[idx].drop_duplicates()


class Plotter():
    def __init__(self, series, db, d_value):
        self.series = series
        self.db = db
        self.d_value = d_value

    # Plot counts of dollar and tick bars
    def plot_counts(self):
        plt.figure(figsize=(12,5))
        plt.plot(self.db.groupby(pd.Grouper(freq='1W'))['Close'].count(), label="Dollar Bars" )
        plt.plot(self.series.groupby(pd.Grouper(freq='1W'))['Close'].count(), label="Tick Bars")
        plt.title('Weekly bar count')
        plt.legend()
        plt.show()

    # Plot dollar bar
    def plot_db(self):
        plt.figure(figsize=(12,5))
        plt.plot(self.series[DataProcess.dv_col].divide(self.d_value))
        plt.title('Dollar bar')
        plt.show()
