from db_mod import Plotter, DataProcess

proc = DataProcess("e_mini_sp_series.csv", 18000000)
data = proc.import_csv()
db = proc.dollar_bar_df()
line_chart = Plotter(data, db, D_VAL)

def test_plot():
    assert dollar_plot.get_axes() == []
    dollar_plot = line_chart.plot_db()
    count_plot = line_chart.plot_counts()
    assert count_plot.get_axes() != []
    assert count_plot.get_axes() != []

def test_jarque_bera():
    jarque_bera_orig = stats.jarque_bera(data["Close"])
    jarque_bera_dollar = stats.jarque_bera(db["Close"])
    assert jarque_bera_orig.statistic > jarque_bera_dollar