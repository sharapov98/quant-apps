# Abdul Sharopov's final exam

from db_mod import DataProcess, Plotter
import argparse
from scipy import stats

parser = argparse.ArgumentParser(description='optimizer')
def enable_parsing(parser):

    parser.add_argument("--file", help="Time series in csv",
                        required=True)
    parser.add_argument("--dollar_size", help="Dollar bar size",
                        type = int, required=True)
    return parser

parser = enable_parsing(parser)
args = parser.parse_args()

FILE = args.file
D_VAL = args.dollar_size

proc = DataProcess(FILE, D_VAL)
data = proc.import_csv()
db = proc.dollar_bar_df()

line_chart = Plotter(data, db, D_VAL)
line_chart.plot_db()
line_chart.plot_counts()

jarque_bera_orig = stats.jarque_bera(data["Close"])
print("Jarque-Bera test statistic for the original bar: ", jarque_bera_orig.statistic)
jarque_bera_dollar = stats.jarque_bera(db["Close"])
print("Jarque-Bera test statistic for the dollar bar: ", jarque_bera_dollar.statistic)
