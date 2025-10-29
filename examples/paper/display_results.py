import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', required=True)
args = parser.parse_args()

experiment_name = args.experiment_name

results_df = pd.read_csv('results.csv')

res = results_df.groupby(['experiment_name', 'impl', 'volume_size'])['time'].mean()

latex_table = res.to_latex('latex')