import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

plt.rcParams['figure.figsize'] = [12, 8]


def read_file(filename: Path, columns):
    return pd.read_csv(filename)[columns]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots either one csv or comparison of two data series.')
    parser.add_argument('-b' '--base', required=True, help='A path to a csv file containing data series.',
                        type=str, dest='base')
    parser.add_argument('-e', '--experiment', required=False,
                        help='A path to a csv file to compare with a base file.', type=str, dest='experiment')
    parser.add_argument('-o', '--output', help='An output file path.', required=True)
    parser.add_argument('-c', '--columns', required=True, nargs='+',
                        help='Specify columns to plot, can be multiple',
                        dest='columns')
    parser.add_argument('-x', '--x-label', default='epoch', help='The name for x axis', dest='x_label')
    parser.add_argument('-y', '--y-label', default='accuracy', help='The name for y axis', dest='y_label')
    args = parser.parse_args()

    base_data = read_file(Path(args.base), args.columns)

    if args.experiment:
        experiment_data = read_file(Path(args.experiment), args.columns)

        for column_name in args.columns:
            experiment_data = experiment_data.rename(columns={column_name: column_name + ' experiment'})
            base_data = base_data.rename(columns={column_name: column_name + ' base'})

        data = pd.concat([base_data, experiment_data], axis=1)
        ax = data.plot()
    else:
        ax = base_data.plot()

    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.legend(loc='upper left')
    ax.get_figure().savefig(args.output)
