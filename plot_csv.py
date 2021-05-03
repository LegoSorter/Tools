import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

plt.rcParams['figure.figsize'] = [12, 8]


def read_file(filename: Path, columns):
    return pd.read_csv(filename)[columns]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots either one csv or comparison of many data series. '
                                                 'For each data series (input files) you have to provide corresponding '
                                                 'column and label. '
                                                 'For example: '
                                                 'python plot_csv.py '
                                                 '-i inputA inputB '
                                                 '-c colA colB '
                                                 '-l labelA labelB '
                                                 '-o ABplot.png')
    parser.add_argument('-i', '--input_series', help='Paths to csv files', required=True, nargs='+', dest='series')
    parser.add_argument('-l', '--labels', help='Output labels for each data series', required=False, nargs='+', dest='labels')
    parser.add_argument('-o', '--output', help='An output file path.', required=True)
    parser.add_argument('-c', '--columns', help='Specify columns to plot', required=True, nargs='+', dest='columns')
    parser.add_argument('-x', '--x-label', help='The name for x axis', default='x', dest='x_label')
    parser.add_argument('-y', '--y-label', help='The name for y axis', default='y', dest='y_label')
    parser.add_argument('-yl', '--y-limit', help='The limit for y ax', nargs='+', dest='ylim')
    args = parser.parse_args()

    data_to_plot = pd.DataFrame()

    plot_legend = args.labels

    if not args.labels:
        args.labels = args.series

    for input_file, label, column in zip(args.series, args.labels, args.columns):
        data_to_plot[label] = read_file(Path(input_file), column)

    ax = data_to_plot.plot(legend=plot_legend)
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    #ax.legend(loc='upper left')
    if args.ylim:
        plt.ylim(float(args.ylim[0]), float(args.ylim[1]))
    ax.get_figure().savefig(args.output)
