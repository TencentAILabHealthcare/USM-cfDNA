# Copyright (c) 2025, Tencent Inc. All rights reserved.
import sys

sys.path.append(".")

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from compute_pon import CHROM_ARMS
from tgnn.utils import jload

def visualize_cna(cnv_file, output, pon_file=None, threshold=2):
    """
    Visualize Copy Number Alterations (CNA) results.

    Args:
        cnv_file: Path to the input JSON file containing CNA results.
        output: Path to save the CNA visualization.
    """
    cna = jload(cnv_file)
    bin_df = pd.DataFrame(
        sorted(cna['bin'].items(),
               key=lambda x: (int(x[0].split(":")[0].replace('chr', '')), int(x[0].split(":")[1].split("-")[0]))),
        columns=['Bin', 'Z-Score'])

    bin_df['chrom'] = list(map(lambda x: x.split(":")[0], bin_df.iloc[:, 0]))
    bin_df['start'] = list(map(lambda x: int(x.split(":")[1].split("-")[0]), bin_df.iloc[:, 0]))
    bin_df['end'] = list(map(lambda x: int(x.split(":")[1].split("-")[1]), bin_df.iloc[:, 0]))
    bin_df['chrom_arm'] = bin_df['chrom'] + bin_df.apply(
        lambda row: 'p' if row['end'] <= CHROM_ARMS[row['chrom'].replace('chr', '')]['p'][1] else 'q' if row['start'] >= CHROM_ARMS[row['chrom'].replace('chr', '')]['q'][0] else None,
        axis=1)
    bin_df.columns = ['interval', 'score', 'chrom', 'start', 'end', 'chrom_arm']

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    bin_level_cna = bin_df.iloc[:, 1].to_numpy().flatten()
    for chrom, score in cna['arm'].items():
        x = np.argwhere(bin_df['chrom_arm'] == chrom).flatten()
        axes[0].plot(
            x,
            [score] * len(x),
            color='red' if score > threshold else 'blue' if score < -threshold else 'gray'
        )

    # Add x-axis labels for chromosome arms
    x_ticks = []
    x_labels = []
    for i, (chrom, group) in enumerate(bin_df.groupby("chrom_arm")):
        x_ticks.append(group.index[int(group.shape[0] / 2)])  # Take the middle index of the group
        x_labels.append(chrom)

    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels, rotation=45)

    axes[0].set_ylabel("Z-scored")
    axes[0].set_title("Arm-level CNA Features")

    axes[1].scatter(
        np.arange(len(bin_level_cna)),
        bin_level_cna,
        s=1,
        lw=0,
        c='gray'
    )
    for chrom, score in bin_df.groupby("chrom_arm").agg({'score': 'mean'}).to_dict()['score'].items():
        x = np.argwhere(bin_df['chrom_arm'] == chrom).flatten()
        axes[1].plot(
            x,
            [score] * len(x),
            color='red' if score > threshold else 'blue' if score < -threshold else 'gray'
        )

    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_labels, rotation=45)

    axes[1].set_ylabel("Z-scored")
    axes[1].set_title("Bin-level CNA Features")
    plt.tight_layout()
    plt.savefig(output)
    print(f"CNA visualization saved to {output}")


def main(args):
    if args.output is None:
        output = args.input[:-5] + ".png"
    else:
        output = args.output

    visualize_cna(args.input, pon_file=args.pon_file, output=output, threshold=args.threshold)


if __name__ == "__main__":
    cnv_file = "demo/TBR6222_cna.json"
    output = "demo/test.png"
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default=cnv_file, help='Input CNV file')
    parser.add_argument("-p", "--pon_file", default=None, help="PoN file")
    parser.add_argument("-o", "--output", default=output, help="Output png file")
    parser.add_argument("-t", "--threshold", type=int, default=2, help="score threshold")
    main(parser.parse_args())
