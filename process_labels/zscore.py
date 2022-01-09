# Import libraries
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

def zscore_heatmap(df):
    # Configure matplotlib figure size and layout
    plt.figure(figsize=(16, 12))
    plt.tight_layout()

    # Count relation types and macro groups
    rel_counts = df['relation_type'].value_counts()
    macro_counts = df['macro_group'].value_counts()

    # Filter out relations and macro-groups with less than 30 observations
    df = df[~df['relation_type'].isin(rel_counts[rel_counts < 30].index)]
    df = df[~df['macro_group'].isin(macro_counts[macro_counts < 30].index)]

    # Cross-tabulate counts for macro-groups and RST relation types
    crosstab = pd.crosstab(df['relation_type'], df['macro_group'])

    # Apply zscore() function from scipy to the cross-tabulated data;
    # round the floats to two decimals.
    crosstab = crosstab.apply(zscore, axis=1).round(2)

    # Draw heatmap using seaborn
    heatmap = sns.heatmap(crosstab, annot=True, annot_kws={"size": 10}, cmap='viridis')

    # Save the heatmap to disk
    plt.savefig('zscore_heatmap.pdf', bbox_inches='tight', dpi=300)

def zscore_pattern(df, type):
    # Configure matplotlib figure size and layout
    plt.figure(figsize=(16, 12))
    plt.tight_layout()

    df = df[df.sum(axis = 1) >= 30]

    # df = df.apply(zscore).apply(zscore, axis=1).round(2)
    df = df.apply(zscore).apply(zscore, axis=1).round(2)

    heatmap = sns.heatmap(df, annot=True, annot_kws={"size": 10}, cmap='viridis')

    if type == 'macro':
        plt.savefig('zscore_macro_heatmap.pdf', bbox_inches='tight', dpi=300)
    elif type == 'rel':
        plt.savefig('zscore_rel_heatmap.pdf', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    # Set up the argument parser
    ap = argparse.ArgumentParser()

    # Define arguments
    ap.add_argument("-df", "--dataframe", required=True,
                    help="Path to the pickled DataFrame.")

    # Parse arguments
    args = vars(ap.parse_args())

    # Assign arguments to variables
    df_path = args['dataframe']

    # Read pickled DataFrame
    df = pd.read_pickle(df_path)
    zscore_heatmap(df)
