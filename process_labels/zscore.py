# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Configure matplotlib figure size and layout
plt.figure(figsize=(16, 12))
plt.tight_layout()

# Read pickled DataFrame
df = pd.read_pickle("processed_data/label_dataframe.pickle")

# Cross-tabulate counts for macro-groups and RST relation types
crosstab = pd.crosstab(df['macro_group'], df['relation_type'], normalize='columns')

# Apply zscore() function from scipy to the cross-tabulated data;
# round the floats to two decimals.
# crosstab = crosstab.apply(zscore).round(2)

# Draw heatmap using seaborn
heatmap = sns.heatmap(crosstab, annot=True, annot_kws={"size": 10}, cmap='viridis')

# Save the heatmap to disk
plt.savefig('zscore_heatmap.pdf', bbox_inches='tight', dpi=300)