# process_AI2D-RST

The `process_labels` directory contains two scripts: `construct_dataframes.py` and `form_figures.py`. These scripts are intended for parsing the AI2D and AI2D-RST diagram corpora for certain linguistic features.

## `construct_dataframes.py`

This script is executed first; it iterates over the corpora to find labels that function in certain rhetorical relations and their content, producing a pickled DataFrame in a subdirectory named `processed_data`.

## `form_figures.py`

This script is executed afterwards. It processes labels by rhetorical relation and macro-group using spaCy, extracting part-of-speech (POS) patterns, phrase classes, and average word counts. It also produces CSV files in the `processed_data` subdirectory as well as heatmaps of the most commonly occurring POS patterns by relation and macro-group.
