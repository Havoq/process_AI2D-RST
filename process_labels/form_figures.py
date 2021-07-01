import os
import spacy
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, Counter
from scipy.stats import mannwhitneyu
from pingouin import mwu
from sklearn import preprocessing


class DFParser(object):
    # This class parses the DataFrame and produces figures from it
    def __init__(self):
        self.df = None
        self.nlp = spacy.load('en_core_web_trf')
        self.figures = 0

        self.diagram_patterns = defaultdict(list)
        self.relation_patterns = defaultdict(list)
        self.total_pos_patterns = []

        self.verb_phrases_by_diagram = defaultdict(int)
        self.verb_phrases_by_relation = defaultdict(int)
        self.noun_phrases_by_diagram = defaultdict(int)
        self.noun_phrases_by_relation = defaultdict(int)
        self.total_phrases_by_diagram = defaultdict(int)
        self.total_phrases_by_relation = defaultdict(int)

        self.words_by_diagram = defaultdict(int)
        self.words_by_relation = defaultdict(int)
        self.elaboration_counts = []
        self.identification_counts = []
        self.total_labels_by_diagram = defaultdict(int)
        self.total_labels_by_relation = defaultdict(int)

    def parse_dataframe(self, path):
        # Parse the DataFrame
        self.df = pd.read_pickle(path)
        self.parse_linguistic_content()
        self.write_output()

    def parse_linguistic_content(self):
        # Parse the linguistic content of the DataFrame
        # List and parse macro-groups
        unique_macro_groups = self.df[self.df['macro_group'].notnull()]['macro_group'].unique()

        for group in unique_macro_groups:
            self.parse_column('macro_group', group)

        # List and parse relations
        unique_relations = self.df['relation_type'].unique()

        for rel in unique_relations:
            if rel == 'joint':
                continue
            self.parse_column('relation_type', rel)

    def parse_column(self, column, value):
        """ Parse a single column with a relation or macro-structure as the value

        Keyword arguments:
        column -- the DataFrame column
        value -- a string with a relation or macro-group name as its value
        """
        rows = self.df[self.df[column] == value]
        rows = rows.iloc[:, 5]
        row_content = rows.to_list()
        print(f'Parsing column {column}, type {value}; {len(rows)} total entries')

        i = 0
        for label in row_content:
            i += 1
            print(f'Parsing row {i}...')
            doc = self.nlp(label)
            # Parse the POS pattern, phrase classes and word count in each label
            self.parse_pos(column, value, doc)
            self.parse_phrase_classes(column, value, doc)
            self.add_word_counts(column, value, label)

    def parse_pos(self, column, value, doc):
        # Parse the POS pattern of a single label
        pos_pattern = " ".join([token.pos_ for token in doc if not token.is_punct])
        pos_pattern = pos_pattern.replace('PROPN', 'NOUN')

        # Add to a list of all POS patterns
        self.total_pos_patterns.append(pos_pattern)

        if column == 'macro_group':
            self.diagram_patterns[value].append(pos_pattern)
        elif column == 'relation_type':
            self.relation_patterns[value].append(pos_pattern)

    def parse_phrase_classes(self, column, value, doc):
        # Parse the label for its phrase class by relation or macro-group
        for token in doc:

            if token.dep_ == 'ROOT':

                if column == 'macro_group':
                    self.total_phrases_by_diagram[value] += 1
                elif column == 'relation_type':
                    self.total_phrases_by_relation[value] += 1

                # Add to either verb or noun phrase collection by relation or macro-group
                if token.pos_ == 'VERB':
                    if column == 'macro_group':
                        self.verb_phrases_by_diagram[value] += 1
                    elif column == 'relation_type':
                        self.verb_phrases_by_relation[value] += 1

                elif token.pos_ in ['NOUN', 'PROPN']:
                    if column == 'macro_group':
                        self.noun_phrases_by_diagram[value] += 1
                    elif column == 'relation_type':
                        self.noun_phrases_by_relation[value] += 1

    def add_word_counts(self, column, value, label):
        # Process word count by macro-group or relation
        if column == 'macro_group':
            self.total_labels_by_diagram[value] += 1
            self.words_by_diagram[value] += len(label.split())

        elif column == 'relation_type':
            self.total_labels_by_relation[value] += 1
            length = len(label.split())
            self.words_by_relation[value] += length

            # As we are interested in the figures between ELABORATION and IDENTIFICATION,
            # handle these specifically
            if value == 'elaboration':
                self.elaboration_counts.append(length)
            elif value == 'identification':
                self.identification_counts.append(length)

    def write_output(self):
        # Write output
        print('Producing output...')
        # self.calculate_mannwhitneyu()
        self.construct_heatmaps()
        self.construct_tables()

    def calculate_mannwhitneyu(self):
        # Calculate Mann-Whitney U between IDENTIFICATION and ELABORATION
        print('Calculating Mann-Whitney U...')
        # results = mwu(self.identification_counts, self.elaboration_counts, tail='one-sided')
        results = mannwhitneyu(self.identification_counts, self.elaboration_counts)
        input(results)

    def construct_heatmaps(self):
        # Construct heatmaps of POS pattern frequencies by relation and macro-group
        print('Constructing heatmaps...')
        rst_df = pd.DataFrame()
        macro_df = pd.DataFrame()

        # Set 5 most common relations as columns
        most_common = self.find_most_common_patterns()

        for tup in most_common:
            pattern = tup[0]
            rst_df[pattern] = ''
            macro_df[pattern] = ''

        # Set RST relations as rows
        for rel in self.relation_patterns:

            if rel == 'joint':
                continue

            for column in rst_df:
                # Each column is a pattern
                all_patterns = self.relation_patterns[rel]
                occurrences = all_patterns.count(column)
                rst_df.at[rel, column] = occurrences

        # Another DataFrame with macro-groups as rows
        for mg in self.diagram_patterns:
            for column in macro_df:
                all_patterns = self.diagram_patterns[mg]
                occurrences = all_patterns.count(column)
                macro_df.at[mg, column] = occurrences

        i = 0
        for df in [rst_df, macro_df]:
            i += 1
            df = df.fillna(0)
            df = df.astype(int)
            csv_target = os.path.join('processed_data', f'heatmaps_{i}.csv')
            df.to_csv(csv_target)

        # Create heatmaps from the DataFrames
        i = 0
        for df in [rst_df, macro_df]:
            i += 1
            df_scaled = df.copy()

            vals = df_scaled.values
            min_max_scaler = preprocessing.MinMaxScaler()
            vals_scaled = min_max_scaler.fit_transform(vals)
            df_scaled = pd.DataFrame(vals_scaled)

            df_scaled = df_scaled.astype(float)
            sns.heatmap(df_scaled, cbar_kws={'label': 'Standardized pattern occurrence'},
                        center=0, linewidths=.5, cmap='viridis')
            sns.set(font_scale=1)
            plt.xlabel('Most common POS patterns')
            plt.ylabel('Source category')
            plt.show()
            csv_target = os.path.join('processed_data', f'standardized_heatmaps_{i}.csv')
            df_scaled.to_csv(csv_target)

    def find_most_common_patterns(self):
        c = Counter(self.total_pos_patterns)
        # return c.most_common(10)
        return c.most_common(5)

    def construct_tables(self):
        self.count_unique_patterns()
        self.calculate_overlap()
        self.calculate_average_word_counts()
        self.calculate_phrase_types()
        self.tabulate_total_labels()

    def calculate_average_word_counts(self):
        # Calculate the average word count by relation and macro-group
        print('Calculating average word counts...')

        rst_df = pd.DataFrame()
        macro_df = pd.DataFrame()

        for relation in self.words_by_relation.keys():

            if relation == 'joint':
                continue

            average = self.words_by_relation[relation] / self.total_labels_by_relation[relation]
            average = round(average, 2)
            rst_df = rst_df.append({relation: average}, ignore_index=True)

        for macro_group in self.words_by_diagram.keys():
            average = self.words_by_diagram[macro_group] / self.total_labels_by_diagram[macro_group]
            average = round(average, 2)
            macro_df = macro_df.append({macro_group: average}, ignore_index=True)

        i = 0
        for df in [rst_df, macro_df]:
            i += 1
            df = df.fillna(0)
            csv_target = os.path.join('processed_data', f'word_counts_{i}.csv')
            df.to_csv(csv_target)

    def count_unique_patterns(self):
        # Count the unique POS patterns by relation and macro-group
        print('Counting unique patterns...')

        rst_df = pd.DataFrame()
        macro_df = pd.DataFrame()

        for relation in self.relation_patterns.keys():

            if relation == 'joint':
                continue

            patterns = set(self.relation_patterns[relation])
            patterns = len(patterns)
            rst_df = rst_df.append({relation: patterns}, ignore_index=True)

        for macro_group in self.diagram_patterns.keys():
            patterns = set(self.diagram_patterns[macro_group])
            patterns = len(patterns)
            macro_df = macro_df.append({macro_group: patterns}, ignore_index=True)

        i = 0
        for df in [rst_df, macro_df]:
            i += 1
            df = df.fillna(0)
            df = df.astype(int)
            csv_target = os.path.join('processed_data', f'unique_patterns_{i}.csv')
            df.to_csv(csv_target)

    def calculate_overlap(self):
        # Calculate the overlap between labels in each relation and macro-group
        print('Calculating overlap...')
        new_df = pd.DataFrame()

        for rel in self.relation_patterns.keys():

            if rel == 'joint':
                continue

            for mg in self.diagram_patterns.keys():
                overlapping_rows = self.df.loc[(self.df['relation_type'] == rel) & (self.df['macro_group'] == mg)]
                new_df.at[rel, mg] = len(overlapping_rows.index)

        new_df = new_df.fillna(0)
        new_df = new_df.astype(int)
        csv_target = os.path.join('processed_data', 'overlap.csv')
        new_df.to_csv(csv_target)

    def calculate_phrase_types(self):
        # Count the number of verb and noun phrases by relation and macro-group
        print('Calculating phrase percentages...')
        rst_df = pd.DataFrame(columns=['VP', 'NP', 'VP %', 'NP %'])
        macro_df = pd.DataFrame(columns=['VP', 'NP', 'VP %', 'NP %'])

        for rel in self.total_labels_by_relation.keys():
            total_labels = self.total_phrases_by_relation[rel]
            verb_phrases = self.verb_phrases_by_relation[rel]
            verb_percentage = (self.verb_phrases_by_relation[rel] / self.total_phrases_by_relation[rel]) * 100
            noun_phrases = self.noun_phrases_by_relation[rel]
            noun_percentage = (self.noun_phrases_by_relation[rel] / self.total_phrases_by_relation[rel]) * 100

            rst_df.at[rel, 'VP'] = verb_phrases
            rst_df.at[rel, 'NP'] = noun_phrases
            rst_df.at[rel, 'VP %'] = verb_percentage
            rst_df.at[rel, 'NP %'] = noun_percentage
            rst_df.at[rel, 'total'] = total_labels

        for group in self.total_labels_by_diagram.keys():
            total_labels = self.total_phrases_by_diagram[group]
            verb_phrases = self.verb_phrases_by_diagram[group]
            verb_percentage = (self.verb_phrases_by_diagram[group] / self.total_phrases_by_diagram[group]) * 100
            noun_phrases = self.noun_phrases_by_diagram[group]
            noun_percentage = (self.noun_phrases_by_diagram[group] / self.total_phrases_by_diagram[group]) * 100

            macro_df.at[group, 'VP'] = verb_phrases
            macro_df.at[group, 'NP'] = noun_phrases
            macro_df.at[group, 'VP %'] = verb_percentage
            macro_df.at[group, 'NP %'] = noun_percentage
            macro_df.at[group, 'total'] = total_labels

        i = 0
        for df in [rst_df, macro_df]:
            i += 1
            df = df.fillna(0)
            df = df.astype(int)
            csv_target = os.path.join('processed_data', f'phrase_type_counts_{i}.csv')
            df.to_csv(csv_target)

    def tabulate_total_labels(self):
        # Create tables of total labels in each relation and macro-group
        print('Tabulating total label counts...')
        rst_df = pd.DataFrame()
        macro_df = pd.DataFrame()

        for rel in self.total_labels_by_relation.keys():
            total_labels = self.total_labels_by_relation[rel]
            rst_df = rst_df.append({rel: total_labels}, ignore_index=True)

        for group in self.total_phrases_by_diagram.keys():
            total_labels = self.total_labels_by_diagram[group]
            macro_df = macro_df.append({group: total_labels}, ignore_index=True)

        i = 0
        for df in [rst_df, macro_df]:
            i += 1
            df = df.fillna(0)
            df = df.astype(int)
            csv_target = os.path.join('processed_data', f'total_labels_{i}.csv')
            df.to_csv(csv_target)


def z_score(df):
    df_std = df.copy()

    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std


def normalize(df):
    column_max_values = df.max()
    df_max_value = column_max_values.max()
    column_min_values = df.min()
    df_min_value = column_min_values.min()
    df_norm = (df - df_min_value) / (df_max_value - df_min_value)

    return df_norm


if __name__ == '__main__':
    pickle_path = os.path.join('processed_data', 'label_dataframe.pickle')
    parser = DFParser()
    parser.parse_dataframe(pickle_path)