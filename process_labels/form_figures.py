import os
import spacy
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, Counter
from scipy.stats import mannwhitneyu
from sklearn import preprocessing
from pingouin import mwu
from zscore import *


class DFParser(object):
    def __init__(self):
        self.df = None
        self.word_count_rel_df = pd.DataFrame(columns=['relation', 'words'])
        self.word_count_macro_df = pd.DataFrame(columns=['macro_group', 'words'])
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

        self.nuc_blob_by_diagram = defaultdict(int)
        self.nuc_label_by_diagram = defaultdict(int)
        self.nuc_diag_by_diagram = defaultdict(int)
        self.nuc_blob_by_relation = defaultdict(int)
        self.nuc_label_by_relation = defaultdict(int)
        self.nuc_diag_by_relation = defaultdict(int)

    def parse_dataframe(self, path):
        self.df = pd.read_pickle(path)
        self.parse_linguistic_content()
        self.write_output()

    def parse_linguistic_content(self):
        unique_macro_groups = self.df[self.df['macro_group'].notnull()]['macro_group'].unique()

        for group in unique_macro_groups:
            self.parse_column('macro_group', group)

        unique_relations = self.df['relation_type'].unique()

        for rel in unique_relations:
            if rel == 'joint':
                continue
            self.parse_column('relation_type', rel)

    def parse_column(self, column, value):
        rows = self.df[self.df[column] == value]
        rows = rows.iloc[:, 5]
        row_content = rows.to_list()
        print(f'Parsing column {column}, type {value}; {len(rows)} total entries')

        i = 0
        for label in row_content:
            i += 1
            print(f'Parsing row {i}...')
            doc = self.nlp(label)

            self.parse_pos(column, value, doc)
            self.parse_phrase_classes(column, value, doc)
            self.add_word_counts(column, value, label)

    def parse_pos(self, column, value, doc):
        pos_pattern = " ".join([token.pos_ for token in doc if not token.is_punct])
        pos_pattern = pos_pattern.replace('PROPN', 'NOUN')

        self.total_pos_patterns.append(pos_pattern)

        if column == 'macro_group':
            self.diagram_patterns[value].append(pos_pattern)
        elif column == 'relation_type':
            self.relation_patterns[value].append(pos_pattern)

    def parse_phrase_classes(self, column, value, doc):
        for token in doc:

            if token.dep_ == 'ROOT':

                if column == 'macro_group':
                    self.total_phrases_by_diagram[value] += 1
                elif column == 'relation_type':
                    self.total_phrases_by_relation[value] += 1

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
        if column == 'macro_group':
            self.total_labels_by_diagram[value] += 1
            self.words_by_diagram[value] += len(label.split())
            self.word_count_macro_df = self.word_count_macro_df.append({'macro_group': value,
                                                                        'words': len(label.split())},
                                                                       ignore_index=True)

        elif column == 'relation_type':
            self.total_labels_by_relation[value] += 1
            length = len(label.split())
            self.words_by_relation[value] += length

            self.word_count_rel_df = self.word_count_rel_df.append({'relation': value,
                                                                    'words': len(label.split())},
                                                                   ignore_index=True)

            if value == 'elaboration':
                self.elaboration_counts.append(length)
            elif value == 'identification':
                self.identification_counts.append(length)

    def write_output(self):
        print('Producing output...')
        self.construct_heatmaps()
        zscore_heatmap(self.df)
        self.construct_tables()

    def calculate_mannwhitneyu(self):
        print('Calculating Mann-Whitney U...')
        #results = mwu(self.identification_counts, self.elaboration_counts, tail='one-sided')
        results = mannwhitneyu(self.identification_counts, self.elaboration_counts)
        input(results)

    def construct_heatmaps(self):
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

        i = 1
        for df in [rst_df, macro_df]:
            df = df.fillna(0)
            df = df.astype(int)
            df.to_csv(f'heatmaps_{i}.csv')
            i += 1

        zscore_pattern(rst_df.astype(float), 'rel')
        zscore_pattern(macro_df.astype(float), 'macro')

    def find_most_common_patterns(self):
        c = Counter(self.total_pos_patterns)
        #return c.most_common(10)
        return c.most_common(5)

    def construct_tables(self):
        self.count_phrase_types()
        self.count_unique_patterns()
        self.calculate_overlap()
        self.calculate_average_word_counts()
        self.calculate_phrase_percent()
        self.tabulate_total_labels()

    def count_phrase_types(self):
        print('Counting VP and NP occurrences...')

        rst_df = pd.DataFrame()
        macro_df = pd.DataFrame()

        # Verb phrases by relation
        for rel in self.verb_phrases_by_relation.keys():

            if rel == 'joint':
                continue

            occurrences = self.verb_phrases_by_relation[rel]
            rst_df.at['Verb Phrases', rel] = occurrences

        # Noun phrases by relation
        for rel in self.noun_phrases_by_relation.keys():

            if rel == 'joint':
                continue

            occurrences = self.noun_phrases_by_relation[rel]
            rst_df.at['Noun Phrases', rel] = occurrences

        # Verb phrases by macro-group
        for macro_group in self.verb_phrases_by_diagram.keys():
            occurrences = self.verb_phrases_by_diagram[macro_group]
            macro_df.at['Verb Phrases', macro_group] = occurrences

        # Noun phrases by macro-group
        for macro_group in self.noun_phrases_by_diagram.keys():
            occurrences = self.noun_phrases_by_diagram[macro_group]
            macro_df.at['Noun Phrases', macro_group] = occurrences

        i = 1
        for df in [rst_df, macro_df]:
            df = df.fillna(0)
            df = df.astype(int)
            output = os.path.join('processed_data', f'phrase_types_{i}.csv')
            df.to_csv(output)
            i += 1

    def calculate_average_word_counts(self):
        print('Calculating average word counts...')

        rst_df = pd.DataFrame(columns=['relation', 'word_count', 'total_labels'])
        macro_df = pd.DataFrame(columns=['macro_group', 'word_count', 'total_labels'])
        # rst_df = pd.DataFrame()
        # macro_df = pd.DataFrame()

        for relation in self.words_by_relation.keys():

            if relation == 'joint':
                continue

            average = self.words_by_relation[relation] / self.total_labels_by_relation[relation]
            average = round(average, 2)

            rst_df = rst_df.append({relation: average}, ignore_index=True)


        for macro_group in self.words_by_diagram.keys():
            average = self.words_by_diagram[macro_group] / self.total_labels_by_diagram[macro_group]
            average = round(average, 2)

            macro_df = macro_df.append({relation: average}, ignore_index=True)

        i = 1
        for df in [rst_df, macro_df]:
            df = df.fillna(0)
            df.to_csv(f'word_counts_{i}.csv')
            i += 1
            #self.draw_table(df)

    def count_unique_patterns(self):
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

        i = 1
        for df in [rst_df, macro_df]:
            df = df.fillna(0)
            df = df.astype(int)
            df.to_csv(f'unique_patterns_{i}.csv')
            i += 1

    def calculate_overlap(self):
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
        new_df.to_csv('overlap.csv')
        #self.draw_table(new_df, row_labels=True)

    def calculate_phrase_percent(self):
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

            print(rel)
            print(f'Total labels: {total_labels}')

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

            print(group)
            print(f'Total labels: {total_labels}')
            print(f'Total VPs: {verb_phrases}')
            print(f'Total NPs: {noun_phrases}')

        i = 1
        for df in [rst_df, macro_df]:
            df = df.fillna(0)
            df = df.astype(int)
            df.to_csv(f'phrase_type_percentage_{i}.csv')
            i += 1

    def tabulate_total_labels(self):
        print('Tabulating total label counts...')
        rst_df = pd.DataFrame()
        macro_df = pd.DataFrame()

        for rel in self.total_labels_by_relation.keys():
            total_labels = self.total_labels_by_relation[rel]
            rst_df = rst_df.append({rel: total_labels}, ignore_index=True)

        for group in self.total_phrases_by_diagram.keys():
            total_labels = self.total_labels_by_diagram[group]
            macro_df = macro_df.append({group: total_labels}, ignore_index=True)

        i = 1
        for df in [rst_df, macro_df]:
            df = df.fillna(0)
            df = df.astype(int)
            df.to_csv(f'total_labels_{i}.csv')
            i += 1

    def count_nucleus_types(self):
        print('Counting nucleus types...')
        pass

    def draw_table(self, df, row_labels=False):
        self.figures += 1
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        #ax.axis('tight')

        if row_labels is True:
            table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=list(df.index),
                             cellLoc='center', loc='center')
        else:
            table = ax.table(cellText=df.values, colLabels=df.columns,
                             cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        table.scale(1.5, 1.5)
        plt.subplots_adjust(left=0.2, top=0.8)

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

        plt.show()


# def grouped_weighted_avg(values, weights, by):
#     return (values * weights).groupby(by).sum() / weights.groupby(by).sum()


if __name__ == '__main__':
    pickle_path = os.path.join('data', 'output', 'label_dataframe.pickle')
    parser = DFParser()
    parser.parse_dataframe(pickle_path)