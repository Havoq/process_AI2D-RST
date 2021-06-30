import json
import os
import re
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

# Compile a regex pattern to identify label IDs in AI2D-RST
label_pattern = re.compile(r'^T\d+\.?\d*$')

ai2d_json = os.path.join('..', 'annotations')
ai2d_rst_json = os.path.join('..', 'ai2d-rst')
ai2d_rst_pickle = os.path.join('..', 'AI2D-RST_resources', 'utils', 'reference_1000.pkl')

class DFConstructor(object):
    def __init__(self):
        self.diagram_id = 0
        self.total_labels = 0
        self.total_diagrams = 0
        self.label_content = defaultdict(str)
        self.macro_groups = defaultdict(list)
        self.dataframe = pd.DataFrame(columns=['diagram_id', 'relation_id', 'relation_type',
                                               'macro_group', 'label_id', 'content', 'role'])

    def construct_dataframes(self):
        self.parse_original()
        self.write_output()

    # Iterate over the original reference file
    def parse_original(self):
        df = pd.read_pickle(ai2d_rst_pickle)

        for index, row in df.iterrows():
            # Reset default dictionaries and parse the next row
            self.label_content = defaultdict(str)
            self.macro_groups = defaultdict(list)
            self.parse_row(row)

    # Parse single row
    def parse_row(self, row):
        # Make a new DataFrame to append to the complete one
        new_df = pd.DataFrame(columns=['diagram_id', 'relation_id', 'relation_type',
                                        'macro_group', 'label_id', 'content', 'role'])

        diagram_name = row['image_name']
        self.diagram_id = diagram_name.split('.')[0]

        print(f'Parsing diagram {self.diagram_id}...')
        self.total_diagrams += 1

        # Get the diagram and its layout and RST graphs
        diagram = row['diagram']

        layout_graph = diagram.layout_graph
        rst_graph = diagram.rst_graph

        # Find all labels and extract their content
        node_types = nx.get_node_attributes(layout_graph, 'kind')
        label_tags = [k for k, v in node_types.items() if v == 'text']
        self.total_labels += len(label_tags)

        # Load corresponding AI2D JSON file
        original_json = os.path.join(ai2d_json, f'{diagram_name}.json')

        with open(original_json) as original:
            annotation = json.loads(original.read())

        self.extract_linguistic_content(annotation, label_tags)

        # Establish macrogroups for each tag
        self.parse_grouping(layout_graph)

        # Iterate over this graph's relations
        new_df = self.find_relations(rst_graph, new_df)

        # Add the new data to the entire DataFrame
        self.dataframe = self.dataframe.append(new_df, ignore_index=True)

    def find_relations(self, rst_graph, dataframe):
        # Find each relation in the diagram
        for node_id, node_data in rst_graph.nodes(data=True):

            if node_data['kind'] == 'relation':

                dataframe = self.parse_relation(node_id, node_data, rst_graph, dataframe)

        return dataframe

    def parse_relation(self, node_id, node_data, graph, dataframe):
        # Parse one relation for its participants
        relation_type = node_data['rel_name']

        try:
            nuclei = node_data['nuclei'].split()
        except KeyError:
            nuclei = [node_data['nucleus']]

        if relation_type == 'joint':
            # Process JOINT relations
            dataframe = self.parse_joint(node_id, nuclei, graph, dataframe)

        else:
            # Process other relations
            dataframe = self.parse_text_elements(node_id, relation_type, nuclei, dataframe, 'nuc')

            # Attempt to find the satellites
            try:
                satellites = node_data['satellites'].split()
                dataframe = self.parse_text_elements(node_id, relation_type, satellites, dataframe, 'sat')
            except KeyError:
                pass

        return dataframe

    def parse_joint(self, joint_id, joint_nuclei, graph, dataframe):
        # Parse a JOINT relation
        for adj in graph.adjacency():
            # Found this relation; find its satellite
            if adj[0] == joint_id:
                satellites = False

                for item in adj[1].items():
                    if item[1]['kind'] == 'satellite':
                        satellites = True
                        sat_id = item[0]

                        # Find satellite relation --
                        # that is, the relation the nuclei of this JOINT actually participate in
                        satellite_relation = graph.nodes[sat_id]

                        # Process the nuclei of this JOINT with the satellite relation
                        rel_name = satellite_relation['rel_name']
                        rel_id = satellite_relation['id']

                        dataframe = self.parse_text_elements(rel_id, rel_name, joint_nuclei, dataframe, 'sat')
                        break

                # If there are no satellites, this JOINT functions as a nucleus
                if satellites is False:

                    for node_id, node_data in graph.nodes(data=True):
                        if node_data['kind'] == 'relation':

                            try:
                                nuclei = node_data['nuclei'].split()
                            except KeyError:
                                nuclei = [node_data['nucleus']]

                            if joint_id in nuclei:
                                # Found the relation this JOINT is the nucleus of
                                rel_name = node_data['rel_name']
                                rel_id = node_data['id']

                                dataframe = self.parse_text_elements(rel_id, rel_name, joint_nuclei, dataframe, 'nuc')
                                break

                break

        return dataframe

    def parse_text_elements(self, rel_id, rel_name, elements, dataframe, mode):
        # Parse each text element in a relation
        for elem in elements:

            if re.search(label_pattern, elem):

                dataframe = self.append_content(rel_id, rel_name, elem, mode, dataframe)

        return dataframe

    def parse_grouping(self, layout_graph):
        # Parse the macro-groups in the diagram
        macro_groups = nx.get_node_attributes(layout_graph, 'macro_group')

        # Set the image constant's type as the default macro-group
        try:
            self.macro_groups['I0'] = macro_groups['I0']
            initial_macro_group = macro_groups['I0']
        except KeyError:
            initial_macro_group = None

        # If there are more macro-groups, parse group nodes to find
        # the lowest-level macro-group for each node
        if len(macro_groups) > 1:
            T = nx.dfs_tree(layout_graph, source='I0')
            self.find_dominant_macro_groups(T, 'I0', initial_macro_group, layout_graph)

    def find_dominant_macro_groups(self, tree, group_node, macro_group, layout_graph):
        # Find the dominant macro-group for each element
        for child in nx.descendants(tree, group_node):
            if re.search(label_pattern, child) and macro_group is not None:
                # This child node belongs in its parent's macro-group
                if child not in self.macro_groups[macro_group]:
                    self.macro_groups[macro_group].append(child)

            elif len(child) == 6:
                # This is a group; call this method recursively
                group_node = layout_graph.nodes[child]

                try:
                    macro_group = group_node['macro_group']
                except KeyError:
                    pass

                self.find_dominant_macro_groups(tree, child, macro_group, layout_graph)

    def extract_linguistic_content(self, annotation, label_tags):
        # Add the linguistic content of each tag to a dictionary
        for tag in label_tags:

            text = annotation['text'][tag]['value']

            self.label_content[tag] = text

    def append_content(self, rel_id, rel_name, label_id, mode, dataframe):
        # Append the label's data to the DataFrame

        # Split labels are added with their original ID
        if '.' in label_id:
            tag_text = self.label_content[label_id.split('.')[0]]
        else:
            tag_text = self.label_content[label_id]

        # Try to find the macro-group for this label from the dictionary
        macro_group = None

        for d_type in self.macro_groups.keys():
            if label_id in self.macro_groups[d_type]:
                macro_group = d_type
                break

        if macro_group is None:
            macro_group = self.macro_groups['I0']

            if macro_group == []:
                macro_group = np.NaN

        dataframe = dataframe.append({'diagram_id': self.diagram_id,
                                      'relation_id': rel_id,
                                      'relation_type': rel_name,
                                      'macro_group': macro_group,
                                      'label_id': label_id,
                                      'content': tag_text,
                                      'role': mode},
                                     ignore_index=True)

        return dataframe

    def write_output(self):
        # Write the output pickle
        print('Pickling dataframe...')
        outfile = f'label_dataframe.pickle'
        outputpath = os.path.join('processed_data')
        outname = os.path.join(outputpath, outfile)
        os.makedirs(outputpath, exist_ok=True)
        self.dataframe = self.dataframe.replace('slice', 'cross-section')
        self.dataframe.to_pickle(outname)
        # example_df = self.dataframe.sample(n=5)
        # example_df.to_csv('random_five.csv')

if __name__ == '__main__':
    constructor = DFConstructor()
    constructor.construct_dataframes()