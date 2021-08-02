import json
import argparse
import os
import re
import networkx as nx
import pandas as pd
import numpy as np
import sys
sys.path.append('../../../A2D/utils/')
from collections import defaultdict

# Compile a regex pattern to identify label IDs in AI2D-RST
label_pattern = re.compile(r'^T\d+\.?\d*$')


class DFConstructor(object):
    # This class constructs DataFrames from the data in AI2D and AI2D-RST
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

    def parse_original(self):
        # Iterate over the original reference file
        df = pd.read_pickle(ai2d_rst_pickle)

        for index, row in df.iterrows():
            # Reset default dictionaries and parse the next row
            self.label_content = defaultdict(str)
            self.macro_groups = defaultdict(list)
            self.parse_row(row)

    def parse_row(self, row):
        """ Parse a single row in the AI2D-RST reference DataFrame

        Keyword arguments:
        row -- a single row of the DataFrame
        """
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
        """ Find each relation in the diagram

        Keyword arguments:
        rst_graph -- the RST NetworkX graph from AI2D-RST
        dataframe -- the DataFrame to be appended to
        """

        for node_id, node_data in rst_graph.nodes(data=True):
            # Found a relation; parse it
            if node_data['kind'] == 'relation':

                dataframe = self.parse_relation(node_id, node_data, rst_graph, dataframe)

        return dataframe

    def parse_relation(self, node_id, node_data, graph, dataframe):
        """ Parse one relation for its constituent nodes

        Keyword arguments:
        node_id -- the ID of the relation node
        node_data -- the data of the node in JSON format
        graph -- the RST graph of the diagram
        dataframe -- the DataFrame to be appended to
        """
        relation_type = node_data['rel_name']

        # Find the nuclei or nucleus of the relation
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

            # Find and parse possible satellites
            try:
                satellites = node_data['satellites'].split()
                dataframe = self.parse_text_elements(node_id, relation_type, satellites, dataframe, 'sat')
            except KeyError:
                pass

        return dataframe

    def parse_joint(self, joint_id, joint_nuclei, graph, dataframe):
        """ Parse a JOINT relation

        Keyword arguments:
        joint_id -- the ID of the JOINT relation
        joint_nuclei -- the nuclei of the JOINT relation
        graph -- the RST graph
        dataframe -- the DataFrame to be appended to
        """
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
        """ Parse the text elements in a given relation

        Keyword arguments:
        rel_id -- the ID of the relation
        rel_name -- the name of the relation
        elements -- the nuclei or satellites of the relation to be parsed
        dataframe -- the DataFrame to be appended to
        mode -- either 'nuc' (nucleus) or 'sat' (satellite)
        """
        for elem in elements:

            # Check for text elements
            if re.search(label_pattern, elem):

                dataframe = self.append_content(rel_id, rel_name, elem, mode, dataframe)

        return dataframe

    def parse_grouping(self, layout_graph):
        """ Parse the grouping in the layout graph

        Keyword arguments:
        layout_graph -- the layout graph of the diagram
        """

        # Find all macro-groups
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
        """ Find the dominant macro-group for each node; callable recursively

        Keyword arguments:
        tree -- the NetworkX tree containing the group node and its descendants
        group_node -- the group node the children of which are to be parsed
        macro-group -- the current dominant macro-group
        layout_graph -- the diagram's layout graph
        """

        # Iterate over the group node's descendants
        for child in nx.descendants(tree, group_node):
            if re.search(label_pattern, child) and macro_group is not None:
                # This child node is a label and belongs in its parent's macro-group
                if child not in self.macro_groups[macro_group]:
                    self.macro_groups[macro_group].append(child)

            elif len(child) == 6:
                # This is a group; call this method recursively using this group
                group_node = layout_graph.nodes[child]

                try:
                    macro_group = group_node['macro_group']
                except KeyError:
                    pass

                self.find_dominant_macro_groups(tree, child, macro_group, layout_graph)

    def extract_linguistic_content(self, annotation, label_tags):
        """ Add the linguistic content of each tag to a dictionary

        Keyword arguments:
        annotation -- the annotation JSON of the diagram
        label_tags -- the IDs of the labels the content of which is fetched
        """
        for tag in label_tags:

            # Get the content of the label
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

    # Set up the argument parser
    ap = argparse.ArgumentParser()

    # Define arguments
    ap.add_argument("-a", "--ai2d", required=True,
                    help="Path to the directory containing AI2D JSON annotations.")
    ap.add_argument("-ar", "--ai2d_rst", required=True,
                    help="Path to the directory containing AI2D-RST JSON annotations.")
    ap.add_argument("-ap", "--ai2d_rst_pkl", required=True,
                    help="Path to the file containing the pickled AI2D-RST data.")

    # Parse arguments
    args = vars(ap.parse_args())

    # Assign arguments to variables
    ai2d_json = args['ai2d']
    ai2d_rst_json = args['ai2d_rst']
    ai2d_rst_pickle = args['ai2d_rst_pkl']

	# Construct data
    constructor = DFConstructor()
    constructor.construct_dataframes()