import numpy as np

def calc_entropy(data):
    labels = data[:, -1]
    label, label_count = np.unique(labels, return_counts=True)

    total = np.sum(label_count)
    prob = label_count / total

    entropy = - np.sum(prob * np.log2(prob))

    return entropy


class Node:
    def __init__(self, data, split_column=None, split_value=None, left_child=None, right_child=None, is_leaf=False,
                 depth=None, parent_split_column=None, parent_split_value=None, direction=None):
        self.split_column = split_column
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.depth = depth
        self.data = data
        self.class_label = None
        self.parent_split_column = parent_split_column
        self.parent_split_value = parent_split_value
        self.direction = direction

    def __lt__(self, other):
        return self.split_value < other.split_value

    def __gt__(self, other):
        return self.split_value > other.split_value

    def __repr__(self):

        if self.depth == 0:
            node_type = 'Root Node'
            return f'{node_type}: [ depth: {self.depth}]'

        elif self.is_leaf:
            node_type = 'Leaf Node'
            return f'{node_type} : [class: {self.class_label}.split column: {self.parent_split_column}, split_value: {self.parent_split_value}, depth: {self.depth}]'
        else:
            node_type = 'Branch Node'

        return f'{node_type} : [split column: {self.parent_split_column}, split_value: {self.parent_split_value}, depth: {self.depth}]'

    def check_pure(self):
        labels = self.data[:, -1]
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            self.is_leaf = True
        else:
            self.is_leaf = False

    def get_potential_splits(self):
        potential_splits = {}
        columns = self.data.shape[1]
        for column_index in range(columns - 1):
            potential_splits[column_index] = []

            for datapoint in self.data:
                if datapoint[column_index] not in potential_splits[column_index]:
                    potential_splits[column_index].append(datapoint[column_index])

        return potential_splits

    def split_data(self, split_column, split_value):

        data_below = self.data[(self.data[:, split_column] <= split_value)]
        data_above = self.data[(self.data[:, split_column] > split_value)]

        return data_below, data_above

    def information_gain(self, split_column, split_value):

        data_below, data_above = self.split_data(split_column, split_value)

        ig = calc_entropy(self.data) - (len(data_below) / len(self.data)) * calc_entropy(data_below) - (
                len(data_above) / len(self.data)) * calc_entropy(data_above)
        return ig

    def choose_split(self):
        if not self.is_leaf:
            best_split_ig = -float('inf')
            potential_splits = self.get_potential_splits()

            for key in potential_splits.keys():
                for split in potential_splits[key]:

                    ig = self.information_gain(key, split)

                    if ig > best_split_ig:
                        best_split_ig = ig
                        split_column = key
                        split_value = split
        return split_column, split_value

    def make_baby_nodes(self):
        if not self.is_leaf:
            data_below, data_above = self.split_data(self.split_column, self.split_value)
            # Set new child nodes inheriting parent's thresholds
            self.left_child = Node(data_below, parent_split_column=self.split_column,
                                   parent_split_value=self.split_value, direction='left')
            self.right_child = Node(data_above, parent_split_column=self.split_column,
                                    parent_split_value=self.split_value, direction='right')

        return None

    def classification(self):
        if self.is_leaf:
            labels = self.data[:, -1]
            label, label_count = np.unique(labels, return_counts=True)

            class_choice = label[label_count.argmax()]
            self.class_label = class_choice

        else:
            class_choice = None

        return class_choice

    def get_thresholds(self):
        return dict(depth=self.depth, class_label=self.class_label, split_column=self.split_column,
                    split_value=self.split_value)

    def has_leaves(self):
        if self.left_child:
            return (self.left_child.is_leaf and self.left_child.is_leaf)
        else:
            return False

    def classify(self):
        if self.is_leaf:
            labels = self.data[:, -1]
            label, label_count = np.unique(labels, return_counts=True)
            class_choice = label[label_count.argmax()]
            self.class_label = class_choice

        else:
            self.class_label = None

if __name__ == '__main__':
    pass
