from Nodes import *
import numpy as np


class DecisionTree:
    def __init__(self, root=None):
        self.max_depth = None
        self.root = root
        self.data = self.root.data
        self.nodes_dict = {}
        self.nodes = []
        self.leaves = []

    def __repr__(self):
        branch_count = len(self.nodes) - len(self.leaves) - 1  # Taking away the root
        leaf_count = len(self.leaves)
        max_depth = self.get_max_depth()
        return f'A decision tree of depth {max_depth} with {branch_count} branches and {leaf_count} leaves.'

    def classify_example(self, example):
        node = self.root
        is_leaf = node.is_leaf
        while not is_leaf:
            col = node.split_column
            val = node.split_value
            if example[col] <= val:
                node = node.left_child
            else:
                node = node.right_child
            is_leaf = node.is_leaf
        return node.class_label

    def classify_tests(self, data):
        predictions = np.zeros(data.shape[0], dtype='f')
        for row in range(data.shape[0]):
            predictions[row] = self.classify_example(data[row, :])
        return predictions

    def get_errors(self, data):
        predictions = np.zeros(data.shape[0], dtype='f')
        for row in range(data.shape[0]):
            predictions[row] = self.classify_example(data[row, :])

        return sum(data[:, -1] != predictions) / len(data)

    def get_max_depth(self):
        max_depth = 0
        depths = list(self.nodes_dict.keys())
        for depth in depths:
            if len(self.nodes_dict[depth]) > 0:
                max_depth = max(depth, max_depth)
            else:
                del self.nodes_dict[depth]
        self.max_depth = max_depth
        return max_depth

    def get_max_depth2(self):
        # This method has been deprecated
        nodes = [self.root]
        max_depth = 0
        while len(nodes) > 0:
            for node in nodes:
                max_depth = max(node.depth, max_depth)

                if node.left_child:
                    nodes.append(node.left_child)
                    nodes.append(node.right_child)
                nodes.remove(node)
        self.max_depth = max_depth
        return max_depth

    def get_nodes_and_leaves(self):
        nodes_dict = {}
        nodes = []
        nodes_checked = [self.root]
        while len(nodes_checked) > 0:
            for node in nodes_checked:
                depth = node.depth
                nodes_dict.setdefault(depth, [])
                nodes_dict[depth].append(node)
                nodes.append(node)
                if node.left_child or node.right_child:
                    nodes_checked.append(node.left_child)
                    nodes_checked.append(node.right_child)
                if node.is_leaf:
                    self.leaves.append(node)

                nodes_checked.remove(node)
        self.nodes_dict = nodes_dict
        self.nodes = nodes

    def prune_node(self, node):
        # Check if node is connected to two leaves
        if node.left_child:
            if node.left_child.is_leaf and node.right_child.is_leaf:
                # Delete the nodes left and right children. Classify the data for the node and set it to a leaf
                self.nodes_dict[node.depth + 1].remove(node.left_child)
                self.nodes_dict[node.depth + 1].remove(node.right_child)
                self.nodes.remove(node.left_child)
                self.nodes.remove(node.right_child)
                self.leaves.remove(node.left_child)
                self.leaves.remove(node.right_child)
                # Change the node to a leaf and classify the data based on highest number of labels
                node.is_leaf = True
                node.classification()
                self.leaves.append(node)
        return self

    def make_leaf(self, node):
        # convert a branch node to a leaf
        node.is_leaf = True
        node.classify()

    def make_branch(self, node):
        # convert a leaf node to a branch
        node.is_leaf = False
        node.classify()
        pass


def tree_algorithm(node, max_depth=10000):
    # global depth
    assert type(max_depth) == int, "Max tree depth must be an integer"

    depth = node.depth

    if node.check_pure() or node.is_leaf or depth >= max_depth:
        # At a leaf node/pure node or maximum depth, we choose a class label for a node based on the most frequent
        # class label
        node.is_leaf = True
        node.classification()
        return
    else:
        depth += 1
        node.split_column, node.split_value = node.choose_split()
        node.make_baby_nodes()
        data_below, data_above = node.left_child, node.right_child
        data_below.depth = depth
        data_above.depth = depth

        sub_node_lower = tree_algorithm(data_below, max_depth)
        sub_node_upper = tree_algorithm(data_above, max_depth)

        return


def decision_tree_learning(data, max_depth=10000):
    # Builds a tree with data provided and maximum depth
    root = Node(data, depth=0)
    # Recursive Tree Algorithm that attaches nodes to the root
    tree_algorithm(node=root, max_depth=max_depth)
    tree = DecisionTree(root=root)
    assert tree.get_max_depth() <= max_depth, "Maximum depth exceeds requirement"
    tree.get_nodes_and_leaves()
    return tree


def test_tree(data):
    # Train a tree and assert that it's an instance of a DecisionTree object
    t_tree = decision_tree_learning(data)
    assert isinstance(t_tree, DecisionTree), "Object generated is not a decision tree object"


def test_max_depth(data):
    test_depths = np.random.choice(list(range(1, 20)), 4)
    # Picks 4 random integers between from 1 to 19 and for each generates a tree of maximum depth of that integer.
    # The test is to make sure this requirement is respected

    for i in test_depths:
        t_tree = decision_tree_learning(data, int(i))
        assert t_tree.get_max_depth() <= i, "Max Depth has been exceeded!"


if __name__ == '__main__':

    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    datasets = [clean_data, noisy_data]
    for data in datasets:
        test_max_depth(data)
        test_tree(data)
