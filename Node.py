class Node:
    """
    The class that stores the relevant information for each Node instance in the tree.
    id: the ID of the node (int)
    attribute: the feature for which this Node splits the dataset (int)
    value: the split value for the specified feature at which the node splits the dataset (float)
    left: the left child Node of the Node (Node)
    right: the right child Node of the Node (Node)
    label: the class label assigned to the predictions, if the node is a leaf node (int), otherwise None for non leaves
    count: the count for each class underneath the Node, if the node is a leaf node (used to determine the majority
            label for pruning
    depth: the depth of the Node in the tree
    """
    def __init__(self, id, attribute, value, left=None, right=None, label=None, count=None, depth=None):
        self.id = id
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.count = count
        self.depth = depth

    def __repr__(self):
        return f"Node({self.id}, {self.attribute}, {self.value}, {self.left}, {self.right}, {self.label}, {self.count}, {self.depth})"

