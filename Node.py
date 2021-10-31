class Node:
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

