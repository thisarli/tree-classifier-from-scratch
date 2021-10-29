class Node:
    def __init__(self, attribute, value, left=None, right=None, is_leaf=True, label=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.label = label

    # def is_leaf(self):
    #     return self.left is None and self.right is None

    def create(self):
        branch_dict = {'attribute': self.attribute,
                       'value': self.value,
                       'left': self.left,
                       'right': self.right}
                       # 'leaf': self.is_leaf

        return branch_dict

    def __repr__(self):
        return f"Node({self.attribute}, {self.value}, {self.left}, {self.right}, {self.label} )"

