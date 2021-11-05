import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from DecisionTreeBuilder import DecisionTreeBuilder
from data_utils import load_dataset
from script import FILEPATH


def plot_tree(tree):
    plt.figure(figsize=(9, 14))
    ax1 = plt.axes(frameon=True)
    ax1.axes.get_xaxis().set_visible(False)
    plt.gca().invert_yaxis()
    plt.ylabel("Depth")
    green = mlines.Line2D([], [], color='seagreen', marker='.', linestyle='None',markersize=13, label='Room 1')
    orange = mlines.Line2D([], [], color='darkorange', marker='.', linestyle='None',markersize=13, label='Room 2')
    blue = mlines.Line2D([], [], color='royalblue', marker='.', linestyle='None',markersize=13, label='Room 3')
    pink = mlines.Line2D([], [], color='palevioletred', marker='.', linestyle='None',markersize=13, label='Room 4')
    plt.legend(handles=[green, orange, blue, pink])
    plot_binary(tree)
    plt.show()

def plot_binary(node_tree, x=0):
    y = node_tree.depth
    if node_tree.right == None and node_tree.left == None:
        room_colors = {1: 'seagreen', 2: 'darkorange', 3: 'royalblue', 4: 'palevioletred'}
        room_color = room_colors[node_tree.label]
        plt.scatter(x,y, color = room_color, marker = 'o', zorder = 2)
        return None
    else:
        plt.plot((x, x + 0.5**y), (y, y + 1), color = 'grey', zorder = 1)
        plt.plot((x, x - 0.5**y), (y, y + 1), color = 'grey', zorder = 1)
        if node_tree.depth < 4:
            plt.annotate(f"X{node_tree.attribute} < {node_tree.value}", (x, y), textcoords="offset points", xytext=(0,2), ha='center')
        plot_binary(node_tree.left,  x + 0.5**y)
        plot_binary(node_tree.right, x - 0.5**y)
        return None


if __name__ == "__main__":
    data = load_dataset(FILEPATH)
    tree_builder = DecisionTreeBuilder()
    tree, max_depth = tree_builder.build(dataset=data)
    plot_tree(tree)
