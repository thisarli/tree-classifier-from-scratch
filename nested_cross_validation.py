# tree has been trained on training set
# function from dictionary to node tree

#This function will be called 10 times within 1 fold (=1 "held-out" test set) so 100 times in total
def pruning(tree, validation_set):
  """
    Args:
      tree (dict {id : Node})
      validation_set (np.array)
  """
  node_tree = get_tree_from_dict(tree)
  predictions = predict(validation_set[:,:-1], node_tree)
  labels = validation_set[:, -1]
  orginial_accuracy = accuracy(labels, predictions)
  # the candidates for pruning are the nodes only connected to leaf nodes
  leaves_id = [node.id for node in tree if node.label is not None]
  candidates = [node for node in tree.values() if node.left in leaves_id and node.right in leaves_id]

  while len(candidates) > 0 :
    pruning_accuracies = {}                         # key : id of node where we prune, value: accuracy of pruned tree wrt validation set

    for candidate in candidates:
      # pruning is simulated by assigning majority class label to the candidate
      # (when we predict the evaluation set on the pruned tree, the DecisionTreeBuilder.traverse() function will stop at this node with a not-None label)
      #tree[candidate.id].label = get_majority_label(tree, candidate)
      update_to_leaf_node(tree, candidate)
      node_tree = get_tree_from_dict(tree)
      new_predictions = predict(validation_set[:,:-1], tree)            # THIS SUPPOSES THAT tree IS NOW A Node
      pruned_accuracy = accuracy(new_predictions, labels)
      pruning_accuracies[candidate.id] = pruned_accuracy
      # back to unpruned tree so that we can evaluate the other pruning possibilities against the original tree
      tree[candidate.id].label = None  
      tree[candidate.id].count = None               
    best_candidate = tree[max(pruning_accuracies, key=pruning_accuracies.get)]

    if pruning_accuracies[best_candidate.id] > orginial_accuracy:
      #tree[best_candidate.id].label = get_majority_label(tree, best_candidate)
      #tree[best_candidate.id].count = get_total_children_instances(tree, best_candidate) 
      update_to_leaf_node(tree, best_candidate) 
      break

    leaves_id = [id for id, node in tree if node.label is not None]
    candidates = [node for node in tree.values() if node.left in leaves_id and node.right in leaves_id]

  return tree       #we could also return a node_tree instead

def update_to_leaf_node(tree, node):
  assert tree[node.left].label is not None and tree[node.right].label is not None

  # count is a list of arrays with count[0] labels and count[1] the number of instances for each label
  left_count = tree[node.left].count
  right_count = tree[node.right].count

  count_dict = {left_count[0][i]: left_count[1][i] for i in range(len(left_count[0]))}

  for i in range(len(right_count[0])):
    if right_count[0][i] not in count_dict:
      count_dict[right_count[0][i]] = right_count[1][i]
    else: 
      count_dict[right_count[0][i]] += right_count[1][i]
  # count_dict has key:label, value:total instances with this label in the children nodes
  majority_label = max(count_dict, key=count_dict.get)

  tree[node.id].label =  majority_label
  labels = np.array([label for label in count_dict.keys()])
  counts = np.array([count for count in count_dict.values()])
  tree[node.id].count = (labels, counts)
  return
