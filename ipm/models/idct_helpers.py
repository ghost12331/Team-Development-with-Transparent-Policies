import numpy  as np
import torch
import sys
sys.path.insert(0, '../ipm/')
from ipm.models.idct import IDCT
import copy


class Node:
    def __init__(self, idx: int, node_depth: int, is_leaf: bool=False,
                 left_child=None, right_child=None, domain_range=None):
        self.idx = idx
        self.node_depth = node_depth
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.domain_range = domain_range

def find_ancestors(root, node_idx):
    q = [(root, [], [])]
    while q:
        node, curr_left_ancestors, curr_right_ancestors = q.pop(0)
        if node.idx == node_idx:
            return node, curr_left_ancestors, curr_right_ancestors
        if node.left_child and not node.left_child.is_leaf:
            q.append((node.left_child, curr_left_ancestors + [node], curr_right_ancestors))
        if node.right_child and not node.right_child.is_leaf:
            q.append((node.right_child, curr_left_ancestors, curr_right_ancestors + [node]))
    raise ValueError(f'Node with idx {node_idx} not found in tree')

def find_root(leaves):
    root_node = 0
    nodes_in_leaf_path = []
    for leaf in leaves:
        combined_ancestors = leaf[1][0] + leaf[1][1] # these are both lists, concat operation
        nodes_in_leaf_path.append(combined_ancestors)
    for node in nodes_in_leaf_path[0]:
        found_root = True
        for nodes in nodes_in_leaf_path:
            if node not in nodes:
                found_root = False
        if found_root:
            root_node = node
            break
    return root_node

def find_children(node, leaves, current_depth):
    # dfs
    left_subtree = [leaf for leaf in leaves if node.idx in leaf[1][0]]
    right_subtree = [leaf for leaf in leaves if node.idx in leaf[1][1]]

    for _, leaf in left_subtree:
        leaf[0].remove(node.idx)

    for _, leaf in right_subtree:
        leaf[1].remove(node.idx)

    left_child_is_leaf = len(left_subtree) == 1
    right_child_is_leaf = len(right_subtree) == 1

    if not left_child_is_leaf:
        left_child = find_root(left_subtree)
    else:
        left_child = left_subtree[0][0]
    if not right_child_is_leaf:
        right_child = find_root(right_subtree)
    else:
        right_child = right_subtree[0][0]

    left_child = Node(left_child, current_depth, left_child_is_leaf)
    right_child = Node(right_child, current_depth, right_child_is_leaf)
    node.left_child = left_child
    node.right_child = right_child

    if not left_child_is_leaf:
        find_children(left_child, left_subtree, current_depth + 1)
    if not right_child_is_leaf:
        find_children(right_child, right_subtree, current_depth + 1)


def convert_decision_to_leaf(network, decision_node_index, use_gpu=False):
    leaf_info = network.leaf_init_information

    leaves_with_idx = copy.deepcopy([(leaf_idx, leaf_info[leaf_idx]) for leaf_idx in range(len(leaf_info))])
    n_actions = len(leaves_with_idx[0][1][2])
    root = Node(find_root(leaves_with_idx), 0)
    find_children(root, leaves_with_idx, current_depth=1)

    node = root
    q = [root]
    while len(q) > 0:
        node = q.pop(0)
        if node.idx == decision_node_index:
            break
        if node.left_child is not None and node.left_child.is_leaf is False:
            q.append(node.left_child)
        if node.right_child is not None and node.right_child.is_leaf is False:
            q.append(node.right_child)

    descendants = []
    q = [node]
    while len(q) > 0:
        node = q.pop(0)
        descendants.append(node.idx)
        if node.left_child is not None and node.left_child.is_leaf is False:
            q.append(node.left_child)
        if node.right_child is not None and node.right_child.is_leaf is False:
            q.append(node.right_child)

    _, node_removed_left_ancestors, node_removed_right_ancestors = find_ancestors(root, decision_node_index)
    node_removed_left_ancestors = [node.idx for node in node_removed_left_ancestors]
    node_removed_right_ancestors = [node.idx for node in node_removed_right_ancestors]

    new_leaf_info_pruned = [leaf for leaf_idx, leaf in enumerate(leaf_info) \
                 if decision_node_index not in leaf_info[leaf_idx][0] and \
                 decision_node_index not in leaf_info[leaf_idx][1]]

    n_decision_nodes, _ = network.alpha.shape
    old_idx_to_new_idx = {idx:idx for idx in range(n_decision_nodes)}
    for idx in range(n_decision_nodes):
        for descendant in descendants:
            if idx > descendant:
                old_idx_to_new_idx[idx] -= 1

    new_leaf_info = []
    for leaf in new_leaf_info_pruned:
        left_ancestors = []
        for i in range(len(leaf[0])):
            new_node_idx = leaf[0][i]
            left_ancestors.append(old_idx_to_new_idx[new_node_idx])
        right_ancestors = []
        for i in range(len(leaf[1])):
            new_node_idx = leaf[1][i]
            right_ancestors.append(old_idx_to_new_idx[new_node_idx])
        new_leaf_info.append([left_ancestors, right_ancestors, leaf[2]])

    # replace with an arbitrary leaf
    action_idx = np.random.randint(n_actions)
    new_vals_leaf = [-2 for _ in range(n_actions)]
    new_vals_leaf[action_idx] = 2
    new_leaf_info.append([node_removed_left_ancestors, node_removed_right_ancestors, new_vals_leaf])

    old_weights = network.layers  # Get the weights out
    old_comparators = network.comparators  # get the comparator values out
    old_alpha = network.alpha

    new_weights = [old_weights[i].detach().clone().data.cpu().numpy() \
                   for i in range(len(old_weights)) if i not in descendants]
    new_comparators = [old_comparators[i].detach().clone().data.cpu().numpy() \
                       for i in range(len(old_comparators)) if i not in descendants]
    new_alpha = [old_alpha[i].detach().clone().data.cpu().numpy() \
                       for i in range(len(old_alpha)) if i not in descendants]


    new_weights = torch.Tensor(new_weights)
    new_comparators = torch.Tensor(new_comparators)
    new_alpha = torch.Tensor(new_alpha)

    new_network = IDCT(input_dim=network.input_dim, weights=new_weights, comparators=new_comparators,
                       leaves=new_leaf_info, alpha=new_alpha, is_value=network.is_value,
                       device='cuda' if use_gpu else 'cpu', output_dim=network.output_dim, fixed_idct=True)
    # TODO: Need to fix fixed_idct=True, because it means weights are not updated
    # need to instead determine whether to randomize leaf logits or not

    if use_gpu:
        new_network = new_network.cuda()
    return new_network

def convert_leaf_to_decision(network, leaf_index, use_gpu=False):
    """
    Duplicates the network and returns a new one, where the node at leaf_index as been turned into a splitting node
    with two leaves that are slightly noisy copies of the previous node
    :param network: prolonet in
    :param deeper_network: deeper_network to take the new node / leaves from
    :param leaf_index: index of leaf to turn into a split
    :return: new prolonet (value or normal)
    """
    old_leaf_info = copy.deepcopy(network.leaf_init_information)
    old_weights = network.layers  # Get the weights out
    old_comparators = network.comparators  # get the comparator values out
    old_alphas = network.alpha
    leaf_information = old_leaf_info[leaf_index]  # get the old leaf init info out
    left_path = leaf_information[0]
    right_path = leaf_information[1]

    new_weight = np.random.normal(scale=0.2,
                                  size=old_weights[0].size()[0])
    new_comparator = np.random.normal(scale=0.2,
                                      size=old_comparators[0].size()[0])
    new_alpha = np.random.normal(scale=1, size=1)

    # to do: we can replace to ensure less entropy
    new_leaf1 = np.random.normal(scale=0.2,
                                 size=network.action_mus[leaf_index].size()[0]).tolist()
    new_leaf2 = np.random.normal(scale=0.2,
                                 size=network.action_mus[leaf_index].size()[0]).tolist()

    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]
    new_weights.append(new_weight)  # Add it to the list of nodes
    new_comparators = [comp.detach().clone().data.cpu().numpy() for comp in old_comparators]
    new_comparators.append(new_comparator)
    new_alphas = [alpha.detach().clone().data.cpu().numpy() for alpha in old_alphas]
    new_alphas.append(new_alpha)
    # Add it to the list of nodes

    new_weights = torch.Tensor(new_weights)
    new_comparators = torch.Tensor(new_comparators)
    new_alphas = torch.Tensor(new_alphas)

    new_node_ind = len(new_weights) - 1  # Remember where we put it

    # Create the paths, which are copies of the old path but now with a left / right at the new node
    new_leaf1_left = left_path.copy()
    new_leaf1_right = right_path.copy()
    new_leaf2_left = left_path.copy()
    new_leaf2_right = right_path.copy()
    # Leaf 1 goes left at the new node, leaf 2 goes right
    new_leaf1_left.append(new_node_ind)
    new_leaf2_right.append(new_node_ind)

    new_leaf_information = old_leaf_info
    for index, leaf_prob_vec in enumerate(network.action_mus):  # Copy over the learned leaf weight
        new_leaf_information[index][-1] = leaf_prob_vec.detach().clone().data.cpu().numpy().tolist()
    new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
    new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])
    # Remove the old leaf
    del new_leaf_information[leaf_index]
    new_network = IDCT(input_dim=network.input_dim, weights=new_weights, comparators=new_comparators,
                       leaves=new_leaf_information, alpha=new_alphas, is_value=network.is_value,
                       device='cuda' if use_gpu else 'cpu', output_dim=network.output_dim)
    if use_gpu:
        new_network = new_network.cuda()
    return new_network

def compute_entropy(input: []):
    """
    Computes the entropy of a list of probabilities
    :param input: list of probabilities
    :return: entropy
    """
    return -np.sum([p * np.log(p) for p in input])

def logits_to_probs(logits):
    """
    Converts logits to probabilities
    :param logits: list of logits
    :return: list of probabilities
    """
    return [np.exp(logit) / np.sum(np.exp(logits)) for logit in logits]

def expand_idct(idct):
    max_entropy = float('-inf')
    max_entropy_idx = -1

    for i, leaf in enumerate(idct.leaf_init_information):
        entropy = compute_entropy(logits_to_probs(leaf[2]))
        if entropy > max_entropy:
            max_entropy = entropy
            max_entropy_idx = i

    return convert_leaf_to_decision(idct, max_entropy_idx)


def dfs_populate_domain_values(node):
    pass

def prune_idct(idct, env):
    pruned_node_idx = -1
    leaf_info = idct.leaf_init_information
    leaves_with_idx = copy.deepcopy([(leaf_idx, leaf_info[leaf_idx]) for leaf_idx in range(len(leaf_info))])
    root = Node(find_root(leaves_with_idx), 0)
    find_children(root, leaves_with_idx, current_depth=1)
    dfs_populate_domain_values(root)
    domains = env.observation_space
    return  convert_decision_to_leaf(idct, pruned_node_idx)
