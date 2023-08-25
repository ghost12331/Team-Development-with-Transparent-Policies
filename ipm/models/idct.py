import torch.nn as nn
import torch
import numpy as np
import typing as t
import torch.nn.functional as F
import time


class IDCT(nn.Module):
    def __init__(self,
                 input_dim: int,
                 weights: t.Union[t.List[np.array], np.array, None],
                 comparators: t.Union[t.List[np.array], np.array, None],
                 alpha: t.Union[t.List[np.array], np.array, None],
                 leaves: t.Union[None, int, t.List],
                 output_dim: t.Optional[int] = None,
                 use_individual_alpha=False,
                 device: str = 'cpu',
                 hard_node: bool = False,
                 argmax_tau: float = 1.0,
                 l1_hard_attn=False,
                 num_sub_features=1,
                 use_gumbel_softmax=False,
                 is_value=False,
                 fixed_idct=False,
                 alg_type='ppo',
                 only_optimize_leaves=False):
        super(IDCT, self).__init__()
        """
        Initialize the Interpretable Continuous Control Tree (ipm)

        :param input_dim: (observation/feature) input dimensionality
        :param weights: the weight vector for each node to initialize
        :param comparators: the comparator vector for each node to initialize
        :param alpha: the alpha to initialize
        :param leaves: the number of leaves of ipm
        :param output_dim: (action) output dimensionality
        :param use_individual_alpha: whether use different alphas for different nodes 
                                    (sometimes it helps boost the performance)
        :param device: which device should ipm run on [cpu|cuda]
        :param use_submodels: whether use linear sub-controllers (submodels)
        :param hard_node: whether use differentiable crispification (this arg does not
                          influence the differentiable crispification procedure in the 
                          sparse linear controllers)
        :param argmax_tau: the temperature of the diff_argmax function
        :param sparse_submodel_type: the type of the sparse sub-controller, 1 for L1 
                                    regularization, 2 for feature selection, other 
                                    values (default: 0) for not sparse
        :param fs_submodel_version: the version of feature-section submodel to use
        :param l1_hard_attn: whether only sample one linear controller to perform L1 
                             regularization for each update when using l1-reg submodels
        :param num_sub_features: the number of chosen features for sparse sub-controllers
        :param use_gumbel_softmax: whether use gumble softmax instead of the differentiable 
                                   argmax (diff_argmax) proposed in the paper
        :param alg_type: current supported RL methods [SAC|TD3] (the results in the paper 
                         were obtained by SAC)
        :param only_optimize_leaves: whether only optimize the leaves of the tree
        """
        self.device = device
        self.leaf_init_information = leaves
        self.hard_node = hard_node
        self.argmax_tau = argmax_tau
        self.fixed_idct = fixed_idct

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = weights
        self.comparators = comparators
        self.l1_hard_attn = l1_hard_attn
        self.num_sub_features = num_sub_features
        self.use_gumbel_softmax = use_gumbel_softmax
        self.use_individual_alpha = use_individual_alpha
        self.is_value = is_value
        self.alg_type = alg_type
        self.only_optimize_leaves = only_optimize_leaves

        self.init_comparators(comparators)
        self.init_weights(weights)
        self.init_alpha(alpha)
        self.init_paths()
        self.init_leaves()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.num_leaves = self.layers.size(0) + 1

        # Experimental parameter for using soft labels in the nodes instead of hard labels
        self.training = False

        if self.alg_type == 'td3':
            self.tanh = nn.Tanh()

    def init_comparators(self, comparators):
        if comparators is None:
            comparators = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2 ** level):
                    comparators.append(np.random.normal(0, 1.0, 1))
        new_comps = torch.tensor(comparators, dtype=torch.float).to(self.device)
        if not self.fixed_idct:
            new_comps.requires_grad = True
            self.comparators = nn.Parameter(new_comps, requires_grad=True)
        else:
            new_comps.requires_grad = False
            self.comparators = nn.Parameter(new_comps, requires_grad=False)

        if self.only_optimize_leaves:
            self.comparators = nn.Parameter(new_comps, requires_grad=False)


    def init_weights(self, weights):
        if weights is None:
            weights = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2 ** level):
                    weights.append(np.random.rand(self.input_dim))

        new_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        if not self.fixed_idct:
            new_weights.requires_grad = True
            self.layers = nn.Parameter(new_weights, requires_grad=True)
        else:
            new_weights.requires_grad = False
            self.layers = nn.Parameter(new_weights, requires_grad=False)

        if self.only_optimize_leaves:
            self.layers = nn.Parameter(new_weights, requires_grad=False)

    def init_alpha(self, alpha):
        if alpha is None:
            if self.use_individual_alpha:
                alphas = []
                if type(self.leaf_init_information) is int:
                    depth = int(np.floor(np.log2(self.leaf_init_information)))
                else:
                    depth = 4
                for level in range(depth):
                    for node in range(2 ** level):
                        alphas.append([1.0])
            else:
                alphas = [1.0]
        else:
            alphas = alpha
        self.alpha = torch.tensor(alphas, dtype=torch.float).to(self.device)

        if not self.fixed_idct:
            self.alpha.requires_grad = True
            self.alpha = nn.Parameter(self.alpha, requires_grad=True)
        else:
            self.alpha.requires_grad = False
            self.alpha = nn.Parameter(self.alpha, requires_grad=False)

        if self.only_optimize_leaves:
            self.alpha = nn.Parameter(self.alpha, requires_grad=False)

    def init_paths(self):
        if type(self.leaf_init_information) is list:
            left_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)), dtype=torch.float)
            right_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)), dtype=torch.float)
            for n in range(0, len(self.leaf_init_information)):
                for i in self.leaf_init_information[n][0]:
                    left_branches[i][n] = 1.0
                for j in self.leaf_init_information[n][1]:
                    right_branches[j][n] = 1.0
        else:
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            left_branches = torch.zeros((2 ** depth - 1, 2 ** depth), dtype=torch.float)
            for n in range(0, depth):
                row = 2 ** n - 1
                for i in range(0, 2 ** depth):
                    col = 2 ** (depth - n) * i
                    end_col = col + 2 ** (depth - 1 - n)
                    if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
                        break
                    left_branches[row + i, col:end_col] = 1.0
            right_branches = torch.zeros((2 ** depth - 1, 2 ** depth), dtype=torch.float)
            left_turns = np.where(left_branches == 1)
            for row in np.unique(left_turns[0]):
                cols = left_turns[1][left_turns[0] == row]
                start_pos = cols[-1] + 1
                end_pos = start_pos + len(cols)
                right_branches[row, start_pos:end_pos] = 1.0
        left_branches.requires_grad = False
        right_branches.requires_grad = False
        self.left_path_sigs = nn.Parameter(left_branches.to(self.device), requires_grad=False)
        self.right_path_sigs = nn.Parameter(right_branches.to(self.device), requires_grad=False)

    def init_leaves(self):
        if type(self.leaf_init_information) is list:
            new_leaves = [leaf[-1] for leaf in self.leaf_init_information]
        else:
            new_leaves = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4

            last_level = np.arange(2 ** (depth - 1) - 1, 2 ** depth - 1)
            going_left = True
            leaf_index = 0
            self.leaf_init_information = []
            for level in range(2 ** depth):
                curr_node = last_level[leaf_index]
                turn_left = going_left
                left_path = []
                right_path = []
                while curr_node >= 0:
                    if turn_left:
                        left_path.append(int(curr_node))
                    else:
                        right_path.append(int(curr_node))
                    prev_node = np.ceil(curr_node / 2) - 1
                    if curr_node // 2 > prev_node:
                        turn_left = False
                    else:
                        turn_left = True
                    curr_node = prev_node
                if going_left:
                    going_left = False
                else:
                    going_left = True
                    leaf_index += 1
                new_probs = np.random.uniform(0, 1, self.output_dim)
                self.leaf_init_information.append([sorted(left_path), sorted(right_path), new_probs])
                new_leaves.append(new_probs)

        labels = torch.tensor(new_leaves, dtype=torch.float).to(self.device)

        if not self.fixed_idct:
            labels.requires_grad = True
            self.action_mus = nn.Parameter(labels, requires_grad=True)
            # torch.nn.init.xavier_uniform_(self.action_mus)
        else:
            labels.requires_grad = False
            self.action_mus = nn.Parameter(labels, requires_grad=False)


    def update_leaf_init_information(self):
        """
        function assumes that you have just done an command like
        idct.action_mus = nn.Parameter(action_mus, requires_grad=True)
        Returns:

        """
        for i in range(0, len(self.leaf_init_information)):
            self.leaf_init_information[i][-1] = self.action_mus[i].detach().cpu().numpy()

    def diff_argmax(self, logits, dim=-1):
        tau = self.argmax_tau
        sample = self.use_gumbel_softmax

        if sample:
            gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            logits = logits + gumbels

        y_soft = (logits / tau).softmax(-1)
        # straight through

        if self.training:
            return y_soft

        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft

        return ret

    def forward(self, input_data, embedding_list=None):

        # CARTPOLE DEBUGGING
        # oracle method
        # oracle_policy = False
        # if oracle_policy:
        #     preds = []
        #     for sample in range(len(input_data)):
        #         theta = input_data[sample, 2]
        #         w = input_data[sample, 3]
        #         if abs(theta) < 0.03:
        #             if  w < 0:
        #                 preds.append([1, 0])
        #             else:
        #                 preds.append([0, 1])
        #         else:
        #             if theta < 0:
        #                 preds.append([1, 0])
        #             else:
        #                 preds.append([0, 1])
        #     return torch.Tensor(preds).to(self.device)

        self.hard_node = True
        if self.hard_node:
            ## node crispification
            weights = torch.abs(self.layers)
            # onehot_weights: [num_nodes, num_leaves]
            onehot_weights = self.diff_argmax(weights)
            # divisors: [num_node, 1]
            divisors = (weights * onehot_weights).sum(-1).unsqueeze(-1)
            # fill 0 with 1
            divisors_filler = torch.zeros(divisors.size()).to(divisors.device)
            divisors_filler[divisors == 0] = 1
            divisors = divisors + divisors_filler
            new_comps = self.comparators / divisors
            new_weights = self.layers * onehot_weights / divisors
            new_alpha = self.alpha
        else:
            new_comps = self.comparators
            new_weights = self.layers
            new_alpha = self.alpha

        # original input_data dim: [batch_size, input_dim]
        input_copy = input_data.clone()
        input_data = input_data.t().expand(new_weights.size(0), *input_data.t().size())
        # layers: [num_node, input_dim]
        # input_data dim: [batch_size, num_node, input_dim]
        input_data = input_data.permute(2, 0, 1)
        # after discretization, some weights can be -1 depending on their origal values
        # comp dim: [batch_size, num_node, 1]
        comp = new_weights.mul(input_data)
        comp = comp.sum(dim=2).unsqueeze(-1)
        comp = comp.sub(new_comps.expand(input_data.size(0), *new_comps.size()))
        if self.use_individual_alpha:
            comp = comp.mul(new_alpha.expand(input_data.size(0), *new_alpha.size()))
        else:
            comp = comp.mul(new_alpha)
        if self.hard_node:
            ## outcome crispification
            # sig_vals: [batch_size, num_node, 2]
            sig_vals = self.diff_argmax(
                torch.cat((comp, torch.zeros((input_data.size(0), self.layers.size(0), 1)).to(comp.device)), dim=-1))

            sig_vals = torch.narrow(sig_vals, 2, 0, 1).squeeze(-1)
        else:
            sig_vals = self.sig(comp)
        # sig_vals: [batch_size, num_node]
        sig_vals = sig_vals.view(input_data.size(0), -1)
        # one_minus_sig: [batch_size, num_node]
        one_minus_sig = torch.ones(sig_vals.size()).to(sig_vals.device)
        one_minus_sig = torch.sub(one_minus_sig, sig_vals)

        # left_path_probs: [num_leaves, num_nodes]
        left_path_probs = self.left_path_sigs.t()
        right_path_probs = self.right_path_sigs.t()
        # left_path_probs: [batch_size, num_leaves, num_nodes]
        left_path_probs = left_path_probs.expand(input_data.size(0), *left_path_probs.size()) * sig_vals.unsqueeze(
            1)
        right_path_probs = right_path_probs.expand(input_data.size(0),
                                                   *right_path_probs.size()) * one_minus_sig.unsqueeze(1)
        # left_path_probs: [batch_size, num_nodes, num_leaves]
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)

        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        left_filler = torch.zeros(self.left_path_sigs.size()).to(left_path_probs.device)
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size()).to(left_path_probs.device)
        right_filler[self.right_path_sigs == 0] = 1

        # left_path_probs: [batch_size, num_nodes, num_leaves]
        left_path_probs = left_path_probs.add(left_filler)
        right_path_probs = right_path_probs.add(right_filler)

        # probs: [batch_size, 2*num_nodes, num_leaves]
        probs = torch.cat((left_path_probs, right_path_probs), dim=1)
        # probs: [batch_size, num_leaves]
        probs = probs.prod(dim=1)
        mus = probs.mm(self.action_mus)
        return mus
        #return mus if self.is_value else self.softmax(mus)

    def predict(self, observation):
        # [0.0137, -0.0230, -0.0459, -0.0483]
        observation = torch.from_numpy(observation).to(self.device).float()
        observation = observation.unsqueeze(0)
        logits = self.forward(observation)
        return logits.argmax(dim=-1).cpu().numpy()[0]

    def predict_proba(self, observation):
        # [0.0137, -0.0230, -0.0459, -0.0483]
        observation = torch.from_numpy(observation).to(self.device).float()
        observation = observation.unsqueeze(0)
        logits = self.forward(observation)
        return F.softmax(logits).multinomial(1).item()
