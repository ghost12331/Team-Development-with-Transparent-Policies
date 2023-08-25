import os
import sys
from collections import Counter
import time

from ipm.models.idct import IDCT
import ipm.algos.idct_ppo_policy
from overcooked.overcooked_envs import OvercookedPlayWithFixedPartner

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    pass
else:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import FlattenExtractor
from ipm.bin.utils import CheckpointCallbackWithRew
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3 import PPO


def load_idct_from_torch(filepath, input_dim, output_dim, device, randomize=True, only_optimize_leaves=True, use_ego=False):
    if use_ego:
        model = torch.load(filepath)['ego_state_dict'] # old is alt, use ego because of FCP
    else:
        model = torch.load(filepath)['alt_state_dict']
    try:
        layers = model['action_net.layers']
        comparators = model['action_net.comparators']
        alpha = model['action_net.alpha']
        # assuming an symmetric tree here
        n_nodes, n_feats = layers.shape
        assert n_feats == input_dim

        action_mus = model['action_net.action_mus']
    except:
        layers = model['layers']
        comparators = model['comparators']
        alpha = model['alpha']
        # assuming an symmetric tree here
        n_nodes, n_feats = layers.shape
        assert n_feats == input_dim

        action_mus = model['action_mus']
    n_leaves, _ = action_mus.shape
    if not randomize:
        # why only optimize leaves true here?
        if only_optimize_leaves:
            idct = IDCT(input_dim=input_dim, output_dim=output_dim, leaves=n_leaves, hard_node=True, device=device,
                        argmax_tau=1.0,
                        alpha=alpha, comparators=comparators, weights=layers, only_optimize_leaves=True)
            idct.action_mus.to(device)
            idct.action_mus = nn.Parameter(action_mus, requires_grad=True)
            idct.update_leaf_init_information()
            idct.action_mus.to(device)
        else:
            idct = IDCT(input_dim=input_dim, output_dim=output_dim, leaves=n_leaves, hard_node=True, device=device,
                        argmax_tau=1.0,
                        alpha=alpha, comparators=comparators, weights=layers, only_optimize_leaves=False)
            idct.action_mus.to(device)
            idct.action_mus = nn.Parameter(action_mus, requires_grad=True)
            idct.update_leaf_init_information()
            idct.action_mus.to(device)
    else:
        # again why is hard note false when we randomize
        idct = IDCT(input_dim=input_dim, output_dim=output_dim, leaves=n_leaves, hard_node=True, device=device,
                    argmax_tau=1.0,
                    alpha=None, comparators=None, weights=None, only_optimize_leaves=False)
    return idct


class RobotModel:
    def __init__(self, layout, idct_policy_filepath, human_policy,
                 input_dim, output_dim, randomize_initial_idct=False, only_optimize_leaves=True, with_key=False):

        device = torch.device("cpu")
        if layout == 'tutorial':
            use_ego = False
        else:
            use_ego = True
        self.robot_idct_policy = load_idct_from_torch(idct_policy_filepath, input_dim, output_dim,
                                                      device=device, randomize=randomize_initial_idct,
                                                      only_optimize_leaves=only_optimize_leaves, use_ego=use_ego)
        self.robot_idct_policy.to(device)
        self.human_policy = human_policy
        self.layout = layout

        self.player_idx = 0
        if not self.layout == "two_rooms_narrow":
            self.action_mapping = {
                "Nothing": 0,
                "Picking Up Onion From Dispenser": 1,
                "Picking Up Onion From Counter": 2,
                "Picking Up Dish From Dispenser": 3,
                "Picking Up Dish From Counter": 4,
                "Picking Up Soup From Pot": 5,
                "Picking Up Soup From Counter": 6,
                "Serving At Dispensary": 7,
                "Bringing To Closest Pot": 8,
                "Placing On Closest Counter": 9,
            }
            self.intent_mapping = {
                "Picking Up Onion From Dispenser": 0,  # picking up ingredient
                "Picking Up Onion From Counter": 0,  # picking up ingredient
                "Picking Up Dish From Dispenser": 1,  # picking up dish
                "Picking Up Dish From Counter": 1,  # picking up dish
                "Picking Up Soup From Counter": 2,  # picking up soup
                "Picking Up Soup From Pot": 2,  # picking up soup
                "Serving At Dispensary": 3,  # serving dish
                "Bringing To Closest Pot": 4,  # placing item down
                "Placing On Closest Counter": 4,  # placing item down
                "Nothing": 5,
            }
        else:
            self.action_mapping = {
                "Nothing": 0,
                "Picking Up Onion From Dispenser": 1,
                "Picking Up Onion From Counter": 2,
                "Picking Up Tomato From Dispenser": 3,
                "Picking Up Tomato From Counter": 4,
                "Picking Up Dish From Dispenser": 5,
                "Picking Up Dish From Counter": 6,
                "Picking Up Soup From Pot": 7,
                "Picking Up Soup From Counter": 8,
                "Serving At Dispensary": 9,
                "Bringing To Closest Pot": 10,
                "Placing On Closest Counter": 11
            }
            self.intent_mapping = {
                "Picking Up Onion From Dispenser": 0,  # picking up ingredient
                "Picking Up Onion From Counter": 0,  # picking up ingredient
                "Picking Up Tomato From Dispenser": 1,  # picking up ingredient
                "Picking Up Tomato From Counter": 1,  # picking up ingredient
                "Picking Up Dish From Dispenser": 2,  # picking up dish
                "Picking Up Dish From Counter": 2,  # picking up dish
                "Picking Up Soup From Counter": 3,  # picking up soup
                "Picking Up Soup From Pot": 3,  # picking up soup
                "Serving At Dispensary": 4,  # serving dish
                "Bringing To Closest Pot": 5,  # placing item down
                "Placing On Closest Counter": 5,  # placing item down
                "Nothing": 6,
            }

        self.env = OvercookedPlayWithFixedPartner(partner=self.human_policy,
                                                  layout_name=layout,
                                                  reduced_state_space_ego=True, reduced_state_space_alt=True,
                                                  use_skills_ego=True, use_skills_alt=True,
                                                  failed_skill_rew=0)

        self.mdp = self.env.mdp
        self.base_env = self.env.base_env

    def translate_recent_data_to_labels(self, recent_data_loc):
        """
        For now, assumes one trajectory
        Args:
            recent_data_loc:

        Returns:

        """
        # uncomment line below if you just want to skip through everything
        # recent_data_loc = 'data/experiments/human_modifies_tree/user_4/forced_coordination/iteration_0.tar'
        recent_data = torch.load(recent_data_loc)
        reduced_observations_human = recent_data['human_obs']
        reduced_observations_AI = recent_data['AI_obs']
        actions = recent_data['human_action']
        trajectory_states = recent_data['states']
        traj_lengths = len(trajectory_states)
        self.reduced_observations_human = reduced_observations_human
        self.reduced_observations_AI = reduced_observations_AI

        # for each trajectory in this data set
        for k in range(1):
            # go through and find all the indices where the action is 5
            indices = [i for i in range(len(actions)) if actions[i] == 5]

            if indices[-1] == traj_lengths - 1:
                # if last action is an interact, then there will be no next timestep.
                indices.remove(indices[-1])
            indices_array = np.array(indices)
            episode_observations = []
            episode_observations_reduced = []
            episode_observations_reduced_no_intent = []
            episode_high_level_actions = []
            episode_intents = []
            episode_primitive_actions = []
            episode_action_dict = {
            }
            for e, i in enumerate(indices):
                before_state = trajectory_states[i]
                after_state = trajectory_states[i + 1]

                before_object = before_state.players[self.player_idx].held_object
                if before_object is None:
                    before_object = "nothing"
                else:
                    before_object = before_object.name
                after_object = after_state.players[self.player_idx].held_object
                if after_object is None:
                    after_object = "nothing"
                else:
                    after_object = after_object.name

                def item_is_on_counter(state, item_str):
                    item_on_counter = 0
                    for key, obj in state.objects.items():
                        if obj.name == item_str:
                            item_on_counter = 1
                    return item_on_counter

                onion_on_counter_before = item_is_on_counter(before_state, 'onion')
                onion_on_counter_after = item_is_on_counter(after_state, 'onion')
                soup_on_counter_before = item_is_on_counter(before_state, 'soup')
                soup_on_counter_after = item_is_on_counter(after_state, 'soup')
                dish_on_counter_before = item_is_on_counter(before_state, 'dish')
                dish_on_counter_after = item_is_on_counter(after_state, 'dish')
                tomato_on_counter_before = item_is_on_counter(before_state, 'tomato')
                tomato_on_counter_after = item_is_on_counter(after_state, 'tomato')

                def get_num_steps_to_loc(state, loc_name):

                    if loc_name == 'onion_dispenser':
                        obj_loc = self.mdp.get_onion_dispenser_locations()
                    elif loc_name == 'tomato_dispenser':
                        obj_loc = self.mdp.get_tomato_dispenser_locations()
                    elif loc_name == 'dish_dispenser':
                        obj_loc = self.mdp.get_dish_dispenser_locations()
                    elif loc_name == 'soup_pot':
                        potential_locs = self.mdp.get_pot_locations()
                        obj_loc = []
                        for pos in potential_locs:
                            if self.base_env.mdp.soup_ready_at_location(state, pos):
                                obj_loc.append(pos)
                    elif loc_name == 'serve':
                        obj_loc = self.mdp.get_serving_locations()
                    elif loc_name == 'pot':
                        obj_loc = self.mdp.get_pot_locations()
                    else:
                        raise 'Unknown location name'

                    pos_and_or = state.players[self.player_idx].pos_and_or
                    min_dist = np.Inf

                    for loc in obj_loc:
                        results = self.base_env.mlam.motion_planner.motion_goals_for_pos[loc]
                        for result in results:
                            if self.base_env.mlam.motion_planner.positions_are_connected(pos_and_or, result):
                                plan = self.base_env.mp._get_position_plan_from_graph(pos_and_or, result)
                                plan_results = self.base_env.mp.action_plan_from_positions(plan, pos_and_or, result)
                                curr_dist = len(plan_results[1])
                                if curr_dist < min_dist:
                                    min_dist = curr_dist
                    return min_dist

                n_steps_onion_dispenser_before = get_num_steps_to_loc(before_state, 'onion_dispenser')
                n_steps_tomato_dispenser_before = get_num_steps_to_loc(before_state, 'tomato_dispenser')
                n_steps_dish_dispenser_before = get_num_steps_to_loc(before_state, 'dish_dispenser')
                n_steps_soup_pot_before = get_num_steps_to_loc(before_state, 'soup_pot')
                n_steps_pot_before = get_num_steps_to_loc(before_state, 'pot')
                n_steps_serve_before = get_num_steps_to_loc(before_state, 'serve')

                if after_object == 'onion' and before_object == "nothing":
                    if n_steps_onion_dispenser_before == 1:
                        action_taken = "Picking Up Onion From Dispenser"
                    else:
                        action_taken = "Picking Up Onion From Counter"
                elif after_object == 'tomato' and before_object == "nothing":
                    if n_steps_tomato_dispenser_before == 1:
                        action_taken = "Picking Up Tomato From Dispenser"
                    else:
                        action_taken = "Picking Up Tomato From Counter"
                elif after_object == 'soup' and before_object == "dish":
                    if n_steps_soup_pot_before == 1:
                        action_taken = "Picking Up Soup From Pot"
                    else:
                        print('WARNING: Soup was picked up somehow even though we were not at the pot')
                        action_taken = "Picking Up Soup From Pot"
                elif after_object == 'dish' and before_object == "nothing":
                    if n_steps_dish_dispenser_before == 1:
                        action_taken = "Picking Up Dish From Dispenser"
                    else:
                        action_taken = "Picking Up Dish From Counter"
                elif after_object == 'nothing' and before_object == "onion":
                    if n_steps_pot_before == 1:
                        action_taken = "Bringing To Closest Pot"
                    else:
                        action_taken = "Placing On Closest Counter"
                elif after_object == 'nothing' and before_object == "tomato":
                    if n_steps_pot_before == 1:
                        action_taken = "Bringing To Closest Pot"
                    else:
                        action_taken = "Placing On Closest Counter"
                elif after_object == 'nothing' and before_object == "dish":
                    action_taken = "Placing On Closest Counter"
                elif after_object == 'nothing' and before_object == "soup":
                    if n_steps_serve_before == 1:
                        action_taken = "Serving At Dispensary"
                    else:
                        action_taken = 'Placing On Closest Counter'
                else:
                    # check if timer was put on
                    action_taken = 'Nothing'

                # high_level action
                episode_action_dict[i] = action_taken

            # go through a second time and pair each observation with action
            for timestep in range(len(trajectory_states)):
                try:
                    next_action = indices_array[indices_array > timestep].min()
                except:
                    # no next action
                    continue
                # episode_observations.append(reduced_observations[timestep])
                episode_observations_reduced.append(
                    [reduced_observations_human[timestep], reduced_observations_AI[timestep]])
                episode_observations_reduced_no_intent.append(
                    [reduced_observations_human[timestep][:int(self.intent_input_dim_size / 2)],
                     reduced_observations_AI[timestep][:int(self.intent_input_dim_size / 2)]])
                episode_primitive_actions.append(actions[timestep])
                episode_high_level_actions.append(self.action_mapping[episode_action_dict[next_action]])
                print('At timestep ', timestep, ' the action is ', episode_action_dict[next_action])
                # TODO: check the mapping below.
                episode_intents.append(self.intent_mapping[episode_action_dict[next_action]])

        self.episode_high_level_actions = episode_high_level_actions
        self.episode_intents = episode_intents
        self.episode_primitive_actions = episode_primitive_actions
        self.episode_observations_reduced = episode_observations_reduced
        self.episode_observations_reduced_no_intent = episode_observations_reduced_no_intent

        # print distribution for self.training_intents and self.training_actions
        print("Distribution of intents: ", Counter(self.episode_intents))
        print("Distribution of actions: ", Counter(self.episode_high_level_actions))
        print("Distribution of primitives: ", Counter(self.episode_primitive_actions))

    def finetune_robot_idct_policy_parallel(self):
        import time
        start_time = time.time()
        if self.layout == 'forced_coordination':
            os.system('./eval_hyperparams_parallel_fc.sh')
            # will save into seeds 2000 and 2001
        elif self.layout == 'two_rooms':
            os.system('./eval_hyperparams_parallel_2r.sh')
        else:
            os.system('./eval_hyperparams_parallel_narrow.sh')
            # will save into seeds 4000 and 4001

        # change this to a timer that ensures models are done training
        while True:
            end_time = time.time()
            time.sleep(1)
            print(end_time - start_time)
            if end_time - start_time > 60 * 4.5:
                break

        if self.layout == 'forced_coordination':
            paths = [
                'robot_policy_optimization_1_seed_2000_best.tar',
                'robot_policy_optimization_1_seed_2001_best.tar']
        else:
            paths = ['robot_policy_optimization_1_seed_4000_best.tar',
                     'robot_policy_optimization_1_seed_4001_best.tar']
        results = []
        model_info = []

        model_A = torch.load(paths[0])
        model_B = torch.load(paths[1])

        if model_A['best_reward'] > model_B['best_reward']:
            best_model = model_A
            print('loading in model A with reward,', model_A['best_reward'])
        else:
            best_model = model_B
            print('loading in model B with reward,', model_B['best_reward'])

        self.robot_idct_policy.load_state_dict(best_model['alt_state_dict_tree'])

    def compare_models(self, printer, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print(printer, 'Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print(printer, 'Models match perfectly! :)')

    def finetune_robot_idct_policy(self,
                                   rl_n_steps=70000,
                                   rl_learning_rate=0.0003,
                                   ga_depth=2,
                                   ga_n_gens=100,
                                   ga_n_pop=30,
                                   ga_n_parents_mating=15,
                                   ga_crossover_prob=0.5,
                                   ga_crossover_type="two_points",
                                   ga_mutation_prob=0.2,
                                   ga_mutation_type="random",
                                   recent_data_file='data/11_trajs_tar',
                                   algorithm_choice='ga+rl',
                                   unique_id=None
                                   ):

        checkpoint_freq = rl_n_steps // 100
        save_models = True
        ego_idx = 1  # robot policy is always the second player

        seed = 0

        env = OvercookedPlayWithFixedPartner(partner=self.human_policy,
                                             layout_name=self.layout,
                                             seed_num=seed,
                                             ego_idx=ego_idx,
                                             reduced_state_space_ego=True,
                                             reduced_state_space_alt=True,
                                             use_skills_ego=True,
                                             use_skills_alt=True,
                                             use_true_intent_ego=True,
                                             use_true_intent_alt=True)

        initial_model_path = os.path.join('data', self.layout, 'robot_online_optimization', 'initial_model.zip')
        medium_model_path = os.path.join('data', self.layout, 'robot_online_optimization', 'medium_model.zip')
        final_model_path = os.path.join('data', self.layout, 'robot_online_optimization', 'final_model.zip')

        if unique_id is None:
            save_dir = os.path.join('data', self.layout, 'robot_online_optimization', 'robot_idct_policy')
        else:
            save_dir = os.path.join('data', self.layout, 'robot_online_optimization', 'robot_idct_policy' + unique_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint_callback = CheckpointCallbackWithRew(
            n_steps=rl_n_steps,
            save_freq=checkpoint_freq,
            save_path=save_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            initial_model_path=initial_model_path,
            medium_model_path=medium_model_path,
            final_model_path=final_model_path,
            save_model=save_models,
            verbose=1
        )

        full_save_dir = "./" + save_dir + "/"
        if not os.path.exists(full_save_dir):
            os.makedirs(full_save_dir)

        env = Monitor(env, full_save_dir)

        input_dim = get_obs_shape(env.observation_space)[0]
        output_dim = env.n_actions_ego
        seed = 1

        device = torch.device("cpu")

        old_action_mus = self.robot_idct_policy.action_mus.clone()
        old_weights = self.robot_idct_policy.layers.clone()
        if 'rl' in algorithm_choice:
            torch.set_num_threads(3)
            model = self.robot_idct_policy

            ppo_lr = rl_learning_rate
            ppo_batch_size = 64
            ppo_n_steps = 1000

            # why is hard node false and use_individual alpha True
            ddt_kwargs = {
                'num_leaves': len(model.leaf_init_information),
                'hard_node': True,
                'weights': model.layers,
                'alpha': model.alpha,
                'comparators': model.comparators,
                'leaves': model.leaf_init_information,
                'fixed_idct': False,
                'device': device,
                'argmax_tau': 1.0,
                'ddt_lr': 0.001,  # this param is irrelevant for the IDCT
                'use_individual_alpha': False,
                'l1_reg_coeff': 1.0,
                'l1_reg_bias': 1.0,
                'l1_hard_attn': 1.0,
                'use_gumbel_softmax': False,
                'alg_type': 'ppo'
            }

            features_extractor = FlattenExtractor
            policy_kwargs = dict(features_extractor_class=features_extractor, ddt_kwargs=ddt_kwargs)
            agent = PPO("IDCT_PPO_Policy", env,
                        n_steps=ppo_n_steps,
                        # batch_size=args.batch_size,
                        # buffer_size=args.buffer_size,
                        learning_rate=ppo_lr,
                        policy_kwargs=policy_kwargs,
                        tensorboard_log='log',
                        gamma=0.99,
                        verbose=1,
                        device=device
                        # seed=1
                        )

            # loading in value function
            initial_policy_path = os.path.join('data', 'prior_tree_policies', self.layout + '.tar')
            initial_policy = torch.load(initial_policy_path)
            current_weights = agent.policy.state_dict()
            for k in ['mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.0.bias',
                      'mlp_extractor.policy_net.2.weight', 'mlp_extractor.policy_net.2.bias',
                      'mlp_extractor.value_net.0.weight', 'mlp_extractor.value_net.0.bias',
                      'mlp_extractor.value_net.2.weight', 'mlp_extractor.value_net.2.bias', 'value_net.weight',
                      'value_net.bias']:
                current_weights[k] = initial_policy['alt_state_dict'][k]

            agent.policy.load_state_dict(current_weights)

            agent.policy.action_net.layers.requires_grad = self.robot_idct_policy.layers.requires_grad
            agent.policy.action_net.action_mus.requires_grad = self.robot_idct_policy.action_mus.requires_grad
            agent.policy.action_net.comparators.requires_grad = self.robot_idct_policy.comparators.requires_grad
            agent.policy.action_net.alpha.requires_grad = self.robot_idct_policy.alpha.requires_grad
            print(f'Agent training...')
            # timer
            start_time = time.time()
            agent.learn(total_timesteps=rl_n_steps, callback=checkpoint_callback)
            end_time = time.time()
            print(f'Training took {end_time - start_time} seconds')
            print(f'Finished training agent with best average reward of {checkpoint_callback.best_mean_reward}')

            agent.policy.load_state_dict(checkpoint_callback.final_model_weights)
            self.robot_idct_policy.load_state_dict(agent.policy.action_net.state_dict())

            new_action_mus = self.robot_idct_policy.action_mus.clone()
            new_weights = self.robot_idct_policy.layers.clone()

            print('old matches new leaves', new_action_mus.eq(old_action_mus).all())
            print('old matches new weights', new_weights.eq(old_weights).all())
            print('done')

    def predict(self, obs):
        """
        Args:
            obs: observation from environment

        Returns:
            action: action to take
        """
        # reshape into a torch batch of 1
        observation = torch.from_numpy(obs).to(self.robot_idct_policy.device).float()
        observation = observation.unsqueeze(0)
        logits = self.robot_idct_policy.forward(observation)
        return F.softmax(logits).multinomial(1).item()
