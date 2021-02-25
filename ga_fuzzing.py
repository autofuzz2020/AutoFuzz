import sys
import os
sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')
sys.path.append('.')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('scenario_runner')
sys.path.append('carla_project')
sys.path.append('carla_project/src')





from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination, NoDuplicateElimination

from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.model.evaluator import Evaluator

from pymoo.algorithms.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.algorithms.random import RandomAlgorithm

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.performance_indicator.hv import Hypervolume

import matplotlib.pyplot as plt

from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

from customized_utils import create_transform, rand_real,  convert_x_to_customized_data, make_hierarchical_dir, exit_handler, arguments_info, is_critical_region, setup_bounds_mask_labels_distributions_stage1, setup_bounds_mask_labels_distributions_stage2, customize_parameters, customized_bounds_and_distributions, static_general_labels, pedestrian_general_labels, vehicle_general_labels, waypoint_labels, waypoints_num_limit, if_violate_constraints, customized_routes, parse_route_and_scenario, get_distinct_data_points, is_similar, check_bug, is_distinct, filter_critical_regions, estimate_objectives, correct_travel_dist, encode_fields, remove_fields_not_changing, get_labels_to_encode, customized_fit, customized_standardize, customized_inverse_standardize, decode_fields, encode_bounds, recover_fields_not_changing, eliminate_duplicates_for_list, process_X, inverse_process_X, determine_y_upon_weights, calculate_rep_d, select_batch_max_d_greedy, if_violate_constraints_vectorized, is_distinct_vectorized, eliminate_repetitive_vectorized, get_sorted_subfolders, load_data, choose_weight_inds, get_F, get_unique_bugs

from collections import deque


import numpy as np
import carla

from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.customized.object_params import Static, Pedestrian, Vehicle

import traceback
import json
import re
import time
from datetime import datetime

import pathlib
from distutils.dir_util import copy_tree
import dill as pickle
# import pickle
import argparse
import atexit
import traceback
import math



import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.model.individual import Individual
from pymoo.model.repair import Repair
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_crossover, get_mutation
from pymoo.model.mating import Mating

from dask.distributed import Client, LocalCluster

from pymoo.model.initialization import Initialization
from pymoo.model.duplicate import NoDuplicateElimination
from pymoo.operators.sampling.random_sampling import FloatRandomSampling

from pymoo.model.survival import Survival
from pymoo.model.individual import Individual
from distutils.dir_util import copy_tree

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from pgd_attack import pgd_attack, train_net, train_regression_net
from scipy.stats import rankdata


default_objective_weights = np.array([-1., 1., 1., 1., 1., -1., 0., 0., 0., -1.])
default_objectives = np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.])
default_check_unique_coeff = [0, 0.1, 0.5]

parser = argparse.ArgumentParser()
parser.add_argument('-p','--ports', nargs='+', type=int, default=[2003, 2006], help='TCP port(s) to listen to (default: 2003 2006)')
parser.add_argument("-s", "--scheduler_port", type=int, default=8785)
parser.add_argument("-d", "--dashboard_address", type=int, default=8786)
parser.add_argument("-r", "--route_type", type=str, default='town05_right_0')
parser.add_argument("-c", "--scenario_type", type=str, default='default')
parser.add_argument('-a','--algorithm_name', type=str, default='nsga2')
parser.add_argument("-m", "--ego_car_model", type=str, default='lbc')
parser.add_argument("--has_display", type=str, default='0')
parser.add_argument("--root_folder", type=str, default='run_results')

parser.add_argument("--episode_max_time", type=int, default=60)
parser.add_argument("--n_gen", type=int, default=2)
parser.add_argument("--pop_size", type=int, default=50)
parser.add_argument("--survival_multiplier", type=int, default=1)
parser.add_argument("--n_offsprings", type=int, default=300)
parser.add_argument("--has_run_num", type=int, default=1000)
parser.add_argument("--outer_iterations", type=int, default=3)
parser.add_argument('--objective_weights', nargs='+', type=float, default=default_objective_weights)
parser.add_argument('--check_unique_coeff', nargs='+', type=float, default=default_check_unique_coeff)
parser.add_argument('--use_single_objective', type=int, default=1)
parser.add_argument('--rank_mode', type=str, default='none')
parser.add_argument('--ranking_model', type=str, default='nn_pytorch')
parser.add_argument('--initial_fit_th', type=int, default=100)
parser.add_argument('--min_bug_num_to_fit_dnn', type=int, default=10)

parser.add_argument('--pgd_eps', type=float, default=1.01)
parser.add_argument('--adv_conf_th', type=float, default=-4)
parser.add_argument('--attack_stop_conf', type=float, default=0.9)
parser.add_argument('--use_single_nn', type=int, default=1)

parser.add_argument('--uncertainty', type=str, default='')
parser.add_argument('--model_type', type=str, default='one_output')


parser.add_argument('--explore_iter_num', type=int, default=2)
parser.add_argument('--exploit_iter_num', type=int, default=1)
parser.add_argument('--high_conf_num', type=int, default=60)
parser.add_argument('--low_conf_num', type=int, default=60)

parser.add_argument('--warm_up_path', type=str, default=None)
parser.add_argument('--warm_up_len', type=int, default=-1)
parser.add_argument('--use_alternate_nn', type=int, default=0)
parser.add_argument('--diversity_mode', type=str, default='none')

parser.add_argument('--regression_nn_use_running_data', type=int, default=1)

parser.add_argument('--adv_exploitation_only', type=int, default=0)

parser.add_argument('--sample_multiplier', type=int, default=200)
parser.add_argument('--mating_max_iterations', type=int, default=200)

parser.add_argument('--uncertainty_exploration', type=str, default='confidence')

parser.add_argument('--only_run_unique_cases', type=int, default=1)

parser.add_argument('--consider_interested_bugs', type=int, default=1)

parser.add_argument('--record_every_n_step', type=int, default=2000)

arguments = parser.parse_args()


ports = arguments.ports
scheduler_port = arguments.scheduler_port
dashboard_address = arguments.dashboard_address

# ['town01_left_0', 'town07_front_0', 'town05_front_0', 'town05_right_0']
route_type = arguments.route_type
# ['default', 'leading_car_braking', 'vehicles_only', 'no_static']
scenario_type = arguments.scenario_type
# [random', 'nsga2', 'nsga2-dt', 'nsga2-un-dt', 'nsga2-emcmc', 'nsga2-un', 'nsga2-un-emcmc', 'random-un', 'nn', 'adv_nn']
algorithm_name = arguments.algorithm_name
# ['lbc', 'auto_pilot', 'pid_agent']
ego_car_model = arguments.ego_car_model

# ['none', 'nn', 'adv', 'inversion']
rank_mode = arguments.rank_mode
# ['sklearn', 'pytorch']
ranking_model = arguments.ranking_model
initial_fit_th = arguments.initial_fit_th
min_bug_num_to_fit_dnn = arguments.min_bug_num_to_fit_dnn
pgd_eps = arguments.pgd_eps
adv_conf_th = arguments.adv_conf_th
attack_stop_conf = arguments.attack_stop_conf
use_single_nn = arguments.use_single_nn
uncertainty = arguments.uncertainty

model_type = arguments.model_type
# ['BNN', 'one_output']
# BALD and BatchBALD only support BNN
if uncertainty.split('_')[0] in ['BALD', 'BatchBALD']:
    model_type = 'BNN'

explore_iter_num = arguments.explore_iter_num
exploit_iter_num = arguments.exploit_iter_num
high_conf_num = arguments.high_conf_num
low_conf_num = arguments.low_conf_num

regression_nn_use_running_data = arguments.regression_nn_use_running_data

adv_exploitation_only = arguments.adv_exploitation_only

warm_up_path = arguments.warm_up_path
warm_up_len = arguments.warm_up_len
use_alternate_nn = arguments.use_alternate_nn
# ['none', 'nn_rep']
diversity_mode = arguments.diversity_mode

sample_multiplier = arguments.sample_multiplier
mating_max_iterations = arguments.mating_max_iterations

uncertainty_exploration = arguments.uncertainty_exploration

os.environ['HAS_DISPLAY'] = arguments.has_display
root_folder = arguments.root_folder


episode_max_time = arguments.episode_max_time
global_n_gen = arguments.n_gen

pop_size = arguments.pop_size
survival_multiplier = arguments.survival_multiplier
n_offsprings = arguments.n_offsprings
# only used when algorithm_name is nsga2-dt or nsga2-un-dt
outer_iterations = arguments.outer_iterations

if 'un' in algorithm_name:
    use_unique_bugs = True
else:
    use_unique_bugs = False

if algorithm_name in ['nsga2-emcmc', 'nsga2-un-emcmc']:
    emcmc = True
else:
    emcmc = False

# [ego_linear_speed, closest_dist, offroad_d, wronglane_d, dev_dist]
# objective_weights = np.array([-1, 1, 1, 1, -1])
objective_weights = np.array(arguments.objective_weights)

check_unique_coeff = arguments.check_unique_coeff

has_run_num = arguments.has_run_num
finish_after_has_run = True

only_run_unique_cases = arguments.only_run_unique_cases

use_single_objective = arguments.use_single_objective
consider_interested_bugs = arguments.consider_interested_bugs
record_every_n_step = arguments.record_every_n_step

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import random
import torch
import numpy as np
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False








random_seeds = [10, 20, 30]
rng = np.random.default_rng(random_seeds[0])

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

scenario_folder = 'scenario_files'
if not os.path.exists('scenario_files'):
    os.mkdir(scenario_folder)
scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

# This is used to control how this program use GPU
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

resume_run = False
save = True
save_path = 'ga_intermediate.pkl'


max_running_time = 3600*24

# ['generations', 'max_time']
global_termination_condition = 'generations'
















'''
for customizing weather choices, static_types, pedestrian_types, vehicle_types, and vehicle_colors, make changes to object_types.py
'''



class MyProblem(Problem):

    def __init__(self, elementwise_evaluation, bug_parent_folder, non_bug_parent_folder, town_name, scenario, direction, route_str, scenario_file, ego_car_model, scheduler_port, dashboard_address, customized_config, ports=[2000], episode_max_time=10000, customized_parameters_distributions={}, customized_center_transforms={}, call_from_dt=False, dt=False, estimator=None, critical_unique_leaves=None, cumulative_info=None, objective_weights=default_objective_weights, check_unique_coeff=default_check_unique_coeff, consider_interested_bugs=1, record_every_n_step=2000, use_single_objective=True):

        customized_parameters_bounds = customized_config['customized_parameters_bounds']
        customized_parameters_distributions = customized_config['customized_parameters_distributions']
        customized_center_transforms = customized_config['customized_center_transforms']
        customized_constraints = customized_config['customized_constraints']


        self.objective_weights = objective_weights
        self.customized_constraints = customized_constraints

        self.call_from_dt = call_from_dt
        self.dt = dt
        self.estimator = estimator
        self.critical_unique_leaves = critical_unique_leaves


        self.objectives_list = []
        self.x_list = []
        self.y_list = []
        self.F_list = []

        self.use_single_objective = use_single_objective
        self.consider_interested_bugs = consider_interested_bugs
        self.record_every_n_step = record_every_n_step

        self.scheduler_port = scheduler_port
        self.dashboard_address = dashboard_address
        self.ports = ports
        self.episode_max_time = episode_max_time



        self.bug_folder = bug_parent_folder
        self.non_bug_folder = non_bug_parent_folder


        self.town_name = town_name
        self.scenario = scenario
        self.direction = direction
        self.route_str = route_str
        self.scenario_file = scenario_file
        self.ego_car_model = ego_car_model

        self.cumulative_info = cumulative_info

        if cumulative_info:
            self.counter = cumulative_info['counter']
            self.has_run = cumulative_info['has_run']
            self.start_time = cumulative_info['start_time']

            self.time_list = cumulative_info['time_list']

            self.bugs = cumulative_info['bugs']
            self.unique_bugs = cumulative_info['unique_bugs']
            self.interested_unique_bugs = cumulative_info['interested_unique_bugs']

            self.bugs_type_list = cumulative_info['bugs_type_list']
            self.bugs_inds_list = cumulative_info['bugs_inds_list']

            self.bugs_num_list = cumulative_info['bugs_num_list']
            self.unique_bugs_num_list = cumulative_info['unique_bugs_num_list']
            self.has_run_list = cumulative_info['has_run_list']
        else:
            self.counter = 0
            self.has_run = 0
            self.start_time = time.time()

            self.time_list = []
            self.bugs = []
            self.unique_bugs = []
            self.interested_unique_bugs = []

            self.bugs_type_list = []
            self.bugs_inds_list = []

            self.bugs_num_list = []
            self.unique_bugs_num_list = []
            self.has_run_list = []




        fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels = setup_bounds_mask_labels_distributions_stage1()
        customize_parameters(parameters_min_bounds, customized_parameters_bounds)
        customize_parameters(parameters_max_bounds, customized_parameters_bounds)


        fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels, parameters_distributions, n_var = setup_bounds_mask_labels_distributions_stage2(fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels)
        customize_parameters(parameters_min_bounds, customized_parameters_bounds)
        customize_parameters(parameters_max_bounds, customized_parameters_bounds)
        customize_parameters(parameters_distributions, customized_parameters_distributions)




        for d in [fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds]:
            for k, v in d.items():
                assert not hasattr(self, k), k+'should not appear twice.'
                setattr(self, k, v)


        xl = [pair[1] for pair in parameters_min_bounds.items()]
        xu = [pair[1] for pair in parameters_max_bounds.items()]


        self.parameters_min_bounds = parameters_min_bounds
        self.parameters_max_bounds = parameters_max_bounds
        self.mask = mask
        self.labels = labels
        self.parameters_distributions = parameters_distributions
        self.customized_center_transforms = customized_center_transforms

        self.p, self.c, self.th = check_unique_coeff
        # self.th = int(len(self.labels) * th)
        self.check_unique_coeff = (self.p, self.c, self.th)

        self.launch_server = True

        super().__init__(n_var=n_var, n_obj=4, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=elementwise_evaluation)






    def _evaluate(self, X, out, *args, **kwargs):
        objective_weights = self.objective_weights
        customized_center_transforms = self.customized_center_transforms

        waypoints_num_limit = self.waypoints_num_limit
        num_of_static_max = self.num_of_static_max
        num_of_pedestrians_max = self.num_of_pedestrians_max
        num_of_vehicles_max = self.num_of_vehicles_max

        episode_max_time = self.episode_max_time
        call_from_dt = self.call_from_dt
        bug_folder = self.bug_folder
        non_bug_folder = self.non_bug_folder

        parameters_min_bounds = self.parameters_min_bounds
        parameters_max_bounds = self.parameters_max_bounds
        labels = self.labels
        mask = self.mask
        xl = self.xl
        xu = self.xu
        customized_constraints = self.customized_constraints

        dt = self.dt
        estimator = self.estimator
        critical_unique_leaves = self.critical_unique_leaves

        mean_objectives_across_generations_path = os.path.join(self.bug_folder, 'mean_objectives_across_generations.txt')


        town_name = self.town_name
        scenario = self.scenario
        direction = self.direction
        route_str = self.route_str
        scenario_file = self.scenario_file
        ego_car_model = self.ego_car_model
        record_every_n_step = self.record_every_n_step

        all_final_generated_transforms_list = []




        def fun(x, launch_server, counter):
            not_critical_region = dt and not is_critical_region(x[:-1], estimator, critical_unique_leaves)
            violate_constraints, _ = if_violate_constraints(x, customized_constraints, labels, verbose=True)
            if not_critical_region or violate_constraints:
                objectives = default_objectives
                return objectives, None, None, None, 0, None

            else:


                # x = denormalize_by_entry(self, x)

                customized_data = convert_x_to_customized_data(x, waypoints_num_limit, num_of_static_max, num_of_pedestrians_max, num_of_vehicles_max, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)
                # print('x', x)

                # run simulation
                objectives, loc, object_type, route_completion, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model, record_every_n_step=record_every_n_step)



                # [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision]

                is_bug = check_bug(objectives)

                # change data in case the original x is used elsewhere
                data = x[:-1]
                # correct_travel_dist(data, labels, customized_data['tmp_travel_dist_file'])


                info = {**info, 'x':x, 'waypoints_num_limit':waypoints_num_limit, 'num_of_static_max':num_of_static_max, 'num_of_pedestrians_max':num_of_pedestrians_max, 'num_of_vehicles_max':num_of_vehicles_max, 'customized_center_transforms':customized_center_transforms,
                'parameters_min_bounds':parameters_min_bounds,
                'parameters_max_bounds':parameters_max_bounds}

                cur_info = {'counter':counter, 'x':x, 'data':data, 'objectives':objectives,  'loc':loc, 'object_type':object_type, 'labels':labels, 'mask':mask, 'xl':xl, 'xu':xu, 'is_bug':is_bug, 'route_completion':route_completion,
                'customized_constraints':customized_constraints, 'info': info}

                print(counter, is_bug, objectives)


                if is_bug:
                    cur_folder = make_hierarchical_dir([bug_folder, str(counter)])
                else:
                    cur_folder = make_hierarchical_dir([non_bug_folder, str(counter)])

                with open(cur_folder+'/'+'cur_info.pickle', 'wb') as f_out:
                    pickle.dump(cur_info, f_out)



                try:
                    print('save_path, cur_folder', save_path, cur_folder)
                    copy_tree(save_path, cur_folder)
                except:
                    print('fail to copy from', save_path)
                    traceback.print_exc()

                # hack:
                cur_port = int(x[-1])
                filename = 'tmp_folder/'+str(cur_port)+'.pickle'
                with open(filename, 'rb') as f_in:
                    all_final_generated_transforms = pickle.load(f_in)

                return objectives, loc, object_type, info, 1, all_final_generated_transforms




        def submit_and_run_jobs(ind_start, ind_end, launch_server, job_results):
            time_elapsed = 0
            jobs = []
            for i in range(ind_start, ind_end):
                j = i % len(self.ports)
                port = self.ports[j]
                worker = workers[j]
                x = np.concatenate([X[i], np.array([port])])
                jobs.append(client.submit(fun, x, launch_server, self.counter, workers=worker))
                print(i, self.counter)
                self.counter += 1


            for i in range(len(jobs)):
                job = jobs[i]
                cur_i = i + ind_start
                total_i = i + (self.counter-len(jobs))
                objectives, loc, object_type, info, has_run, all_final_generated_transforms_i = job.result()
                all_final_generated_transforms_list.append(all_final_generated_transforms_i)

                self.has_run_list.append(has_run)
                self.has_run += has_run
                # record bug
                if check_bug(objectives):
                    bug_str = ''
                    bug_type = 5
                    if objectives[0] > 0.1:
                        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
                        for k,v in collision_types.items():
                            if object_type in v:
                                bug_str = k
                        if not bug_str:
                            bug_str = 'unknown_collision'+'_'+object_type
                        bug_type = 1
                    elif objectives[-3]:
                        bug_str = 'offroad'
                        bug_type = 2
                    elif objectives[-2]:
                        bug_str = 'wronglane'
                        bug_type = 3
                    if objectives[-1]:
                        bug_str += 'run_red_light'
                        if bug_type > 4:
                            bug_type = 4


                    with open(mean_objectives_across_generations_path, 'a') as f_out:
                        f_out.write(str(total_i)+','+bug_str+'\n')

                    self.bugs.append(X[cur_i].astype(float))
                    self.bugs_inds_list.append(total_i)
                    self.bugs_type_list.append(bug_type)

                    self.y_list.append(bug_type)
                else:
                    self.y_list.append(0)
                # we don't want to store port number
                self.x_list.append(X[cur_i])
                self.objectives_list.append(np.array(objectives))
                job_results.append(np.array(objectives))

            # print(all_final_generated_transforms_list)

            # hack:
            with open('tmp_folder/total.pickle', 'wb') as f_out:
                pickle.dump(all_final_generated_transforms_list, f_out)


            # record time elapsed and bug numbers
            self.time_list.append(time_elapsed)



        def process_specific_bug(bug_ind):
            # print('self.bugs_type_list', self.bugs_type_list)
            chosen_bugs = np.array(self.bugs_type_list) == bug_ind

            specific_bugs = np.array(self.bugs)[chosen_bugs]
            specific_bugs_inds_list = np.array(self.bugs_inds_list)[chosen_bugs]

            # print('specific_bugs', specific_bugs)
            unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(specific_bugs, self.mask, self.xl, self.xu, self.p, self.c, self.th)
            # print('\n'*5)
            # print('self.mask, self.xl, self.xu, self.p, self.c, self.th', self.mask, self.xl, self.xu, self.p, self.c, self.th)
            # print('\n'*5)

            unique_specific_bugs_inds_list = specific_bugs_inds_list[specific_distinct_inds]

            # print(bug_ind, specific_distinct_inds, specific_bugs_inds_list, unique_specific_bugs_inds_list)

            return list(unique_specific_bugs), list(unique_specific_bugs_inds_list), len(unique_specific_bugs)




        job_results = []

        with LocalCluster(scheduler_port=self.scheduler_port, dashboard_address=self.dashboard_address, n_workers=len(self.ports), threads_per_worker=1) as cluster, Client(cluster) as client:
            workers = []
            for k in client.has_what():
                workers.append(k[len('tcp://'):])

            end_ind = np.min([len(self.ports), X.shape[0]])
            print('end_ind, X.shape[0]', end_ind, X.shape[0])
            rng = np.random.default_rng(random_seeds[1])
            submit_and_run_jobs(0, end_ind, self.launch_server, job_results)
            self.launch_server = False
            time_elapsed = time.time() - self.start_time


            if X.shape[0] > len(self.ports):
                rng = np.random.default_rng(random_seeds[2])
                submit_and_run_jobs(end_ind, X.shape[0], self.launch_server, job_results)



            current_F = get_F(job_results, self.objectives_list, objective_weights, self.use_single_objective)

            out["F"] = current_F
            self.F_list.append(current_F)



            time_elapsed = time.time() - self.start_time
            print('\n'*10)
            print('+'*100)
            mean_objectives_this_generation = np.mean(np.array(self.objectives_list[-X.shape[0]:]), axis=0)




            unique_collision_bugs, unique_collision_bugs_inds_list, unique_collision_num = process_specific_bug(1)
            unique_offroad_bugs, unique_offroad_bugs_inds_list, unique_offroad_num = process_specific_bug(2)
            unique_wronglane_bugs, unique_wronglane_bugs_inds_list, unique_wronglane_num = process_specific_bug(3)
            unique_redlight_bugs, unique_redlight_bugs_inds_list, unique_redlight_num = process_specific_bug(4)

            # print(unique_collision_bugs, unique_offroad_bugs, unique_wronglane_bugs, type(unique_collision_bugs), type(unique_offroad_bugs), type(unique_wronglane_bugs))
            self.unique_bugs = unique_collision_bugs + unique_offroad_bugs + unique_wronglane_bugs + unique_redlight_bugs


            if self.consider_interested_bugs:
                collision_activated = np.sum(objective_weights[:3] != 0) > 0
                offroad_activated = (np.abs(objective_weights[3]) > 0) | (
                    np.abs(objective_weights[5]) > 0
                )
                wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
                    np.abs(objective_weights[5]) > 0
                )
                red_light_activated = np.abs(objective_weights[-1]) > 0
                self.interested_unique_bugs = []
                if collision_activated:
                    self.interested_unique_bugs += unique_collision_bugs
                if offroad_activated:
                    self.interested_unique_bugs += unique_offroad_bugs
                if wronglane_activated:
                    self.interested_unique_bugs += unique_wronglane_bugs
                if red_light_activated:
                    self.interested_unique_bugs += unique_redlight_bugs
            else:
                self.interested_unique_bugs = self.unique_bugs

            unique_bugs_inds_list = unique_collision_bugs_inds_list + unique_offroad_bugs_inds_list + unique_wronglane_bugs_inds_list + unique_redlight_bugs_inds_list


            num_of_bugs = len(self.bugs)
            num_of_unique_bugs = len(self.unique_bugs)
            num_of_interested_unique_bugs = len(self.interested_unique_bugs)

            self.bugs_num_list.append(num_of_bugs)
            self.unique_bugs_num_list.append(num_of_unique_bugs)



            num_of_collisions = np.sum(np.array(self.bugs_type_list)==1)
            num_of_offroad = np.sum(np.array(self.bugs_type_list)==2)
            num_of_wronglane = np.sum(np.array(self.bugs_type_list)==3)
            num_of_redlight = np.sum(np.array(self.bugs_type_list)==4)


            print(self.counter, self.has_run, time_elapsed, num_of_bugs, num_of_unique_bugs, num_of_collisions, num_of_offroad, num_of_wronglane, num_of_redlight, mean_objectives_this_generation, unique_collision_num, unique_offroad_num, unique_wronglane_num, unique_redlight_num)
            print(num_of_interested_unique_bugs)
            print(self.bugs_inds_list)
            print(unique_bugs_inds_list)


            for i in range(X.shape[0]-1):
                for j in range(i+1, X.shape[0]):
                    if np.sum(X[i]-X[j])==0:
                        print(X.shape[0], i, j, 'same')

            with open(mean_objectives_across_generations_path, 'a') as f_out:
                f_out.write(','.join([str(x) for x in [self.counter, self.has_run, time_elapsed, num_of_bugs, num_of_unique_bugs, num_of_collisions, num_of_offroad, num_of_wronglane, num_of_redlight, unique_collision_num, unique_offroad_num, unique_wronglane_num, unique_redlight_num, num_of_interested_unique_bugs]]+[str(x) for x in mean_objectives_this_generation])+'\n')
                f_out.write(';'.join([str(ind) for ind in unique_bugs_inds_list])+' objective_weights : '+str(self.objective_weights)+'\n')
            print('+'*100)
            print('\n'*10)
            # os.system('sudo chmod -R 777 '+self.bug_folder)



            # save intermediate results
            if len(self.x_list) > 0:
                X = np.stack(self.x_list)
                # print(X.shape)
                # print('self.F_list', self.F_list)
                F = np.concatenate(self.F_list)
                objectives = np.stack(self.objectives_list)
            else:
                X = []
                F = []
                objectives = []

            non_dt_save_file = '_'.join([algorithm_name, route_type, scenario_type, ego_car_model, str(global_n_gen), str(pop_size)])
            pth = os.path.join(self.bug_folder, non_dt_save_file)

            np.savez(pth, X=X, y=np.array(self.y_list), F=F, objectives=objectives, time_list=np.array(self.time_list), bugs_num_list=np.array(self.bugs_num_list), unique_bugs_num_list=np.array(self.unique_bugs_num_list), has_run_list=self.has_run_list, labels=self.labels, mask=self.mask, xl=self.xl, xu=self.xu, p=self.p, c=self.c, th=self.th, route_type=route_type, scenario_type=scenario_type)
            print('npz saved')








def run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model, ego_car_model_path=None, rerun=False, record_every_n_step=2000):
    arguments = arguments_info()
    arguments.record_every_n_step = record_every_n_step
    arguments.port = customized_data['port']
    arguments.debug = 1
    if rerun:
        arguments.debug = 0



    if ego_car_model == 'lbc':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'
        arguments.agent_config = 'models/epoch=24.ckpt'
        # arguments.agent_config = 'models/stage2_0.01_augmented_epoch=11.ckpt'
        base_save_folder = 'collected_data_customized'
    elif ego_car_model == 'lbc_augment':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'
        arguments.agent_config = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/checkpoints/stage2_pretrained/town03_ga-adv-nn-un-1-1e-4/epoch=0.ckpt'
        base_save_folder = 'collected_data_lbc_augment'
    elif ego_car_model == 'auto_pilot':
        arguments.agent = 'leaderboard/team_code/auto_pilot.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_autopilot'
    elif ego_car_model == 'pid_agent':
        arguments.agent = 'scenario_runner/team_code/pid_agent.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_pid_agent'
    elif ego_car_model == 'map_model':
        arguments.agent = 'scenario_runner/team_code/map_agent.py'
        arguments.agent_config = 'models/stage1_default_50_epoch=16.ckpt'
        base_save_folder = 'collected_data_map_model'
    else:
        print('unknown ego_car_model:', ego_car_model)

    if ego_car_model_path:
        arguments.agent_config = ego_car_model_path


    if rerun:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, '/rerun', str(int(arguments.port)), str(call_from_dt)])
    else:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, str(int(arguments.port)), str(call_from_dt)])



    arguments.scenarios = scenario_file




    statistics_manager = StatisticsManager()


    # Fixed Hyperparameters
    multi_actors_scenarios = ['Scenario12']
    # sample_factor is an integer between [1, 8]
    sample_factor = 5
    weather_index = customized_data['weather_index']


    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = weather_index
    os.environ['WEATHER_INDEX'] = str(weather_index)

    town_scenario_direction = town_name + '/' + scenario

    folder_1 = os.environ['SAVE_FOLDER'] + '/' + town_name
    if not os.path.exists(folder_1):
        os.mkdir(folder_1)
    folder_2 = folder_1 + '/' + scenario
    if not os.path.exists(folder_2):
        os.mkdir(folder_2)

    if scenario in multi_actors_scenarios:
        town_scenario_direction += '/' + direction
        folder_2 += '/' + direction
        if not os.path.exists(folder_2):
            os.mkdir(folder_2)

    os.environ['SAVE_FOLDER'] = folder_2
    arguments.save_folder = os.environ['SAVE_FOLDER']

    route_prefix = 'leaderboard/data/customized_routes/' + town_scenario_direction + '/route_'

    arguments.routes = route_prefix + route_str + '.xml'
    os.environ['ROUTES'] = arguments.routes

    save_path = os.path.join(arguments.save_folder, 'route_'+route_str)

    # TBD: for convenience
    arguments.deviations_folder = save_path


    # extract waypoints along route
    import xml.etree.ElementTree as ET
    tree = ET.parse(arguments.routes)
    route_waypoints = []



    # this iteration should only go once since we only keep one route per file
    for route in tree.iter("route"):
        route_id = route.attrib['id']
        route_town = route.attrib['town']

        for waypoint in route.iter('waypoint'):
            route_waypoints.append(create_transform(float(waypoint.attrib['x']), float(waypoint.attrib['y']), float(waypoint.attrib['z']), float(waypoint.attrib['pitch']), float(waypoint.attrib['yaw']), float(waypoint.attrib['roll'])))


    # --------------------------------------------------------------------------



    customized_data['using_customized_route_and_scenario'] = True
    customized_data['destination'] = route_waypoints[-1].location
    customized_data['sample_factor'] = sample_factor
    customized_data['number_of_attempts_to_request_actor'] = 10




    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager, launch_server, episode_max_time)
        leaderboard_evaluator.run(arguments, customized_data)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator
        # collect signals for estimating objectives


        objectives, loc, object_type, route_completion = estimate_objectives(save_path, default_objectives)




    info = {'episode_max_time':episode_max_time,
    'call_from_dt':call_from_dt,
    'town_name':town_name,
    'scenario':scenario,
    'direction':direction,
    'route_str':route_str,
    'route_type':route_type,
    'ego_car_model':ego_car_model}





    return objectives, loc, object_type, route_completion, info, save_path






class MySampling(Sampling):
    '''
    dimension correspondence

    Define:
    n1=problem.waypoints_num_limit
    n2=problem.num_of_static_max
    n3=problem.num_of_pedestrians_max
    n4=problem.num_of_vehicles_max

    global
    0: friction, real, [0, 1].
    1: weather_index, int, [0, problem.num_of_weathers].
    2: num_of_static, int, [0, n2].
    3: num_of_pedestrians, int, [0, n3].
    4: num_of_vehicles, int, [0, n4].

    ego-car
    5 ~ 4+n1*2: waypoints perturbation [(dx_i, dy_i)] with length n1.
    dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


    static
    5+n1*2 ~ 4+n1*2+n2*4: [(static_type_i, x w.r.t. center, y w.r.t. center, yaw)] with length n2.
    static_type_i, int, [0, problem.num_of_static_types).
    x_i, real, [problem.static_x_min, problem.static_x_max].
    y_i, real, [problem.static_y_min, problem.static_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).

    pedestrians
    5+n1*2+n2*4 ~ 4+n1*2+n2*4+n3*7: [(pedestrian_type_i, x_i, y_i, yaw_i, trigger_distance_i, speed_i, dist_to_travel_i)] with length n3.
    pedestrian_type_i, int, [0, problem.num_of_static_types)
    x_i, real, [problem.pedestrian_x_min, problem.pedestrian_x_max].
    y_i, real, [problem.pedestrian_y_min, problem.pedestrian_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).
    trigger_distance_i, real, [problem.pedestrian_trigger_distance_min, problem.pedestrian_trigger_distance_max].
    speed_i, real, [problem.pedestrian_speed_min, problem.pedestrian_speed_max].
    dist_to_travel_i, real, [problem.pedestrian_dist_to_travel_min, problem.pedestrian_dist_to_travel_max].

    vehicles
    5+n1*2+n2*4+n3*7 ~ 4+n1*2+n2*4+n3*7+n4*(14+n1*2): [(vehicle_type_i, x_i, y_i, yaw_i, initial_speed_i, trigger_distance_i, targeted_speed_i, waypoint_follower_i, targeted_x_i, targeted_y_i, avoid_collision_i, dist_to_travel_i, target_yaw_i, color_i, [(dx_i, dy_i)] with length n1)] with length n4.
    vehicle_type_i, int, [0, problem.num_of_vehicle_types)
    x_i, real, [problem.vehicle_x_min, problem.vehicle_x_max].
    y_i, real, [problem.vehicle_y_min, problem.vehicle_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).
    initial_speed_i, real, [problem.vehicle_initial_speed_min, problem.vehicle_initial_speed_max].
    trigger_distance_i, real, [problem.vehicle_trigger_distance_min, problem.vehicle_trigger_distance_max].
    targeted_speed_i, real, [problem.vehicle_targeted_speed_min, problem.vehicle_targeted_speed_max].
    waypoint_follower_i, boolean, [0, 1]
    targeted_x_i, real, [problem.targeted_x_min, problem.targeted_x_max].
    targeted_y_i, real, [problem.targeted_y_min, problem.targeted_y_max].
    avoid_collision_i, boolean, [0, 1]
    dist_to_travel_i, real, [problem.vehicle_dist_to_travel_min, problem.vehicle_dist_to_travel_max].
    target_yaw_i, real, [problem.yaw_min, problem.yaw_max).
    color_i, int, [0, problem.num_of_vehicle_colors).
    dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


    '''
    def __init__(self, use_unique_bugs, check_unique_coeff, sample_multiplier=500):
        self.use_unique_bugs = use_unique_bugs
        self.check_unique_coeff = check_unique_coeff
        self.sample_multiplier = sample_multiplier
        assert len(self.check_unique_coeff) == 3
    def _do(self, problem, n_samples, **kwargs):
        p, c, th = self.check_unique_coeff
        xl = problem.xl
        xu = problem.xu
        mask = np.array(problem.mask)
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions
        max_sample_times = n_samples * self.sample_multiplier

        algorithm = kwargs['algorithm']

        tmp_off = algorithm.tmp_off

        # print(tmp_off)
        tmp_off_and_X = []
        if len(tmp_off) > 0:
            tmp_off = [off.X for off in tmp_off]
            tmp_off_and_X = tmp_off
        # print(tmp_off)


        def subroutine(X, tmp_off_and_X):
            def sample_one_feature(typ, lower, upper, dist, label):
                assert lower <= upper, label+','+str(lower)+'>'+str(upper)
                if typ == 'int':
                    val = rng.integers(lower, upper+1)
                elif typ == 'real':
                    if dist[0] == 'normal':
                        if dist[1] == None:
                            mean = (lower+upper)/2
                        else:
                            mean = dist[1]
                        val = rng.normal(mean, dist[2], 1)[0]
                    else: # default is uniform
                        val = rand_real(rng, lower, upper)
                    val = np.clip(val, lower, upper)
                return val

            sample_time = 0
            while sample_time < max_sample_times and len(X) < n_samples:
                sample_time += 1
                x = []
                for i, dist in enumerate(parameters_distributions):
                    typ = mask[i]
                    lower = xl[i]
                    upper = xu[i]
                    label = labels[i]
                    val = sample_one_feature(typ, lower, upper, dist, label)
                    x.append(val)


                if not if_violate_constraints(x, problem.customized_constraints, problem.labels)[0]:
                    if not self.use_unique_bugs or (is_distinct(x, tmp_off_and_X, mask, xl, xu, p, c, th) and is_distinct(x, problem.interested_unique_bugs, mask, xl, xu, p, c, th)):
                        x = np.array(x).astype(float)
                        X.append(x)
                        if len(tmp_off) > 0:
                            tmp_off_and_X = tmp_off + X
                        else:
                            tmp_off_and_X = X
                        # if self.use_unique_bugs:
                        #     if disable_unique_x_for_X:
                        #         X = eliminate_duplicates_for_list(mask, xl, xu, p, c, th, X, problem.unique_bugs)
                        #     else:
                        #         X = eliminate_duplicates_for_list(mask, xl, xu, p, c, th, X, problem.unique_bugs, tmp_off=tmp_off)

            return X, sample_time


        X = []
        X, sample_time_1 = subroutine(X, tmp_off_and_X)

        if len(X) > 0:
            X = np.stack(X)
        else:
            X = np.array([])
        print('\n'*3, 'We sampled', X.shape[0], '/', n_samples, 'samples', 'by sampling', sample_time_1, 'times' '\n'*3)

        return X

class MySamplingVectorized(Sampling):

    def __init__(self, use_unique_bugs, check_unique_coeff, sample_multiplier=500):
        self.use_unique_bugs = use_unique_bugs
        self.check_unique_coeff = check_unique_coeff
        self.sample_multiplier = sample_multiplier
        assert len(self.check_unique_coeff) == 3
    def _do(self, problem, n_samples, **kwargs):
        p, c, th = self.check_unique_coeff
        xl = problem.xl
        xu = problem.xu
        mask = np.array(problem.mask)
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions

        if self.sample_multiplier >= 50:
            max_sample_times = self.sample_multiplier // 50
            n_samples_sampling = n_samples * 50
        else:
            max_sample_times = self.sample_multiplier
            n_samples_sampling = n_samples

        algorithm = kwargs['algorithm']

        tmp_off = algorithm.tmp_off

        # print(tmp_off)
        tmp_off_and_X = []
        if len(tmp_off) > 0:
            tmp_off = [off.X for off in tmp_off]
            tmp_off_and_X = tmp_off
        # print(tmp_off)


        def subroutine(X, tmp_off_and_X):
            def sample_one_feature(typ, lower, upper, dist, label, size=1):
                assert lower <= upper, label+','+str(lower)+'>'+str(upper)
                if typ == 'int':
                    val = rng.integers(lower, upper+1, size=size)
                elif typ == 'real':
                    if dist[0] == 'normal':
                        if dist[1] == None:
                            mean = (lower+upper)/2
                        else:
                            mean = dist[1]
                        val = rng.normal(mean, dist[2], size=size)
                    else: # default is uniform
                        val = rng.random(size=size) * (upper - lower) + lower
                    val = np.clip(val, lower, upper)
                return val

            # TBD: temporary
            sample_time = 0
            while sample_time < max_sample_times and len(X) < n_samples:
                print('sample_time / max_sample_times', sample_time, '/', max_sample_times, 'len(X)', len(X))
                sample_time += 1
                cur_X = []
                for i, dist in enumerate(parameters_distributions):
                    typ = mask[i]
                    lower = xl[i]
                    upper = xu[i]
                    label = labels[i]
                    val = sample_one_feature(typ, lower, upper, dist, label, size=n_samples_sampling)
                    cur_X.append(val)
                cur_X = np.swapaxes(np.stack(cur_X),0,1)


                remaining_inds = if_violate_constraints_vectorized(cur_X, problem.customized_constraints, problem.labels, verbose=False)
                if len(remaining_inds) == 0:
                    continue

                cur_X = cur_X[remaining_inds]

                if not self.use_unique_bugs:
                    X.extend(cur_X)
                    if len(X) > n_samples:
                        X = X[:n_samples]
                else:
                    if len(tmp_off_and_X) > 0 and len(problem.interested_unique_bugs) > 0:
                        prev_X = np.concatenate([problem.interested_unique_bugs, tmp_off_and_X])
                    elif len(tmp_off_and_X) > 0:
                        prev_X = tmp_off_and_X
                    else:
                        prev_X = problem.interested_unique_bugs
                    # print('prev_X.shape', prev_X.shape)
                    remaining_inds = is_distinct_vectorized(cur_X, prev_X, mask, xl, xu, p, c, th, verbose=False)

                    if len(remaining_inds) == 0:
                        continue
                    else:
                        cur_X = cur_X[remaining_inds]
                        X.extend(cur_X)
                        if len(X) > n_samples:
                            X = X[:n_samples]
                        if len(tmp_off) > 0:
                            tmp_off_and_X = tmp_off + X
                        else:
                            tmp_off_and_X = X


            return X, sample_time


        X = []
        X, sample_time_1 = subroutine(X, tmp_off_and_X)

        if len(X) > 0:
            X = np.stack(X)
        else:
            X = np.array([])
        print('\n'*3, 'We sampled', X.shape[0], '/', n_samples, 'samples', 'by sampling', sample_time_1, 'times' '\n'*3)

        return X



def do_emcmc(parents, off, n_gen, objective_weights):
    base_val = np.sum(np.array(default_objectives[:len(objective_weights)])*np.array(objective_weights))
    filtered_off = []
    F_list = []
    for i in off:
        for p in parents:
            print(i.F, p.F)

            i_val = np.sum(np.array(i.F) * np.array(objective_weights))
            p_val = np.sum(np.array(p.F) * np.array(objective_weights))

            print('1', base_val, i_val, p_val)
            i_val = np.abs(base_val-i_val)
            p_val = np.abs(base_val-p_val)
            prob = np.min([i_val / p_val, 1])
            print('2', base_val, i_val, p_val, prob)

            if np.random.uniform() < prob:
                filtered_off.append(i.X)
                F_list.append(i.F)

    pop = Population(len(filtered_off), individual=Individual())
    pop.set("X", filtered_off, "F", F_list, "n_gen", n_gen, "CV", [0 for _ in range(len(filtered_off))], "feasible", [[True] for _ in range(len(filtered_off))])

    return Population.merge(parents, off)


class MyMating(Mating):
    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 use_unique_bugs,
                 emcmc,
                 **kwargs):

        super().__init__(selection, crossover, mutation, **kwargs)
        self.use_unique_bugs = use_unique_bugs
        self.mating_max_iterations = mating_max_iterations
        self.emcmc = emcmc

    def do(self, problem, pop, n_offsprings, **kwargs):

        # the population object to be used
        off = pop.new()
        parents = pop.new()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:
            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off, _parents = self._do(problem, pop, n_remaining, **kwargs)


            # repair the individuals if necessary - disabled if repair is NoRepair
            _off_first = self.repair.do(problem, _off, **kwargs)

            # Previous
            _off = []
            for x in _off_first:
                if not if_violate_constraints(x.X, problem.customized_constraints, problem.labels)[0]:
                    _off.append(x.X)

            _off = pop.new("X", _off)

            # Previous
            # eliminate the duplicates - disabled if it is NoRepair
            if self.use_unique_bugs and len(_off) > 0:
                _off, no_duplicate, _ = self.eliminate_duplicates.do(_off, problem.unique_bugs, off, return_indices=True, to_itself=True)
                _parents = _parents[no_duplicate]
                assert len(_parents)==len(_off)




            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                # IMPORTANT: Interestingly, this makes a difference in performance
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]
                _parents = _parents[:n_remaining]


            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            parents = Population.merge(parents, _parents)
            n_infills += 1

            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills > self.mating_max_iterations:
                break

        # assert len(parents)==len(off)
        print('Mating finds', len(off), 'offsprings after doing', n_infills-1, '/', self.mating_max_iterations, 'mating iterations')
        return off, parents



    # only to get parents
    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:
            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)
            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)
            parents_obj = pop[parents].reshape([-1, 1]).squeeze()
        else:
            parents_obj = parents


        # do the crossover using the parents index and the population - additional data provided if necessary
        _off = self.crossover.do(problem, pop, parents, **kwargs)
        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)

        return _off, parents_obj

class MyMatingVectorized(Mating):
    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 use_unique_bugs,
                 emcmc,
                 **kwargs):

        super().__init__(selection, crossover, mutation, **kwargs)
        self.use_unique_bugs = use_unique_bugs
        self.mating_max_iterations = mating_max_iterations
        self.emcmc = emcmc


    def do(self, problem, pop, n_offsprings, **kwargs):

        if self.mating_max_iterations >= 5:
            mating_max_iterations = self.mating_max_iterations // 5
            n_offsprings_sampling = n_offsprings * 5
        else:
            mating_max_iterations = self.mating_max_iterations
            n_offsprings_sampling = n_offsprings

        # the population object to be used
        off = pop.new()
        parents = pop.new()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:
            n_infills += 1
            print('n_infills / mating_max_iterations', n_infills, '/', mating_max_iterations, 'len(off)', len(off))
            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills >= mating_max_iterations:
                break

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off, _parents = self._do(problem, pop, n_offsprings_sampling, **kwargs)


            # repair the individuals if necessary - disabled if repair is NoRepair
            _off_first = self.repair.do(problem, _off, **kwargs)


            # Vectorized
            _off_X = np.array([x.X for x in _off_first])
            remaining_inds = if_violate_constraints_vectorized(_off_X, problem.customized_constraints, problem.labels, verbose=False)
            _off_X = _off_X[remaining_inds]

            _off = _off_first[remaining_inds]
            _parents = _parents[remaining_inds]

            # Vectorized
            if self.use_unique_bugs:
                if len(_off_X) == 0:
                    continue

                if len(off) > 0:
                    prev_X = np.concatenate([np.array([x.X for x in off]), problem.interested_unique_bugs])
                else:
                    prev_X = problem.interested_unique_bugs

                remaining_inds = is_distinct_vectorized(_off_X, prev_X, problem.mask, problem.xl, problem.xu, problem.p, problem.c, problem.th, verbose=False)

                if len(remaining_inds) == 0:
                    continue

                _off = _off[remaining_inds]
                _parents = _parents[remaining_inds]
                assert len(_parents)==len(_off)



            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                # IMPORTANT: Interestingly, this makes a difference in performance
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]
                _parents = _parents[:n_remaining]


            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            parents = Population.merge(parents, _parents)




        # assert len(parents)==len(off)
        print('Mating finds', len(off), 'offsprings after doing', n_infills, '/', mating_max_iterations, 'mating iterations')
        return off, parents



    # only to get parents
    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:
            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)
            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)
            parents_obj = pop[parents].reshape([-1, 1]).squeeze()
        else:
            parents_obj = parents


        # do the crossover using the parents index and the population - additional data provided if necessary
        _off = self.crossover.do(problem, pop, parents, **kwargs)
        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)

        return _off, parents_obj


class NSGA2_DT(NSGA2):
    def __init__(self, dt=False, X=None, F=None, emcmc=False, plain_sampling=None, algorithm_name='nsga2-un', **kwargs):
        self.dt = dt
        self.X = X
        self.F = F
        self.emcmc = emcmc
        self.algorithm_name = algorithm_name
        self.sampling = kwargs['sampling']

        super().__init__(**kwargs)

        self.plain_initialization = Initialization(plain_sampling, individual=Individual(), repair=self.repair, eliminate_duplicates= NoDuplicateElimination())


        # heuristic: we keep up about 2 times of each generation's population
        self.survival_size = self.pop_size * survival_multiplier


        self.all_pop_run_X = []
        self.initial_fit_th = initial_fit_th
        self.min_bug_num_to_fit_dnn = min_bug_num_to_fit_dnn
        self.rank_mode = rank_mode
        self.ranking_model = ranking_model
        self.use_unique_bugs = use_unique_bugs

        # hack: defined separately w.r.t. MyMating
        self.mating_max_iterations = 1

        self.tmp_off = []
        self.tmp_off_type_1_len = 0
        # self.tmp_off_type_1and2_len = 0

        self.pgd_eps = pgd_eps
        self.adv_conf_th = adv_conf_th
        self.attack_stop_conf = attack_stop_conf

        self.uncertainty = uncertainty
        self.model_type = model_type
        self.use_single_nn = use_single_nn

        self.high_conf_configs_stack = []
        self.high_conf_configs_ori_stack = []
        self.explore_iter_num = explore_iter_num
        self.exploit_iter_num = exploit_iter_num
        self.high_conf_num = high_conf_num
        self.low_conf_num = low_conf_num

        self.warm_up_path = warm_up_path
        self.warm_up_len = warm_up_len
        self.use_alternate_nn = use_alternate_nn
        self.diversity_mode = diversity_mode

        self.regression_nn_use_running_data = regression_nn_use_running_data

        self.adv_exploitation_only = adv_exploitation_only

        self.uncertainty_exploration = uncertainty_exploration

        self.only_run_unique_cases = only_run_unique_cases

    def set_off(self):
        self.tmp_off = []
        if self.algorithm_name == 'random':
            self.tmp_off = self.plain_initialization.do(self.problem, self.n_offsprings, algorithm=self)
        else:
            if self.algorithm_name == 'random-un':
                self.tmp_off, parents = [], []

            else:
                print('len(self.pop)', len(self.pop))
                # do the mating using the current population
                if len(self.pop) > 0:
                    self.tmp_off, parents = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

            print('\n'*3, 'after mating len 0', len(self.tmp_off), 'self.n_offsprings', self.n_offsprings, '\n'*3)


            if len(self.tmp_off) < self.n_offsprings:
                remaining_num = self.n_offsprings - len(self.tmp_off)
                remaining_off = self.initialization.do(self.problem, remaining_num, algorithm=self)
                remaining_parrents = remaining_off
                if len(self.tmp_off) == 0:
                    self.tmp_off = remaining_off
                    parents = remaining_parrents
                else:
                    self.tmp_off = Population.merge(self.tmp_off, remaining_off)
                    parents = Population.merge(parents, remaining_parrents)

                print('\n'*3, 'unique after random generation len 1', len(self.tmp_off), '\n'*3)

            self.tmp_off_type_1_len = len(self.tmp_off)

            if len(self.tmp_off) < self.n_offsprings:
                remaining_num = self.n_offsprings - len(self.tmp_off)
                remaining_off = self.plain_initialization.do(self.problem, remaining_num, algorithm=self)
                remaining_parrents = remaining_off

                self.tmp_off = Population.merge(self.tmp_off, remaining_off)
                parents = Population.merge(parents, remaining_parrents)

                print('\n'*3, 'random generation len 2', len(self.tmp_off), '\n'*3)




        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.tmp_off) == 0 or (not self.problem.call_from_dt and finish_after_has_run and self.problem.has_run >= has_run_num):
            self.termination.force_termination = True
            print("Mating cannot generate new springs, terminate earlier.")
            print('self.tmp_off', len(self.tmp_off), self.tmp_off)
            return

        # if not the desired number of offspring could be created
        elif len(self.tmp_off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")


        if len(self.all_pop_run_X) > 0:
            print('self.all_pop_run_X.shape', self.all_pop_run_X.shape)

        # additional step to rank and select self.off after gathering initial population
        if self.rank_mode != 'none':
            # print(self.rank_mode in ['nn', 'adv_nn'])
            # print(len(self.problem.objectives_list), self.initial_fit_th)
            # print(np.sum(determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)), self.min_bug_num_to_fit_dnn)
            if (self.rank_mode in ['nn', 'adv_nn', 'alternate_nn'] and len(self.problem.objectives_list) >= self.initial_fit_th and  np.sum(determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)) >= self.min_bug_num_to_fit_dnn) or (self.rank_mode in ['regression_nn'] and len(self.problem.objectives_list) >= self.pop_size):
                if self.rank_mode in ['regression_nn']:
                    # only consider collision case for now
                    from customized_utils import pretrain_regression_nets

                    if self.regression_nn_use_running_data:
                        initial_X = self.all_pop_run_X
                        initial_objectives_list = self.problem.objectives_list
                        cutoff = len(initial_X)
                        cutoff_end = cutoff
                    else:
                        subfolders = get_sorted_subfolders(self.warm_up_path)
                        initial_X, _, initial_objectives_list, _, _ = load_data(subfolders)

                        cutoff = self.warm_up_len
                        cutoff_end = self.warm_up_len + 100

                        if cutoff == 0:
                            cutoff = len(initial_X)
                        if cutoff_end > len(initial_X):
                            cutoff_end = len(initial_X)


                    # print('self.problem.labels', self.problem.labels)
                    clfs, confs, chosen_weights, standardize_prev = pretrain_regression_nets(initial_X, initial_objectives_list, self.problem.objective_weights, self.problem.xl, self.problem.xu, self.problem.labels, self.problem.customized_constraints, cutoff, cutoff_end)
                else:
                    standardize_prev = None

                X_train_ori = self.all_pop_run_X
                print('X_train_ori.shape[0]', X_train_ori.shape[0])
                X_test_ori = self.tmp_off.get("X")
                # print(np.array(X_train_ori).shape, np.array(X_test_ori).shape)

                initial_X = np.concatenate([X_train_ori, X_test_ori])
                cutoff = X_train_ori.shape[0]
                cutoff_end = initial_X.shape[0]
                partial = True
                # print('initial_X.shape', np.array(initial_X).shape, cutoff, cutoff_end)
                # print('len(self.problem.labels)', len(self.problem.labels))
                X_train, X_test, xl, xu, labels_used, standardize, one_hot_fields_len, param_for_recover_and_decode = process_X(initial_X, self.problem.labels, self.problem.xl, self.problem.xu, cutoff, cutoff_end, partial, len(self.problem.interested_unique_bugs), standardize_prev=standardize_prev)
                # print('X_train.shape[0]', X_train.shape[0])
                # print('labels_used', labels_used)
                # print('process_X X_train.shape, X_test.shape', X_train.shape, X_test.shape)
                (X_removed, kept_fields, removed_fields, enc, inds_to_encode, inds_non_encode, encoded_fields, _, _, unique_bugs_len) = param_for_recover_and_decode

                print('process_X finished')
                if self.rank_mode in ['regression_nn']:
                    # only consider collision case for now

                    # print('X_test.shape', X_test.shape)
                    weight_inds = choose_weight_inds(self.problem.objective_weights)
                    obj_preds = []
                    for clf in clfs:
                        obj_preds.append(clf.predict(X_test))

                    tmp_objectives = np.concatenate(obj_preds, axis=1)
                    # print('tmp_objectives', tmp_objectives)
                    # when using unique bugs give preference to unique inputs

                    if self.use_unique_bugs:
                        tmp_objectives[:self.tmp_off_type_1_len] -= 100*chosen_weights

                    # print(len(tmp_objectives), self.tmp_off_type_1_len)
                    # print('tmp_objectives after use_unique_bugs', tmp_objectives)

                    tmp_objectives_minus = tmp_objectives - confs
                    tmp_objectives_plus = tmp_objectives + confs


                    tmp_pop_minus = Population(X_train.shape[0]+X_test.shape[0], individual=Individual())
                    # print(X_train.shape)
                    # print(X_test.shape)
                    tmp_X_minus = np.concatenate([X_train, X_test])

                    # print(np.array(self.problem.objectives_list)[:, :3])
                    # print(tmp_objectives)
                    # print(np.array(default_objective_weights[:3]))
                    tmp_objectives_minus = np.concatenate([np.array(self.problem.objectives_list)[:, weight_inds], tmp_objectives_minus]) * np.array(default_objective_weights[weight_inds])

                    tmp_pop_minus.set("X", tmp_X_minus)
                    tmp_pop_minus.set("F", tmp_objectives_minus)

                    print('len(tmp_objectives_minus)', len(tmp_objectives_minus))
                    inds_minus_top = np.array(self.survival.do(self.problem, tmp_pop_minus, self.pop_size, return_indices=True))
                    print('inds_minus_top', inds_minus_top, 'len(X_train)', len(X_train), np.sum(inds_minus_top<len(X_train)))
                    # print('inds_minus_top', inds_minus_top)
                    num_of_top_already_run = np.sum(inds_minus_top<len(X_train))
                    num_to_run = self.pop_size - num_of_top_already_run

                    if num_to_run > 0:
                        tmp_pop_plus = Population(X_test.shape[0], individual=Individual())

                        tmp_X_plus = X_test
                        tmp_objectives_plus = tmp_objectives_plus * np.array(default_objective_weights[weight_inds])

                        tmp_pop_plus.set("X", tmp_X_plus)
                        tmp_pop_plus.set("F", tmp_objectives_plus)

                        print('tmp_objectives_plus', tmp_objectives_plus)
                        inds_plus_top = np.array(self.survival.do(self.problem, tmp_pop_plus, num_to_run, return_indices=True))

                        print('inds_plus_top', inds_plus_top)
                        self.off = self.tmp_off[inds_plus_top]
                    else:
                        print('no more offsprings to run (regression nn)')
                        self.off = Population(0, individual=Individual())
                else:

                    if self.uncertainty:
                        from pgd_attack import VanillaDataset
                        from acquisition import map_acquisition
                        # [None, 'BUGCONF', 'Random', 'BALD', 'BatchBALD']
                        print('uncertainty', self.uncertainty)
                        uncertainty_key, uncertainty_conf = self.uncertainty.split('_')

                        acquisition_strategy = map_acquisition(uncertainty_key)
                        acquirer = acquisition_strategy(self.pop_size)



                        if uncertainty_conf == 'conf':
                            uncertainty_conf = True
                        else:
                            uncertainty_conf = False

                        pool_data = VanillaDataset(X_test, np.zeros(X_test.shape[0]), to_tensor=True)
                        pool_data = torch.utils.data.Subset(pool_data, np.arange(len(pool_data)))

                        y_train = determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)
                        clf = train_net(X_train, y_train, [], [], batch_train=60, model_type=self.model_type)

                        if self.use_unique_bugs:
                            unique_len = self.tmp_off_type_1_len
                        else:
                            unique_len = 0
                        inds = acquirer.select_batch(clf, pool_data, unique_len=unique_len, uncertainty_conf=uncertainty_conf)
                        print('chosen indices', inds)
                    else:
                        one_clf = True
                        adv_conf_th = self.adv_conf_th
                        attack_stop_conf = self.attack_stop_conf

                        print('self.use_single_nn', self.use_single_nn)
                        if self.use_single_nn:
                            y_train = determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)
                            print('self.ranking_model', self.ranking_model)
                            if self.ranking_model == 'nn_sklearn':
                                clf = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=10000)
                                clf.fit(X_train, y_train)
                            elif self.ranking_model == 'nn_pytorch':
                                print(X_train.shape, y_train.shape)
                                clf = train_net(X_train, y_train, [], [], batch_train=200)
                            elif self.ranking_model == 'adaboost':
                                from sklearn.ensemble import AdaBoostClassifier
                                clf = AdaBoostClassifier()
                                clf = clf.fit(X_train, y_train)
                            elif self.ranking_model == 'regression':
                                # only support collision for now
                                y_train = np.array([obj[1] for obj in self.problem.objectives_list])
                                # print('y_train', y_train)
                                clf = train_net(X_train, y_train, [], [], batch_train=64, model_type=ranking_model)
                            else:
                                raise ValueError('invalid ranking model', ranking_model)
                            print('X_train', X_train.shape)
                            print('clf.predict_proba(X_train)', clf.predict_proba(X_train).shape)
                            if ranking_model == 'adaboost':
                                prob_train = clf.predict_proba(X_train)[:, 0].squeeze()
                            else:
                                prob_train = clf.predict_proba(X_train)[:, 1].squeeze()
                            cur_y = y_train

                            if self.adv_conf_th < 0 and self.rank_mode in ['adv_nn']:
                                print(sorted(prob_train, reverse=True))
                                print('cur_y', cur_y)
                                print('np.abs(self.adv_conf_th)', np.abs(self.adv_conf_th))
                                print(int(np.sum(cur_y)//np.abs(self.adv_conf_th)))
                                adv_conf_th = sorted(prob_train, reverse=True)[int(np.sum(cur_y)//np.abs(self.adv_conf_th))]
                                attack_stop_conf = np.max([self.attack_stop_conf, adv_conf_th])
                            if self.adv_conf_th > attack_stop_conf:
                                self.adv_conf_th = attack_stop_conf

                        else:
                            from customized_utils import get_all_y

                            y_list = get_all_y(self.problem.objectives_list, self.problem.objective_weights)
                            clf_list = []
                            bug_type_nn_activated = []
                            print('self.problem.objectives_list', self.problem.objectives_list)
                            print('self.problem.objective_weights', self.problem.objective_weights)
                            print('y_list', y_list)
                            for i, y_train in enumerate(y_list):
                                print('np.sum(y_train)', np.sum(y_train), 'self.min_bug_num_to_fit_dnn', self.min_bug_num_to_fit_dnn)
                                if np.sum(y_train) >= self.min_bug_num_to_fit_dnn:
                                    if self.ranking_model == 'nn_sklearn':
                                        clf = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=10000)
                                        clf.fit(X_train, y_train)
                                    elif ranking_model == 'nn_pytorch':
                                        clf = train_net(X_train, y_train, [], [], batch_train=200)
                                    else:
                                        raise
                                    clf_list.append(clf)
                                    bug_type_nn_activated.append(i)

                            if len(clf_list) > 1:
                                if self.adv_conf_th < 0:
                                    adv_conf_th = []
                                    attack_stop_conf = []

                                from scipy import stats
                                one_clf = False
                                scores_on_all_nn = np.zeros([X_test.shape[0], len(clf_list)])
                                for j, clf in enumerate(clf_list):
                                    prob_test = clf.predict_proba(X_test)[:, 1].squeeze()
                                    prob_train = clf.predict_proba(X_train)[:, 1].squeeze()
                                    bug_type = bug_type_nn_activated[j]
                                    cur_y = y_list[bug_type]
                                    print('np.sum(cur_y)', np.sum(cur_y), 'np.abs(self.adv_conf_th)', np.abs(self.adv_conf_th), 'np.sum(cur_y)//np.abs(self.adv_conf_th)', np.sum(cur_y)//np.abs(self.adv_conf_th))

                                    th_conf = sorted(prob_train, reverse=True)[int(np.sum(cur_y)//np.abs(self.adv_conf_th))]
                                    adv_conf_th.append(th_conf)
                                    attack_stop_conf.append(np.max([th_conf, self.attack_stop_conf]))
                                    print('adv_conf_th', adv_conf_th)
                                    print('attack_stop_conf', attack_stop_conf)

                                    y_j_bug_perc = np.mean(cur_y)*100
                                    scores_on_all_nn[:, j] = [(stats.percentileofscore(prob_train, prob_test_i) - (100 - y_j_bug_perc)) / y_j_bug_perc for prob_test_i in prob_test]

                                    print('-'*50)
                                    print(j)
                                    print('y_j_bug_perc', y_j_bug_perc)
                                    print('prob_train', prob_train)
                                    print('prob_test', prob_test)
                                    print('scores_on_all_nn[:, j]', scores_on_all_nn[:, j])
                                    print('-'*50)

                                print(scores_on_all_nn)
                                associated_clf_id = np.argmax(scores_on_all_nn, axis=1)

                                print(associated_clf_id)

                                # TBD: change the name to plural for less confusion
                                clf = clf_list
                            else:
                                clf = clf_list[0]




                        print('\n', 'adv_conf_th', adv_conf_th, '\n')
                        if one_clf == True:
                            pred = clf.predict_proba(X_test)
                            if len(pred.shape) == 1:
                                pred = np.expand_dims(pred, axis=0)
                            scores = pred[:, 1]
                        else:
                            scores = np.max(scores_on_all_nn, axis=1)
                        print('initial scores', scores)
                        # when using unique bugs give preference to unique inputs

                        if self.rank_mode == 'adv_nn':
                            X_test_pgd_ori = None
                            X_test_pgd = None

                        if not self.use_alternate_nn:
                            if self.use_unique_bugs:
                                print('self.tmp_off_type_1_len', self.tmp_off_type_1_len)
                                scores[:self.tmp_off_type_1_len] += np.max(scores)
                                # scores[:self.tmp_off_type_1and2_len] += 100
                            scores *= -1

                            inds = np.argsort(scores)[:self.pop_size]
                            print('scores', scores)
                            print('sorted(scores)', sorted(scores))
                            print('chosen indices', inds)

                        else:
                            # if self.warm_up_path:
                            #     cur_gen = (len(self.problem.objectives_list) - self.warm_up_len) // self.pop_size
                            # else:
                            #     cur_gen = (len(self.problem.objectives_list) - self.initial_fit_th) // self.pop_size
                            cur_gen = self.n_gen
                            cycle_num = self.explore_iter_num+self.exploit_iter_num
                            print('cur_gen', cur_gen, 'cycle_num', cycle_num)
                            if (cur_gen-1) % cycle_num < self.explore_iter_num:
                                current_stage = 'exploration'
                                print('\n', 'exploration', '\n')
                                scores *= -1

                                if self.uncertainty_exploration == 'confidence':
                                    inds_used = np.argsort(scores)
                                else: # random
                                    inds_used = np.arange(len(scores))

                                high_inds = inds_used[:self.high_conf_num]
                                mid_inds = inds_used[self.high_conf_num:len(scores)-self.low_conf_num]

                                print(len(mid_inds), self.high_conf_num, len(scores)-self.low_conf_num)

                                if diversity_mode == 'nn_rep':
                                    d_list = calculate_rep_d(clf, X_train, X_test[mid_inds])
                                    if self.use_unique_bugs:
                                        unique_inds = is_distinct_vectorized(X_test_ori[mid_inds], self.problem.interested_unique_bugs, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th, verbose=False)

                                        print('len(unique_inds)', len(unique_inds))
                                        d_list[unique_inds] += np.max(d_list)
                                    print('X_train.shape[0]', X_train.shape[0])
                                    mid_inds_top_inds = select_batch_max_d_greedy(d_list, X_train.shape[0], self.pop_size)
                                    print('mid_inds_top_inds', mid_inds_top_inds)
                                    inds = mid_inds[mid_inds_top_inds]
                                    print('inds', inds)
                                else:
                                    inds = np.random.choice(mid_inds, self.pop_size, replace=False)

                                if self.rank_mode == 'adv_nn' and not self.adv_exploitation_only:
                                    X_test_pgd_ori = X_test_ori[inds]
                                    X_test_pgd = X_test[inds]
                                else:
                                    self.off = self.tmp_off[inds]

                                print('sorted(scores)', sorted(scores))
                                scores_rank = rankdata(scores)
                                print('chosen indices (rank)', scores_rank[inds])
                                print('chosen indices', inds)

                                if diversity_mode == 'nn_rep':
                                    print('all min distance', np.sort(d_list, axis=1)[:, 1])
                                    print('chosen max min distance', np.sort(d_list[mid_inds_top_inds], axis=1)[:, 1])

                                self.high_conf_configs_stack.append(X_test[high_inds])
                                self.high_conf_configs_ori_stack.append(X_test_ori[high_inds])
                            else:
                                current_stage = 'exploitation'
                                print('\n', 'exploitation', '\n')
                                if len(self.high_conf_configs_stack) > 0:
                                    high_conf_configs_stack_np = np.concatenate(self.high_conf_configs_stack)
                                    high_conf_configs_ori_stack_np = np.concatenate(self.high_conf_configs_ori_stack)

                                    print('len(high_conf_configs_ori_stack_np) before filtering', len(high_conf_configs_ori_stack_np))
                                    # high_conf_configs_ori_stack_np, distinct_inds = get_distinct_data_points(high_conf_configs_ori_stack_np, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th)
                                    # high_conf_configs_ori_stack_np = np.array(high_conf_configs_ori_stack_np)
                                    # high_conf_configs_stack_np = high_conf_configs_stack_np[distinct_inds]
                                    # print('len(high_conf_configs_ori_stack_np) after filtering', len(high_conf_configs_ori_stack_np))

                                    scores = clf.predict_proba(high_conf_configs_stack_np)[:, 1]
                                    if self.use_unique_bugs:
                                        unique_inds = is_distinct_vectorized(high_conf_configs_ori_stack_np, self.problem.interested_unique_bugs, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th, verbose=False)

                                        print('len(unique_inds)', len(unique_inds))
                                        if len(unique_inds) > 0:
                                            scores[unique_inds] += np.max(scores)

                                    scores *= -1
                                    inds = np.argsort(scores)[:self.pop_size]

                                    print('sorted(scores)', sorted(scores))
                                    scores_rank = rankdata(scores)
                                    print('chosen indices (rank)', scores_rank[inds])

                                else:
                                    print('\n'*2, 'len(self.high_conf_configs_stack)', len(self.high_conf_configs_stack), '\n'*2)
                                    inds = []


                                print('chosen indices', inds)

                                if self.rank_mode == 'adv_nn':
                                    X_test_pgd_ori = high_conf_configs_ori_stack_np[inds]
                                    X_test_pgd = high_conf_configs_stack_np[inds]
                                else:
                                    X_test_pgd_ori = high_conf_configs_ori_stack_np[inds]
                                    pop = Population(X_test_pgd_ori.shape[0], individual=Individual())
                                    pop.set("X", X_test_pgd_ori)
                                    pop.set("F", [None for _ in range(X_test_pgd_ori.shape[0])])
                                    self.off = pop

                                self.high_conf_configs_stack = []
                                self.high_conf_configs_ori_stack = []




                    if self.use_alternate_nn and self.rank_mode != 'adv_nn' or (self.use_alternate_nn and self.rank_mode == 'adv_nn' and self.adv_exploitation_only and current_stage == 'exploration'):
                        pass
                    elif self.rank_mode == 'nn':
                        self.off = self.tmp_off[inds]
                    elif self.rank_mode == 'adv_nn':
                        if not self.use_alternate_nn:
                            X_test_pgd_ori = X_test_ori[inds]
                            X_test_pgd = X_test[inds]

                        if one_clf == True:
                            associated_clf_id = []
                        else:
                            associated_clf_id = associated_clf_id[inds]


                        # conduct pgd with constraints differently for different types of inputs
                        if self.use_unique_bugs:
                            unique_coeff = (self.problem.p, self.problem.c, self.problem.th)
                            mask = self.problem.mask

                            y_zeros = np.zeros(X_test_pgd.shape[0])
                            X_test_adv, new_bug_pred_prob_list, initial_bug_pred_prob_list = pgd_attack(clf, X_test_pgd, y_zeros, xl, xu, encoded_fields, labels_used, self.problem.customized_constraints, standardize, prev_X=self.problem.interested_unique_bugs, base_ind=0, unique_coeff=unique_coeff, mask=mask, param_for_recover_and_decode=param_for_recover_and_decode, eps=self.pgd_eps, adv_conf_th=adv_conf_th, attack_stop_conf=attack_stop_conf, associated_clf_id=associated_clf_id, X_test_pgd_ori=X_test_pgd_ori, consider_uniqueness=True)


                        else:
                            y_zeros = np.zeros(X_test_pgd.shape[0])
                            X_test_adv, new_bug_pred_prob_list, initial_bug_pred_prob_list = pgd_attack(clf, X_test_pgd, y_zeros, xl, xu, encoded_fields, labels_used, self.problem.customized_constraints, standardize, eps=self.pgd_eps, adv_conf_th=adv_conf_th, attack_stop_conf=attack_stop_conf, associated_clf_id=associated_clf_id, X_test_pgd_ori=X_test_pgd_ori)


                        X_test_adv_processed = inverse_process_X(X_test_adv, standardize, one_hot_fields_len, partial, X_removed, kept_fields, removed_fields, enc, inds_to_encode, inds_non_encode, encoded_fields)


                        # X_test_adv_processed = customized_inverse_standardize(X_test_adv, standardize, one_hot_fields_len, partial)
                        # X_test_adv_processed = recover_fields_not_changing(X_test_adv_processed, X_removed, kept_fields, removed_fields)
                        # X_test_adv_processed = decode_fields(X_test_adv_processed, enc, inds_to_encode, inds_non_encode, encoded_fields, adv=True)


                        use_combined = False
                        if use_combined:
                            X_combined = np.concatenate([X_test_pgd_ori, X_test_adv_processed], axis=0)
                            X_combined_processed = np.concatenate([X_test_pgd, X_test_adv], axis=0)

                            print('before considering constraints', X_combined.shape[0])
                            chosen_inds = []
                            for i, x in enumerate(X_combined):
                                if not if_violate_constraints(x, self.problem.customized_constraints, self.problem.labels)[0]:
                                    chosen_inds.append(i)
                            chosen_inds = np.array(chosen_inds)

                            X_combined = X_combined[chosen_inds]
                            X_combined_processed = X_combined_processed[chosen_inds]
                            print('after considering constraints', X_combined.shape[0])

                            scores = -1*clf.predict_proba(X_combined_processed)[:, 1]
                            inds = np.argsort(scores)[:self.pop_size]
                            print('scores', scores)
                            print('chosen indices', inds)
                            X_off = X_combined[inds]

                        else:
                            X_off = X_test_adv_processed


                        pop = Population(X_off.shape[0], individual=Individual())
                        pop.set("X", X_off)
                        pop.set("F", [None for _ in range(X_off.shape[0])])
                        self.off = pop

                        adv_preds_path = os.path.join(self.problem.bug_folder, 'adv_preds.npz')
                        new_d = {}
                        if os.path.exists(adv_preds_path):
                            d = np.load(adv_preds_path)
                            new_d['new_bug_pred_prob_list'] = np.concatenate([d['new_bug_pred_prob_list'], new_bug_pred_prob_list])
                            new_d['initial_bug_pred_prob_list'] = np.concatenate([d['initial_bug_pred_prob_list'], initial_bug_pred_prob_list])
                        else:
                            new_d = {'new_bug_pred_prob_list': new_bug_pred_prob_list, 'initial_bug_pred_prob_list': initial_bug_pred_prob_list}
                        np.savez(adv_preds_path, **new_d)
            else:
                self.off = self.tmp_off[:self.pop_size]
        else:
            self.off = self.tmp_off[:self.pop_size]

        if self.only_run_unique_cases:
            X_off = [off_i.X for off_i in self.off]
            remaining_inds = is_distinct_vectorized(X_off, self.problem.interested_unique_bugs, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th, verbose=False)
            self.off = self.off[remaining_inds]

        self.off.set("n_gen", self.n_gen)

        print('\n'*2, 'self.n_gen', self.n_gen, '\n'*2)

        if len(self.all_pop_run_X) == 0:
            self.all_pop_run_X = self.off.get("X")
        else:
            if len(self.off.get("X")) > 0:
                self.all_pop_run_X = np.concatenate([self.all_pop_run_X, self.off.get("X")])

    # mainly used to modify survival
    def _next(self):

        # set self.off
        self.set_off()
        # evaluate the offspring
        # print('start evaluator', 'pop', self.off)
        if len(self.off) > 0:
            self.evaluator.eval(self.problem, self.off, algorithm=self)
        # print('end evaluator')


        if self.algorithm_name == 'random':
            self.pop = self.off
        elif self.emcmc:
            new_pop = do_emcmc(parents, self.off, self.n_gen, self.problem.objective_weights)

            self.pop = Population.merge(self.pop, new_pop)

            if self.survival:
                self.pop = self.survival.do(self.problem, self.pop, self.survival_size, algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)
        else:
            # merge the offsprings with the current population
            self.pop = Population.merge(self.pop, self.off)

            # the do survival selection
            if self.survival:
                print('\n'*3)
                print('len(self.pop) before', len(self.pop))
                print('survival')
                self.pop = self.survival.do(self.problem, self.pop, self.survival_size, algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)
                print('len(self.pop) after', len(self.pop))
                print(self.pop_size, self.survival_size)
                print('\n'*3)



    def _initialize(self):
        if self.warm_up_path and ((self.dt and not self.problem.cumulative_info) or (not self.dt)):
            subfolders = get_sorted_subfolders(self.warm_up_path)
            X, _, objectives_list, mask, _ = load_data(subfolders)

            if self.warm_up_len > 0:
                X = X[:self.warm_up_len]
                objectives_list = objectives_list[:self.warm_up_len]
            else:
                self.warm_up_len = len(X)

            xl = self.problem.xl
            xu = self.problem.xu
            p, c, th = self.problem.p, self.problem.c, self.problem.th
            unique_coeff = (p, c, th)
            self.problem.unique_bugs, (self.problem.bugs, self.problem.bugs_type_list, self.problem.bugs_inds_list, self.problem.interested_unique_bugs) = get_unique_bugs(
                X, objectives_list, mask, xl, xu, unique_coeff, self.problem.objective_weights, return_bug_info=True, consider_interested_bugs=self.problem.consider_interested_bugs
            )

            self.all_pop_run_X = np.array(X)
            self.problem.objectives_list = objectives_list.tolist()

        if self.dt:
            X_list = list(self.X)
            F_list = list(self.F)
            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])
            self.pop = pop
            self.set_off()
            pop = self.off

        elif self.warm_up_path:
            X_list = X[-self.pop_size:]
            current_objectives = objectives_list[-self.pop_size:]

            # current_objectives = np.dot(current_objectives, np.expand_dims(np.array(objective_weights), axis=1))
            # F_list = current_objectives.tolist()

            F_list = get_F(current_objectives, objectives_list, self.problem.objective_weights, self.problem.use_single_objective)

            # print('F_list', sorted(F_list))
            # print('len(self.all_pop_run_X)', len(self.all_pop_run_X))

            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])

            self.pop = pop
            self.set_off()
            pop = self.off

        else:
            # create the initial population
            # pop = Population(0, individual=self.individual)
            # pop = self.sampling.do(self.problem, pop_size, pop=pop, algorithm=self)
            # pop = self.repair.do(self.problem, pop, algorithm=self)
            #
            # if len(pop) < self.pop_size:
            #     remaining_num = self.pop_size - len(pop)
            #     remaining_pop = self.plain_initialization.do(self.problem, remaining_num, algorithm=self)
            #     pop = Population.merge(pop, remaining_pop)

            if self.use_unique_bugs:
                pop = self.initialization.do(self.problem, pop_size, algorithm=self)
            else:
                pop = self.plain_initialization.do(self.problem, pop_size, algorithm=self)
            pop.set("n_gen", self.n_gen)


        if len(pop) > 0:
            self.evaluator.eval(self.problem, pop, algorithm=self)
        print('\n'*5, 'after initialize evaluator', '\n'*5)
        print('len(self.all_pop_run_X)', len(self.all_pop_run_X))
        # if not self.warm_up_path:
        #     # then evaluate using the objective function
        #     self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop

        # print('\n'*5)
        # print(self.pop)
        # print(self.pop.individual)
        # print(self.pop.individual.X)
        # print(self.pop.get("X"))
        # print('\n'*5)









class ClipRepair(Repair):
    """
    A dummy class which can be used to simply do no repair.
    """

    def do(self, problem, pop, **kwargs):
        for i in range(len(pop)):
            pop[i].X = np.clip(pop[i].X, np.array(problem.xl), np.array(problem.xu))
        return pop


class SimpleDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, mask, xu, xl, check_unique_coeff, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask = np.array(mask)
        self.xu = np.array(xu)
        self.xl = np.array(xl)
        self.cmp = lambda a, b: self.is_equal(a, b)
        self.check_unique_coeff = check_unique_coeff
        assert len(self.check_unique_coeff) == 3

    def is_equal(self, a, b):
        if type(b).__module__ == np.__name__:
            b_X = b
        else:
            b_X = b.X
        if type(a).__module__ == np.__name__:
            a_X = a
        else:
            a_X = a.X
        p, c, th = self.check_unique_coeff
        return is_similar(a_X, b_X, self.mask, self.xl, self.xu, p, c, th)




class MyEvaluator(Evaluator):
    def _eval(self, problem, pop, **kwargs):

        super()._eval(problem, pop, **kwargs)
        # print(pop[0].X)
        # hack:
        label_to_id = {label:i for i, label in enumerate(problem.labels)}

        def correct_spawn_locations(all_final_generated_transforms_list_i, i, object_type, keys):
            object_type_plural = object_type
            if object_type in ['pedestrian', 'vehicle']:
                object_type_plural += 's'

            num_of_objects_ind = label_to_id['num_of_'+object_type_plural]
            pop[i].X[num_of_objects_ind] = 0

            empty_slots = deque()
            for j, (x, y, yaw) in enumerate(all_final_generated_transforms_list_i[object_type]):
                if x == None:
                    empty_slots.append(j)
                else:
                    pop[i].X[num_of_objects_ind] += 1
                    if object_type+'_x_'+str(j) not in label_to_id:
                        print(object_type+'_x_'+str(j))
                        print(all_final_generated_transforms_list_i[object_type])
                        raise
                    x_j_ind = label_to_id[object_type+'_x_'+str(j)]
                    y_j_ind = label_to_id[object_type+'_y_'+str(j)]
                    yaw_j_ind = label_to_id[object_type+'_yaw_'+str(j)]


                    # print(object_type, j)
                    # print('x', pop[i].X[x_j_ind], '->', x)
                    # print('y', pop[i].X[y_j_ind], '->', y)
                    # print('yaw', pop[i].X[yaw_j_ind], '->', yaw)
                    pop[i].X[x_j_ind] = x
                    pop[i].X[y_j_ind] = y
                    pop[i].X[yaw_j_ind] = yaw

                    if len(empty_slots) > 0:
                        q = empty_slots.popleft()
                        print('shift', j, 'to', q)
                        for k in keys:
                            # print(k)
                            ind_to = label_to_id[k+'_'+str(q)]
                            ind_from = label_to_id[k+'_'+str(j)]
                            pop[i].X[ind_to] = pop[i].X[ind_from]
                        if object_type == 'vehicle':
                            for p in range(waypoints_num_limit):
                                for waypoint_label in waypoint_labels:
                                    ind_to = label_to_id['_'.join(['vehicle', str(q), waypoint_label, str(p)])]
                                    ind_from = label_to_id['_'.join(['vehicle', str(j), waypoint_label, str(p)])]
                                    pop[i].X[ind_to] = pop[i].X[ind_from]

                        empty_slots.append(j)



        with open('tmp_folder/total.pickle', 'rb') as f_in:
            all_final_generated_transforms_list = pickle.load(f_in)

        # for i, all_final_generated_transforms_list_i in enumerate(all_final_generated_transforms_list):
        #     if all_final_generated_transforms_list_i:
        #         # print(i)
        #         correct_spawn_locations(all_final_generated_transforms_list_i, i, 'static', static_general_labels)
        #         correct_spawn_locations(all_final_generated_transforms_list_i, i, 'pedestrian', pedestrian_general_labels)
        #         correct_spawn_locations(all_final_generated_transforms_list_i, i, 'vehicle', vehicle_general_labels)
                # print('\n'*3)
        # print(pop[0].X)



def customized_minimize(problem,
             algorithm,
             resume_run,
             termination=None,
             **kwargs):
    # create a copy of the algorithm object to ensure no side-effects
    algorithm = copy.deepcopy(algorithm)

    # get the termination if provided as a tuple - create an object
    if termination is not None and not isinstance(termination, Termination):
        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)


    # initialize the algorithm object given a problem
    algorithm.initialize(problem, termination=termination, **kwargs)

    # if no termination could be found add the default termination either for single or multi objective
    if algorithm.termination is None:
        if problem.n_obj > 1:
            algorithm.termination = MultiObjectiveDefaultTermination()
        else:
            algorithm.termination = SingleObjectiveDefaultTermination()

    # actually execute the algorithm
    res = algorithm.solve()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm

    return res



def run_nsga2_dt():
    use_initial_x_and_y = True

    end_when_no_critical_region = True
    cumulative_info = None

    X_filtered = None
    F_filtered = None
    X = None
    y = None
    F = None
    objectives = None
    labels = None
    hv = None
    estimator = None
    critical_unique_leaves = None


    now = datetime.now()
    dt_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    if arguments.warm_up_path:
        subfolders = get_sorted_subfolders(warm_up_path)
        X, _, objectives_list, _, _ = load_data(subfolders)

        if arguments.warm_up_len > 0:
            X = X[:arguments.warm_up_len]
            objectives_list = objectives_list[:arguments.warm_up_len]

        y = determine_y_upon_weights(objectives_list, arguments.objective_weights)
        F = get_F(objectives_list, objectives_list, arguments.objective_weights, arguments.use_single_objective)

        estimator, inds, critical_unique_leaves = filter_critical_regions(np.array(X), y)
        X_filtered = np.array(X)[inds]
        F_filtered = F[inds]




    for i in range(outer_iterations):
        dt_time_str_i = dt_time_str
        dt = True
        if (i == 0 and not arguments.warm_up_path) or np.sum(y)==0:
            dt = False
        X_new, y_new, F_new, _, labels, _, parent_folder, cumulative_info, all_pop_run_X, objective_list, objective_weights = run_ga(True, dt, X_filtered, F_filtered, estimator, critical_unique_leaves, dt_time_str_i, i, cumulative_info)

        all_pop_run_X
        objective_list

        if finish_after_has_run and cumulative_info['has_run'] > has_run_num:
            break

        if len(X_new) == 0:
            break

        if i == 0 and not arguments.warm_up_path:
            X = X_new
            y = y_new
            F = F_new
            # objectives = objectives_new
            # hv = hv_new
        else:
            X = np.concatenate([X, X_new])
            y = np.concatenate([y, y_new])
            F = np.concatenate([F, F_new])
            # objectives = np.concatenate([objectives, objectives_new])
            # hv = np.concatenate([hv, hv_new])




        estimator, inds, critical_unique_leaves = filter_critical_regions(X, y)
        # print(X, F, inds)
        X_filtered = X[inds]
        F_filtered = F[inds]

        if len(X_filtered) == 0 and end_when_no_critical_region:
            break


    # Save data
    # has_run_list = cumulative_info['has_run_list']
    # time_list = cumulative_info['time_list']
    # bugs_num_list = cumulative_info['bugs_num_list']
    # unique_bugs_num_list = cumulative_info['unique_bugs_num_list']
    #
    #
    # n_gen = global_n_gen
    #
    # dt_save_file = '_'.join([algorithm_name, route_type, scenario_type, ego_car_model, str(n_gen), str(pop_size), str(outer_iterations), dt_time_str])
    #
    # pth = os.path.join(parent_folder, dt_save_file)
    # np.savez(pth, X=X, y=y, F=F, objectives=objectives, time_list=time_list, bugs_num_list=bugs_num_list, unique_bugs_num_list=unique_bugs_num_list, labels=labels, has_run_list=has_run_list, route_type=route_type, scenario_type=scenario_type)
    #
    # print('npz saved')





def run_ga(call_from_dt=False, dt=False, X=None, F=None, estimator=None, critical_unique_leaves=None, dt_time_str=None, dt_iter=None, cumulative_info=None):

    n_gen = global_n_gen

    if call_from_dt:
        termination_condition = 'generations'
        if dt and len(list(X)) == 0:
            print('No critical leaves!!! Start from random sampling!!!')
            dt = False
    else:
        termination_condition = global_termination_condition


    # scenario_type = 'leading_car_braking'
    customized_d = customized_bounds_and_distributions[scenario_type]
    route_info = customized_routes[route_type]

    town_name = route_info['town_name']
    scenario = 'Scenario12' # This is only for compatibility purpose
    direction = route_info['direction']
    route = route_info['route_id']
    location_list = route_info['location_list']

    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str

    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)


    if call_from_dt:
        time_str = dt_time_str
    else:
        now = datetime.now()
        p, c, th = check_unique_coeff
        time_str = now.strftime("%Y_%m_%d_%H_%M_%S")+','+'_'.join([str(pop_size), str(global_n_gen), rank_mode, str(has_run_num), str(initial_fit_th), str(pgd_eps), str(adv_conf_th), str(attack_stop_conf), 'coeff', str(p), str(c), str(th), uncertainty, model_type, 'n_offsprings', str(n_offsprings), str(mating_max_iterations), str(sample_multiplier), 'only_unique', str(only_run_unique_cases), 'eps', str(pgd_eps)])


    cur_parent_folder = make_hierarchical_dir([root_folder, algorithm_name, route_type, scenario_type, ego_car_model, time_str])

    if call_from_dt:
        parent_folder = make_hierarchical_dir([cur_parent_folder, str(dt_iter)])
    else:
        parent_folder = cur_parent_folder

    bug_parent_folder = make_hierarchical_dir([parent_folder, 'bugs'])
    non_bug_parent_folder = make_hierarchical_dir([parent_folder, 'non_bugs'])




    if resume_run:
        with open(save_path, 'rb') as f_in:
            problem = pickle.load(f_in)

    else:
        problem = MyProblem(elementwise_evaluation=False, bug_parent_folder=bug_parent_folder, non_bug_parent_folder=non_bug_parent_folder, town_name=town_name, scenario=scenario, direction=direction, route_str=route_str, scenario_file=scenario_file, ego_car_model=ego_car_model, scheduler_port=scheduler_port, dashboard_address=dashboard_address, customized_config=customized_d, ports=ports, episode_max_time=episode_max_time,
        call_from_dt=call_from_dt, dt=dt, estimator=estimator, critical_unique_leaves=critical_unique_leaves, cumulative_info=cumulative_info, objective_weights=objective_weights,
        check_unique_coeff=check_unique_coeff, consider_interested_bugs=consider_interested_bugs,
        record_every_n_step=record_every_n_step, use_single_objective=use_single_objective)




    # deal with real and int separately
    crossover = MixedVariableCrossover(problem.mask, {
        "real": get_crossover("real_sbx", prob=0.8, eta=5),
        "int": get_crossover("int_sbx", prob=0.8, eta=5)
    })

    mutation = MixedVariableMutation(problem.mask, {
        "real": get_mutation("real_pm", eta=5, prob=int(0.05*problem.n_var)),
        "int": get_mutation("int_pm", eta=5, prob=int(0.05*problem.n_var))
    })

    # survival = Single_Objective_Survival()

    selection = TournamentSelection(func_comp=binary_tournament)
    repair = ClipRepair()

    if use_unique_bugs:
        eliminate_duplicates = SimpleDuplicateElimination(mask=problem.mask, xu=problem.xu, xl=problem.xl, check_unique_coeff=problem.check_unique_coeff)
    else:
        eliminate_duplicates = NoDuplicateElimination()

    mating = MyMatingVectorized(selection,
                    crossover,
                    mutation,
                    use_unique_bugs,
                    emcmc,
                    repair=repair,
                    eliminate_duplicates=eliminate_duplicates,
                    mating_max_iterations=mating_max_iterations)


    sampling = MySamplingVectorized(use_unique_bugs=use_unique_bugs, check_unique_coeff=problem.check_unique_coeff, sample_multiplier=sample_multiplier)

    plain_sampling = MySamplingVectorized(use_unique_bugs=False, check_unique_coeff=problem.check_unique_coeff, sample_multiplier=sample_multiplier)

    # TBD: customize mutation and crossover to better fit our problem. e.g.
    # might deal with int and real separately
    algorithm = NSGA2_DT(dt=dt, X=X, F=F, emcmc=emcmc, plain_sampling=plain_sampling, algorithm_name=algorithm_name,
                      pop_size=pop_size,
                      sampling=sampling,
                      crossover=crossover,
                      mutation=mutation,
                      eliminate_duplicates=eliminate_duplicates,
                      repair=repair,
                      mating=mating,
                      n_offsprings=n_offsprings,
                      rank_mode=rank_mode,
                      initial_fit_th=initial_fit_th,
                      min_bug_num_to_fit_dnn=min_bug_num_to_fit_dnn,
                      ranking_model=ranking_model, use_unique_bugs=use_unique_bugs, pgd_eps=pgd_eps,
                      adv_conf_th=adv_conf_th, attack_stop_conf=attack_stop_conf,
                      use_single_nn=use_single_nn,
                      uncertainty=uncertainty,
                      model_type=model_type,
                      survival_multiplier=survival_multiplier,
                      explore_iter_num=explore_iter_num,
                      exploit_iter_num=exploit_iter_num,
                      high_conf_num=high_conf_num,
                      low_conf_num=low_conf_num,
                      warm_up_path=warm_up_path,
                      warm_up_len=warm_up_len,
                      use_alternate_nn=use_alternate_nn,
                      diversity_mode=diversity_mode,
                      regression_nn_use_running_data=regression_nn_use_running_data,
                      adv_exploitation_only=adv_exploitation_only,
                      uncertainty_exploration=uncertainty_exploration,
                      only_run_unique_cases=only_run_unique_cases)


    if termination_condition == 'generations':
        termination = ('n_gen', n_gen)
    elif termination_condition == 'max_time':
        termination = ('time', max_running_time)
    else:
        termination = ('n_gen', n_gen)


    # close simulator(s)
    atexit.register(exit_handler, ports)

    # TypeError: can't pickle _asyncio.Task objects when save_history = True
    # verbose has to be set to False to avoid an error when algorithm_name=random
    res = customized_minimize(problem,
                   algorithm,
                   resume_run,
                   termination=termination,
                   seed=0,
                   verbose=False,
                   save_history=False,
                   evaluator=MyEvaluator())

    print('We have found', len(problem.bugs), 'bugs in total.')


    # print("Best solution found: %s" % res.X)
    # print("Function value: %s" % res.F)
    # print("Constraint violation: %s" % res.CV)

    # for drawing hv
    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point=np.array([7.0, 7.0, 7.0, 7.0, 7.0]))
    # collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]
    # receive the population in each generation
    obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]
    # calculate for each generation the HV metric
    hv = np.array([metric.calc(f) for f in obj_and_feasible_each_gen])


    # print('problem.x_list, problem.F_list', problem.x_list, problem.F_list)
    if len(problem.x_list) > 0:
        X = np.stack(problem.x_list)
        F = np.concatenate(problem.F_list)
        objectives = np.stack(problem.objectives_list)
    else:
        X = []
        F = []
        objectives = []
    # print('X, F, objectives', X, F, objectives)
    y = np.array(problem.y_list)
    time_list = np.array(problem.time_list)
    bugs_num_list = np.array(problem.bugs_num_list)
    unique_bugs_num_list = np.array(problem.unique_bugs_num_list)
    labels = problem.labels
    has_run = problem.has_run
    has_run_list = problem.has_run_list

    mask = problem.mask
    xl = problem.xl
    xu = problem.xu
    p = problem.p
    c = problem.c
    th = problem.th

    # save another data npz for easy comparison with dt results


    non_dt_save_file = '_'.join([algorithm_name, route_type, scenario_type, ego_car_model, str(n_gen), str(pop_size)])
    pth = os.path.join(bug_parent_folder, non_dt_save_file)

    np.savez(pth, X=X, y=y, F=F, objectives=objectives, time_list=time_list, bugs_num_list=bugs_num_list, unique_bugs_num_list=unique_bugs_num_list, has_run=has_run, has_run_list=has_run_list, labels=labels, hv=hv, mask=mask, xl=xl, xu=xu, p=p, c=c, th=th, route_type=route_type, scenario_type=scenario_type, rank_mode=rank_mode, ranking_model=ranking_model, initial_fit_th=initial_fit_th)
    print('npz saved')


    if save:
        with open(save_path, 'wb') as f_out:
            pickle.dump(problem, f_out)
            print('-'*100, 'pickled')


    cumulative_info = {
        'has_run': problem.has_run,
        'start_time': problem.start_time,
        'counter': problem.counter,
        'time_list': problem.time_list,
        'bugs': problem.bugs,
        'unique_bugs': problem.unique_bugs,
        'interested_unique_bugs': problem.interested_unique_bugs,
        'bugs_type_list': problem.bugs_type_list,
        'bugs_inds_list': problem.bugs_inds_list,
        'bugs_num_list': problem.bugs_num_list,
        'unique_bugs_num_list': problem.unique_bugs_num_list,
        'has_run_list': problem.has_run_list
    }


    return X, y, F, objectives, labels, hv, cur_parent_folder, cumulative_info, algorithm.all_pop_run_X, problem.objectives_list, problem.objective_weights

if __name__ == '__main__':
    if algorithm_name in ['nsga2-un-dt', 'nsga2-dt']:
        run_nsga2_dt()
    else:
        run_ga()
