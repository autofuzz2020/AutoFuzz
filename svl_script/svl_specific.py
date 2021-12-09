import os
import pickle
import shutil
from distutils.dir_util import copy_tree
import numpy as np

from customized_utils import make_hierarchical_dir, emptyobject, is_distinct_vectorized


import lgsvl
from svl_script.object_params import Pedestrian, Vehicle, Static, Waypoint
from svl_script.simulation_utils import start_simulation
from svl_script.scene_configs import customized_bounds_and_distributions, customized_routes
from svl_script.object_types import pedestrian_types, car_types, large_car_types, static_types

def convert_x_to_customized_data(
    x,
    fuzzing_content,
    port
):

    waypoints_num_limit = fuzzing_content.search_space_info.waypoints_num_limit
    num_of_static_max = fuzzing_content.search_space_info.num_of_static_max
    num_of_pedestrians_max = fuzzing_content.search_space_info.num_of_pedestrians_max
    num_of_vehicles_max = fuzzing_content.search_space_info.num_of_vehicles_max

    customized_center_transforms = fuzzing_content.customized_center_transforms
    parameters_min_bounds = fuzzing_content.parameters_min_bounds
    parameters_max_bounds = fuzzing_content.parameters_max_bounds

    # parameters
    # global

    num_of_static = int(x[0])
    num_of_pedestrians = int(x[1])
    num_of_vehicles = int(x[2])
    damage = int(x[3])
    rain = x[4]
    fog = x[5]
    wetness = x[6]
    cloudiness = x[7]
    hour = x[8]

    ind = 9

    # static
    static_list = []
    for i in range(num_of_static_max):
        if i < num_of_static:
            static_i = Static(
                model=int(x[ind]),
                x=x[ind+1],
                y=x[ind+2],
            )
            static_list.append(static_i)
        ind += 3

    # pedestrians
    pedestrians_list = []
    for i in range(num_of_pedestrians_max):
        if i < num_of_pedestrians:
            pedestrian_type_i = int(x[ind])
            pedestrian_x_i = x[ind+1]
            pedestrian_y_i = x[ind+2]
            pedestrian_speed_i = x[ind+3]
            ind += 4

            pedestrian_waypoints_i = []
            for _ in range(waypoints_num_limit):
                pedestrian_waypoints_i.append(Waypoint(x[ind], x[ind+1], x[ind+2], x[ind+3]))
                ind += 4

            pedestrian_i = Pedestrian(
                model=pedestrian_type_i,
                x=pedestrian_x_i,
                y=pedestrian_y_i,
                speed=pedestrian_speed_i,
                waypoints=pedestrian_waypoints_i,
            )

            pedestrians_list.append(pedestrian_i)

        else:
            ind += 4 + waypoints_num_limit * 4

    # vehicles
    vehicles_list = []
    for i in range(num_of_vehicles_max):
        if i < num_of_vehicles:
            vehicle_type_i = int(x[ind])
            vehicle_x_i = x[ind+1]
            vehicle_y_i = x[ind+2]
            vehicle_speed_i = x[ind+3]
            ind += 4

            vehicle_waypoints_i = []
            for _ in range(waypoints_num_limit):
                vehicle_waypoints_i.append(Waypoint(x[ind], x[ind+1], x[ind+2], x[ind+3]))
                ind += 4

            vehicle_i = Vehicle(
                model=vehicle_type_i,
                x=vehicle_x_i,
                y=vehicle_y_i,
                speed=vehicle_speed_i,
                waypoints=vehicle_waypoints_i,
            )

            vehicles_list.append(vehicle_i)

        else:
            ind += 4 + waypoints_num_limit * 4




    customized_data = {
        # "num_of_static": num_of_static,
        # "num_of_pedestrians": num_of_pedestrians,
        # "num_of_vehicles": num_of_vehicles,
        "damage": damage,
        "rain": rain,
        "fog": fog,
        "wetness": wetness,
        "cloudiness": cloudiness,
        "hour": hour,
        "static_list": static_list,
        "pedestrians_list": pedestrians_list,
        "vehicles_list": vehicles_list,
        "customized_center_transforms": customized_center_transforms,
        # "parameters_min_bounds": parameters_min_bounds,
        # "parameters_max_bounds": parameters_max_bounds,
    }

    return customized_data

# TBD: implement SVL specific unique bugs/objective related functions
# currently borrowed from carla implementation
def estimate_objectives(save_path, default_objectives=np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.]), verbose=True):

    events_path = os.path.join(save_path, "events.txt")
    deviations_path = os.path.join(save_path, "deviations.txt")

    # set thresholds to avoid too large influence
    ego_linear_speed = 0
    min_d = 20
    offroad_d = 7
    wronglane_d = 7
    dev_dist = 0
    d_angle_norm = 1

    ego_linear_speed_max = 7
    dev_dist_max = 7

    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0
    is_collision = 0
    if os.path.exists(deviations_path):
        with open(deviations_path, "r") as f_in:
            for line in f_in:
                type, d = line.split(",")
                d = float(d)
                if type == "min_d":
                    min_d = np.min([min_d, d])
                elif type == "offroad_d":
                    offroad_d = np.min([offroad_d, d])
                elif type == "wronglane_d":
                    wronglane_d = np.min([wronglane_d, d])
                elif type == "dev_dist":
                    dev_dist = np.max([dev_dist, d])
                elif type == "d_angle_norm":
                    d_angle_norm = np.min([d_angle_norm, d])

    x = None
    y = None
    object_type = None

    if not os.path.exists(events_path):
        route_completion = True
    else:
        route_completion = False
        with open(events_path, 'r') as f_in:
            tokens = f_in.read().split('\n')[0].split(',')
            if tokens[0] == 'fail_to_finish':
                pass
            else:
                _, ego_linear_speed, object_type, x, y = tokens
                ego_linear_speed, x, y = float(ego_linear_speed), float(x), float(y)
                if ego_linear_speed > 0.1:
                    is_collision = 1


    # limit impact of too large values
    ego_linear_speed = np.min([ego_linear_speed, ego_linear_speed_max])
    dev_dist = np.min([dev_dist, dev_dist_max])

    return (
        [
            ego_linear_speed,
            min_d,
            d_angle_norm,
            offroad_d,
            wronglane_d,
            dev_dist,
            is_collision,
            is_offroad,
            is_wrong_lane,
            is_run_red_light,
        ],
        (x, y),
        object_type,
        route_completion,
    )


def check_bug(objectives):
    # speed needs to be larger than 0.1 to avoid false positive
    return objectives[0] > 0.1 or objectives[-3] or objectives[-2] or objectives[-1]

def get_if_bug_list(objectives_list):
    if_bug_list = []
    for objective in objectives_list:
        if_bug_list.append(check_bug(objective))
    return np.array(if_bug_list)


def process_specific_bug(
    bug_type_ind, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
):
    if len(bugs) == 0:
        return [], [], 0
    verbose = True
    chosen_bugs = np.array(bugs_type_list) == bug_type_ind

    specific_bugs = np.array(bugs)[chosen_bugs]
    specific_bugs_inds_list = np.array(bugs_inds_list)[chosen_bugs]

    # unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(
    #     specific_bugs, mask, xl, xu, p, c, th
    # )

    specific_distinct_inds = is_distinct_vectorized(specific_bugs, [], mask, xl, xu, p, c, th, verbose=verbose)
    unique_specific_bugs = specific_bugs[specific_distinct_inds]

    unique_specific_bugs_inds_list = specific_bugs_inds_list[specific_distinct_inds]

    return (
        list(unique_specific_bugs),
        list(unique_specific_bugs_inds_list),
        len(unique_specific_bugs),
    )

def classify_bug_type(objectives, object_type=''):
    bug_str = ''
    bug_type = 5
    if objectives[0] > 0.1:
        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types,
        'large_car_collision':large_car_types, 'static_collision':static_types}
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
    return bug_type, bug_str

def get_unique_bugs(
    X, objectives_list, mask, xl, xu, unique_coeff, objective_weights, return_mode='unique_inds_and_interested_and_bugcounts', consider_interested_bugs=1, bugs_type_list=[], bugs=[], bugs_inds_list=[], trajectory_vector_list=[]
):
    p, c, th = unique_coeff
    # hack:
    if len(bugs) == 0:
        for i, (x, objectives) in enumerate(zip(X, objectives_list)):
            if check_bug(objectives):
                bug_type, _ = classify_bug_type(objectives)
                bugs_type_list.append(bug_type)
                bugs.append(x)
                bugs_inds_list.append(i)

    (
        unique_collision_bugs,
        unique_collision_bugs_inds_list,
        unique_collision_num,
    ) = process_specific_bug(
        1, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_offroad_bugs,
        unique_offroad_bugs_inds_list,
        unique_offroad_num,
    ) = process_specific_bug(
        2, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_wronglane_bugs,
        unique_wronglane_bugs_inds_list,
        unique_wronglane_num,
    ) = process_specific_bug(
        3, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_redlight_bugs,
        unique_redlight_bugs_inds_list,
        unique_redlight_num,
    ) = process_specific_bug(
        4, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )

    unique_bugs = unique_collision_bugs + unique_offroad_bugs + unique_wronglane_bugs + unique_redlight_bugs
    unique_bugs_num = len(unique_bugs)
    unique_bugs_inds_list = unique_collision_bugs_inds_list + unique_offroad_bugs_inds_list + unique_wronglane_bugs_inds_list + unique_redlight_bugs_inds_list

    if consider_interested_bugs:
        collision_activated = np.sum(objective_weights[:3] != 0) > 0
        offroad_activated = (np.abs(objective_weights[3]) > 0) | (
            np.abs(objective_weights[5]) > 0
        )
        wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
            np.abs(objective_weights[5]) > 0
        )
        red_light_activated = np.abs(objective_weights[-1]) > 0

        interested_unique_bugs = []
        if collision_activated:
            interested_unique_bugs += unique_collision_bugs
        if offroad_activated:
            interested_unique_bugs += unique_offroad_bugs
        if wronglane_activated:
            interested_unique_bugs += unique_wronglane_bugs
        if red_light_activated:
            interested_unique_bugs += unique_redlight_bugs
    else:
        interested_unique_bugs = unique_bugs

    num_of_collisions = np.sum(np.array(bugs_type_list)==1)
    num_of_offroad = np.sum(np.array(bugs_type_list)==2)
    num_of_wronglane = np.sum(np.array(bugs_type_list)==3)
    num_of_redlight = np.sum(np.array(bugs_type_list)==4)

    if return_mode == 'unique_inds_and_interested_and_bugcounts':
        return unique_bugs, unique_bugs_inds_list, interested_unique_bugs, [num_of_collisions, num_of_offroad, num_of_wronglane, num_of_redlight,
        unique_collision_num, unique_offroad_num, unique_wronglane_num, unique_redlight_num]
    elif return_mode == 'return_bug_info':
        return unique_bugs, (bugs, bugs_type_list, bugs_inds_list, interested_unique_bugs)
    elif return_mode == 'return_indices':
        return unique_bugs, unique_bugs_inds_list
    else:
        return unique_bugs


def choose_weight_inds(objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    if collision_activated:
        weight_inds = np.arange(0,3)
    elif offroad_activated or wronglane_activated:
        weight_inds = np.arange(3,6)
    elif red_light_activated:
        weight_inds = np.arange(9,10)
    else:
        raise
    return weight_inds

def determine_y_upon_weights(objective_list, objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y = np.zeros(len(objective_list))
    for i, obj in enumerate(objective_list):
        cond = 0
        if collision_activated:
            cond |= obj[0] > 0.1
        if offroad_activated:
            cond |= obj[-3] == 1
        if wronglane_activated:
            cond |= obj[-2] == 1
        if red_light_activated:
            cond |= obj[-1] == 1
        y[i] = cond

    return y

def get_all_y(objective_list, objective_weights):
    # is_collision, is_offroad, is_wrong_lane, is_run_red_light
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y_list = np.zeros((4, len(objective_list)))

    for i, obj in enumerate(objective_list):
        if collision_activated:
            y_list[0, i] = obj[0] > 0.1
        if offroad_activated:
            y_list[1, i] = obj[-3] == 1
        if wronglane_activated:
            y_list[2, i] = obj[-2] == 1
        if red_light_activated:
            y_list[3, i] = obj[-1] == 1

    return y_list

def determine_y_upon_weights(objective_list, objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y = np.zeros(len(objective_list))
    for i, obj in enumerate(objective_list):
        cond = 0
        if collision_activated:
            cond |= obj[0] > 0.1
        if offroad_activated:
            cond |= obj[-3] == 1
        if wronglane_activated:
            cond |= obj[-2] == 1
        if red_light_activated:
            cond |= obj[-1] == 1
        y[i] = cond

    return y

def get_all_y(objective_list, objective_weights):
    # is_collision, is_offroad, is_wrong_lane, is_run_red_light
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y_list = np.zeros((4, len(objective_list)))

    for i, obj in enumerate(objective_list):
        if collision_activated:
            y_list[0, i] = obj[0] > 0.1
        if offroad_activated:
            y_list[1, i] = obj[-3] == 1
        if wronglane_activated:
            y_list[2, i] = obj[-2] == 1
        if red_light_activated:
            y_list[3, i] = obj[-1] == 1

    return y_list

#####################################################################




def run_svl_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port):
    '''
    objectives needs to be consistent with the specified objectives



    not using:
    launch_server,
    port

    '''
    print('\n'*3, 'x:\n', x, '\n'*3)
    customized_data = convert_x_to_customized_data(x, fuzzing_content, port)
    parent_folder = fuzzing_arguments.parent_folder
    episode_max_time = fuzzing_arguments.episode_max_time
    mean_objectives_across_generations_path = fuzzing_arguments.mean_objectives_across_generations_path
    ego_car_model = fuzzing_arguments.ego_car_model


    # 5.0: 47b529db-0593-4908-b3e7-4b24a32a0f70
    # 6.0: c354b519-ccf0-4c1c-b3cc-645ed5751bb5
    # 6.0(modular testing): 2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921
    # 6.0(no telephoto camera and clock sensor): 4622f73a-250e-4633-9a3d-901ede6b9551
    # 6.0(no clock sensor): f68151d1-604c-438e-a1a5-aa96d5581f4b
    # 6.0(with signal sensor): 9272dd1a-793a-45b2-bff4-3a160b506d75
    # 6.0(modular testing, birdview): b20c0d8a-f310-46b2-a639-6ce6be4f2b14
    if ego_car_model == 'apollo_6_with_signal':
        model_id = '9272dd1a-793a-45b2-bff4-3a160b506d75'
    elif ego_car_model == 'apollo_6_modular':
        model_id = '2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921'
    elif ego_car_model == 'apollo_6_modular_2gt':
        model_id = 'f0daed3e-4b1e-46ce-91ec-21149fa31758'
    else:
        print('ego car model is invalid:', ego_car_model)
        raise
    print('\n'*3, 'ego car model is:', ego_car_model, '\n'*3)

    route_info = sim_specific_arguments.route_info
    deviations_folder = os.path.join(parent_folder, "current_run_data")
    if os.path.exists(deviations_folder):
        shutil.rmtree(deviations_folder)
    os.mkdir(deviations_folder)

    arguments = emptyobject(deviations_folder=deviations_folder, model_id=model_id, route_info=route_info, record_every_n_step=fuzzing_arguments.record_every_n_step, counter=counter)



    start_simulation(customized_data, arguments, sim_specific_arguments, launch_server, episode_max_time)
    objectives, loc, object_type, route_completion = estimate_objectives(deviations_folder)



    if parent_folder:
        if check_bug(objectives):
            is_bug = True
            bug_type, bug_str = classify_bug_type(objectives, object_type)
        # elif not route_completion:
        #     is_bug = True
        #     bug_type, bug_str = 5, 'fail to complete'
        else:
            is_bug = False
            bug_type, bug_str = None, None
        if is_bug:
            with open(mean_objectives_across_generations_path, 'a') as f_out:
                f_out.write(str(counter)+','+bug_str+'\n')

        bug_folder = make_hierarchical_dir([parent_folder, 'bugs'])
        non_bug_folder = make_hierarchical_dir([parent_folder, 'non_bugs'])
        if is_bug:
            cur_folder = make_hierarchical_dir([bug_folder, str(counter)])
        else:
            cur_folder = make_hierarchical_dir([non_bug_folder, str(counter)])



    xl = [pair[1] for pair in fuzzing_content.parameters_min_bounds.items()]
    xu = [pair[1] for pair in fuzzing_content.parameters_max_bounds.items()]

    import copy
    sim_specific_arguments_copy = copy.copy(sim_specific_arguments)
    sim_specific_arguments_copy.sim = None
    run_info = {
        # for analysis
        'x': x,
        'objectives': objectives,
        'labels': fuzzing_content.labels,

        'is_bug': is_bug,
        'bug_type': bug_type,

        'xl': np.array(xl),
        'xu': np.array(xu),
        'mask': fuzzing_content.mask,

        # for rerun
        'fuzzing_content': fuzzing_content,
        'fuzzing_arguments': fuzzing_arguments,
        'sim_specific_arguments': sim_specific_arguments_copy,
        'dt_arguments': dt_arguments,
        'counter': counter,

        # helpful info
        'route_completion': route_completion,

        # for correction
        # 'all_final_generated_transforms': all_final_generated_transforms,
    }


    copy_tree(deviations_folder, cur_folder)

    with open(cur_folder+'/'+'cur_info.pickle', 'wb') as f_out:
        pickle.dump(run_info, f_out)




    return objectives, run_info


def initialize_svl_specific(fuzzing_arguments):
    route_info = customized_routes[fuzzing_arguments.route_type]
    sim_specific_arguments = emptyobject(route_info=route_info, sim=None)
    return sim_specific_arguments
