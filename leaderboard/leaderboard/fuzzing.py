#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import torchvision

# addition
import pathlib
import time
import numpy as np
import traceback
import logging


import carla
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.control_loss import *
from srunner.scenarios.follow_leading_vehicle import *
from srunner.scenarios.maneuver_opposite_direction import *
from srunner.scenarios.no_signal_junction_crossing import *
from srunner.scenarios.object_crash_intersection import *
from srunner.scenarios.object_crash_vehicle import *
from srunner.scenarios.opposite_vehicle_taking_priority import *
from srunner.scenarios.other_leading_vehicle import *
from srunner.scenarios.signalized_junction_left_turn import *
from srunner.scenarios.signalized_junction_right_turn import *
from srunner.scenarios.change_lane import *
from srunner.scenarios.cut_in import *

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.autoagents.agent_wrapper import SensorConfigurationInvalid
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer


from leaderboard.customized.object_params import Static, Pedestrian, Vehicle
from leaderboard.utils.route_parser import RouteParser


from customized_utils import (
    create_transform,
    specify_args,
    is_port_in_use,
    make_hierarchical_dir,
    start_server,
    port_to_gpu,
)
from object_types import WEATHERS
from leaderboard.utils.route_manipulation import interpolate_trajectory

from psutil import process_iter


sensors_to_icons = {
    "sensor.camera.semantic_segmentation": "carla_camera",
    "sensor.camera.rgb": "carla_camera",
    "sensor.lidar.ray_cast": "carla_lidar",
    "sensor.other.radar": "carla_radar",
    "sensor.other.gnss": "carla_gnss",
    "sensor.other.imu": "carla_imu",
    "sensor.opendrive_map": "carla_opendrive_map",
    "sensor.speedometer": "carla_speedometer",
}


class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds

    # modification: 20.0 -> 10.0
    frame_rate = 10.0  # in Hz

    def __init__(
        self, args, statistics_manager, launch_server=False, episode_max_time=10000
    ):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """

        self.statistics_manager = statistics_manager
        self.sensors = []
        self._vehicle_lights = (
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        )
        self.episode_max_time = episode_max_time

        # First of all, we need to create the client that will send the requests
        # to the simulator.

        # This is currently set to be consistent with os.environ['HAS_DISPLAY'].
        # however, it is possible to control them separately.
        if os.environ["HAS_DISPLAY"] == "0":
            os.environ["DISPLAY"] = ""

        gpu = port_to_gpu(int(args.port))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if launch_server:
            start_server(args.port)

        while True:
            try:
                self.client = carla.Client(args.host, int(args.port))
                break
            except:
                logging.exception("__init__ error")
                traceback.print_exc()

        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion("0.9.9"):
            raise ImportError(
                "CARLA version 0.9.9 or newer required. CARLA version found: {}".format(
                    dist
                )
            )

        # Load agent
        module_name = os.path.basename(args.agent).split(".")[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(
            args.debug,
            args.sync,
            args.challenge_mode,
            args.track,
            self.client_timeout,
            self.episode_max_time,
        )

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # addition
        parent_folder = args.save_folder
        if not os.path.exists(parent_folder):
            os.mkdir(parent_folder)
        string = pathlib.Path(os.environ["ROUTES"]).stem
        current_record_folder = pathlib.Path(parent_folder) / string

        if os.path.exists(str(current_record_folder)):
            import shutil

            shutil.rmtree(current_record_folder)
        current_record_folder.mkdir(exist_ok=False)
        (current_record_folder / "rgb").mkdir()
        (current_record_folder / "rgb_left").mkdir()
        (current_record_folder / "rgb_right").mkdir()
        (current_record_folder / "topdown").mkdir()
        (current_record_folder / "rgb_with_car").mkdir()
        # if args.agent == 'leaderboard/team_code/auto_pilot.py':
        #     (current_record_folder / 'topdown').mkdir()

        self.save_path = str(current_record_folder / "events.txt")

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup(True)
        if hasattr(self, "manager") and self.manager:
            del self.manager
        if hasattr(self, "world") and self.world:
            del self.world

        # addition: manually delete client to avoid RuntimeError: Resource temporarily unavailable
        del self.client

    def _cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """

        self.client.stop_recorder()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if ego:
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if hasattr(self, "agent_instance") and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(
                    CarlaDataProvider.setup_actor(
                        vehicle.model,
                        vehicle.transform,
                        vehicle.rolename,
                        True,
                        color=vehicle.color,
                        vehicle_category=vehicle.category,
                    )
                )
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = (
                        CarlaDataProvider.get_world().get_actors().filter("vehicle.*")
                    )
                    for carla_vehicle in carla_vehicles:
                        if (
                            carla_vehicle.attributes["role_name"]
                            == ego_vehicle.rolename
                        ):
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider and CarlaDataProvider
        """

        while True:
            try:
                self.world = self.client.load_world(town)
                break
            except:
                logging.exception("_load_and_wait_for_world error")
                traceback.print_exc()

                start_server(args.port)
                self.client = carla.Client(args.host, int(args.port))

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True

        self.world.apply_settings(settings)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        spectator = CarlaDataProvider.get_world().get_spectator()
        spectator.set_transform(
            carla.Transform(carla.Location(x=0, y=0, z=20), carla.Rotation(pitch=-90))
        )

        # Wait for the world to be ready
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, args, config, customized_data):
        """
        Load and run the scenario given by config
        """
        # hack:
        if args.weather_index == -1:
            weather = customized_data["fine_grained_weather"]
        else:
            weather = WEATHERS[args.weather_index]

        config.weather = weather
        config.friction = customized_data["friction"]
        config.cur_server_port = customized_data["port"]

        if not self._load_and_wait_for_world(args, config.town, config.ego_vehicles):
            self._cleanup()
            return

        _, route = interpolate_trajectory(self.world, config.trajectory)
        customized_data["center_transform"] = route[int(len(route) // 2)][0]

        """
        customized non-default center transforms for actors
        ['waypoint_ratio', 'absolute_location']
        """
        for k, v in customized_data["customized_center_transforms"].items():
            if v[0] == "waypoint_ratio":
                r = v[1] / 100
                ind = np.min([int(len(route) * r), len(route) - 1])
                loc = route[ind][0].location
                customized_data[k] = create_transform(loc.x, loc.y, 0, 0, 0, 0)
                print("waypoint_ratio", loc.x, loc.y)
            elif v[0] == "absolute_location":
                customized_data[k] = create_transform(v[1], v[2], 0, 0, 0, 0)
            else:
                print("unknown key", k)

        if "weather_index" in customized_data:
            print("-" * 100)
            print("port :", customized_data["port"])
            print(
                "center_transform :",
                "(",
                customized_data["center_transform"].location.x,
                customized_data["center_transform"].location.y,
                ")",
            )
            print("friction :", customized_data["friction"])
            print("weather_index :", customized_data["weather_index"])
            print("num_of_static :", customized_data["num_of_static"])
            print("num_of_pedestrians :", customized_data["num_of_pedestrians"])
            print("num_of_vehicles :", customized_data["num_of_vehicles"])
            print("-" * 100)

        agent_class_name = getattr(self.module_agent, "get_entry_point")()
        try:
            self.agent_instance = getattr(self.module_agent, agent_class_name)(
                args.agent_config
            )

            # addition
            # self.agent_instance.set_trajectory(config.trajectory)
            self.agent_instance.set_args(args)

            config.agent = self.agent_instance
            self.sensors = [
                sensors_to_icons[sensor["type"]]
                for sensor in self.agent_instance.sensors()
            ]
        except Exception as e:
            print("Could not setup required agent due to {}".format(e))
            traceback.print_exc()
            self._cleanup()
            return

        # Prepare scenario
        print("Preparing scenario: " + config.name)

        try:
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            # print('\n'*10, 'RouteScenario config.cur_server_port', config.cur_server_port, '\n'*10)
            scenario = RouteScenario(
                world=self.world,
                config=config,
                debug_mode=args.debug,
                customized_data=customized_data,
            )

        except Exception as exception:
            print("The scenario cannot be loaded")
            if args.debug:
                traceback.print_exc()
            print(exception)
            self._cleanup()
            return

        # Set the appropriate weather conditions
        weather = carla.WeatherParameters(
            cloudiness=config.weather.cloudiness,
            precipitation=config.weather.precipitation,
            precipitation_deposits=config.weather.precipitation_deposits,
            wind_intensity=config.weather.wind_intensity,
            sun_azimuth_angle=config.weather.sun_azimuth_angle,
            sun_altitude_angle=config.weather.sun_altitude_angle,
            fog_density=config.weather.fog_density,
            fog_distance=config.weather.fog_distance,
            wetness=config.weather.wetness,
        )

        self.world.set_weather(weather)

        # Set the appropriate road friction
        if config.friction is not None:
            friction_bp = self.world.get_blueprint_library().find(
                "static.trigger.friction"
            )
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute("friction", str(config.friction))
            friction_bp.set_attribute("extent_x", str(extent.x))
            friction_bp.set_attribute("extent_y", str(extent.y))
            friction_bp.set_attribute("extent_z", str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            self.world.spawn_actor(friction_bp, transform)

        # night mode
        if config.weather.sun_altitude_angle < 0.0:
            for vehicle in scenario.ego_vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
            # addition: to turn on lights of
            actor_list = self.world.get_actors()
            vehicle_list = actor_list.filter("*vehicle*")
            for vehicle in vehicle_list:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}.log".format(args.record, config.name))
            self.manager.load_scenario(scenario, self.agent_instance)
            self.statistics_manager.set_route(
                config.name, config.index, scenario.scenario
            )
            print("start to run scenario")
            self.manager.run_scenario()
            print("stop to run scanario")
            # Stop scenario
            self.manager.stop_scenario()
            # register statistics
            current_stats_record = self.statistics_manager.compute_route_statistics(
                config,
                self.manager.scenario_duration_system,
                self.manager.scenario_duration_game,
            )
            # save
            # modification

            self.statistics_manager.save_record(
                current_stats_record, config.index, self.save_path
            )

            # Remove all actors
            scenario.remove_all_actors()

            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except SensorConfigurationInvalid as e:
            self._cleanup(True)
            sys.exit(-1)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            print(e)

        self._cleanup()

    def run(self, args, customized_data):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(
            args.routes, args.scenarios, args.repetitions, args.background_vehicles
        )
        if args.resume:
            route_indexer.resume(self.save_path)
            self.statistics_manager.resume(self.save_path)
        else:
            self.statistics_manager.clear_record(self.save_path)
        while route_indexer.peek():
            # setup
            config = route_indexer.next()
            # run
            self._load_and_run_scenario(args, config, customized_data)
            self._cleanup(ego=True)

            route_indexer.save_state(self.save_path)
        # save global statistics
        # modification
        global_stats_record = self.statistics_manager.compute_global_statistics(
            route_indexer.total
        )
        StatisticsManager.save_global_record(
            global_stats_record, self.sensors, self.save_path
        )


def main():

    arguments = specify_args()
    arguments.debug = True
    statistics_manager = StatisticsManager()

    # Fixed Hyperparameters
    # if use_actors = False, no other actors will be generated
    use_actors = False
    using_customized_route_and_scenario = True
    multi_actors_scenarios = ["Scenario12"]
    arguments.scenarios = "leaderboard/data/fuzzing_scenarios.json"
    town_name = "Town03"
    scenario = "Scenario12"
    direction = "front"
    route = 0
    # sample_factor is an integer between [1, 5]
    sample_factor = 5
    # waypoints_num_limit: the maximum number of waypoints that we consider to perturb.
    waypoints_num_limit = 10
    # lane_width = 3.5
    max_num_of_vehicle = 2
    max_num_of_pedestrians = 2

    # Parameters to optimize
    # Set up environment parameters
    # real, [0, 1]
    friction = 0.1
    # integer, [0, 20]
    weather_index = 2

    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = weather_index
    os.environ["WEATHER_INDEX"] = str(weather_index)

    town_scenario_direction = town_name + "/" + scenario

    folders = [os.environ["SAVE_FOLDER"], town_name, scenario]
    if scenario in multi_actors_scenarios:
        town_scenario_direction += "/" + direction
        folders.append(direction)

    cur_folder_name = make_hierarchical_dir(folders)

    os.environ["SAVE_FOLDER"] = cur_folder_name
    arguments.save_folder = os.environ["SAVE_FOLDER"]

    route_prefix = (
        "leaderboard/data/customized_routes/" + town_scenario_direction + "/route_"
    )

    route_str = str(route)
    if route < 10:
        route_str = "0" + route_str
    arguments.routes = route_prefix + route_str + ".xml"
    os.environ["ROUTES"] = arguments.routes

    # extract waypoints along route
    import xml.etree.ElementTree as ET

    tree = ET.parse(arguments.routes)
    route_waypoints = []

    # this iteration should only go once since we only keep one route per file
    for route in tree.iter("route"):
        route_id = route.attrib["id"]
        route_town = route.attrib["town"]

        for waypoint in route.iter("waypoint"):
            route_waypoints.append(
                carla.Transform(
                    carla.Location(
                        x=float(waypoint.attrib["x"]),
                        y=float(waypoint.attrib["y"]),
                        z=float(waypoint.attrib["z"]),
                    ),
                    carla.Rotation(
                        float(waypoint.attrib["pitch"]),
                        float(waypoint.attrib["yaw"]),
                        float(waypoint.attrib["roll"]),
                    ),
                )
            )

    # extract waypoints for the scenario
    world_annotations = RouteParser.parse_annotations_file(arguments.scenarios)
    info = world_annotations[town_name][0]["available_event_configurations"][0]

    center = info["center"]
    RouteParser.convert_waypoint_float(center)
    center_location = carla.Location(
        float(center["x"]), float(center["y"]), float(center["z"])
    )
    center_rotation = carla.Rotation(float(center["pitch"]), float(center["yaw"]), 0.0)
    center_transform = carla.Transform(center_location, center_rotation)
    # --------------------------------------------------------------------------

    if use_actors:
        # Set up actors

        # ego car
        ego_car_waypoints_perturbation = []
        for i in range(waypoints_num_limit):
            dx = np.clip(np.random.normal(0, 2, 1)[0], -0.5, 0.5)
            dy = np.clip(np.random.normal(0, 2, 1)[0], -0.5, 0.5)
            ego_car_waypoints_perturbation.append((dx, dy))

        # static
        static_1_transform = center_transform
        static_1 = Static(
            model="static.prop.barrel", spawn_transform=static_1_transform
        )
        static_list = [static_1]

        # pedestrians
        pedestrian_1_transform = create_transform(
            route_waypoints[0].location.x - 2,
            route_waypoints[0].location.y - 8,
            0,
            0,
            route_waypoints[0].rotation.yaw,
            0,
        )
        pedestrian_1 = Pedestrian(
            model="walker.pedestrian.0001",
            spawn_transform=pedestrian_1_transform,
            trigger_distance=20,
            speed=1.5,
            dist_to_travel=6,
            after_trigger_behavior="stop",
        )
        pedestrian_list = [pedestrian_1]

        # vehicles
        waypoint_follower = True

        vehicle_1_transform = create_transform(
            route_waypoints[1].location.x,
            route_waypoints[1].location.y - 5,
            0,
            0,
            route_waypoints[1].rotation.yaw,
            0,
        )

        # if waypoint_follower == False
        vehicle_1_dist_to_travel = 5
        vehicle_1_target_direction = carla.Vector3D(x=0.2, y=1, z=0)

        # else
        targeted_waypoint = create_transform(
            route_waypoints[1].location.x,
            route_waypoints[1].location.y - 40,
            0,
            0,
            route_waypoints[1].rotation.yaw,
            0,
        )
        # targeted_waypoint = route_waypoints[-1]

        vehicle_1_waypoints_perturbation = []

        for i in range(waypoints_num_limit):
            dx = np.clip(np.random.normal(0, 2, 1)[0], -0.5, 0.5)
            dy = np.clip(np.random.normal(0, 2, 1)[0], -0.5, 0.5)
            vehicle_1_waypoints_perturbation.append((dx, dy))

        vehicle_1 = Vehicle(
            model="vehicle.audi.a2",
            spawn_transform=vehicle_1_transform,
            avoid_collision=True,
            initial_speed=0,
            trigger_distance=10,
            waypoint_follower=waypoint_follower,
            targeted_waypoint=targeted_waypoint,
            dist_to_travel=vehicle_1_dist_to_travel,
            target_direction=vehicle_1_target_direction,
            targeted_speed=10,
            after_trigger_behavior="stop",
            color="(0, 0, 0)",
            waypoints_perturbation=vehicle_1_waypoints_perturbation,
        )

        vehicle_list = [vehicle_1]
    else:
        static_list = []
        pedestrian_list = []
        vehicle_list = []
        ego_car_waypoints_perturbation = []

    customized_data = {
        "friction": friction,
        "static_list": static_list,
        "pedestrian_list": pedestrian_list,
        "vehicle_list": vehicle_list,
        "center_transform": center_transform,
        "using_customized_route_and_scenario": True,
        "destination": route_waypoints[-1].location,
        "sample_factor": sample_factor,
        "ego_car_waypoints_perturbation": ego_car_waypoints_perturbation,
        "port": arguments.port,
        "customized_center_transforms": {},
        "parameters_min_bounds": [],
        "parameters_max_bounds": [],
    }

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments, customized_data)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == "__main__":
    main()
