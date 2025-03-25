import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

import copy


class HighwayEnvActionPerturbation(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            'screen_width': 1200,
            'screen_height': 600,
            "initial_lane_id": None,
            "duration": 80,  # [s]
            "ego_spacing": 2, #
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=20),   # 旁车限速
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=24,    # 旁车主车初速
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicle_ap_ref = Vehicle.create_random(
            self.road,
            lane_id=2,
        )
        x, y = vehicle_ap_ref.position
        # 最开始的白车
        for i in range(0,4):
            vehicle_action_perturbation = Vehicle.create_from(vehicle_ap_ref)
            vehicle_action_perturbation.position = [x + 100, y + 4*(i-2)]

            vehicle_action_perturbation.heading = 0
            vehicle_action_perturbation.speed = 0
            vehicle_action_perturbation = other_vehicles_type(self.road, vehicle_action_perturbation.position, vehicle_action_perturbation.heading,
                                               vehicle_action_perturbation.speed)
            vehicle_action_perturbation.MAX_SPEED = 0
            vehicle_action_perturbation.MIN_SPEED = 0
            self.stop_vehicle_id = id(vehicle_action_perturbation) % 1000
            vehicle_action_perturbation.color = (255, 255, 255)
            self.road.vehicles.append(vehicle_action_perturbation)

        for _ in range(25):
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # 中间的红车
        for i in range(0, 2):
            vehicle_action_perturbation = Vehicle.create_from(vehicle_ap_ref)
            vehicle_action_perturbation.position = [x + 1000 + 5*i, y + 4 * 2]
            vehicle_action_perturbation.heading = 0
            vehicle_action_perturbation.speed = 0
            vehicle_action_perturbation = other_vehicles_type(self.road, vehicle_action_perturbation.position,
                                                              vehicle_action_perturbation.heading,
                                                              vehicle_action_perturbation.speed)
            vehicle_action_perturbation.MAX_SPEED = 0
            vehicle_action_perturbation.MIN_SPEED = 0
            self.stop_vehicle_id = id(vehicle_action_perturbation) % 1000
            vehicle_action_perturbation.color = (255, 0, 0)
            self.road.vehicles.append(vehicle_action_perturbation)

        # for _ in range(25):
        #     vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)

        # 最后的黑车
        for i in range(0,4):
            vehicle_action_perturbation = Vehicle.create_from(vehicle_ap_ref)
            vehicle_action_perturbation.position = [x + 2000, y + 4*(i-2)]
            vehicle_action_perturbation.heading = 0
            vehicle_action_perturbation.speed = 0
            vehicle_action_perturbation = other_vehicles_type(self.road, vehicle_action_perturbation.position, vehicle_action_perturbation.heading,
                                               vehicle_action_perturbation.speed)
            vehicle_action_perturbation.MAX_SPEED = 0
            vehicle_action_perturbation.MIN_SPEED = 0
            self.stop_vehicle_id = id(vehicle_action_perturbation) % 1000
            vehicle_action_perturbation.color = (0, 0, 0)
            self.road.vehicles.append(vehicle_action_perturbation)

    def _reward(self, action: Action) -> float:   #   自己修改的
        """  修改后程序  """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])  # "reward_speed_range"= [20,30];
        #  scaled_speed = 0+((forward_speed-20)/(30-20))*(1-0), 有可能为负值

        # reward = self.config["collision_reward"] * self.vehicle.crashed \
        #        + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)  # 原始奖励函数值
        reward = self.config["collision_reward"] * self.vehicle.crashed   # 原始奖励函数值 只计算碰撞奖励
        ''' clip 的用法：把 scaled_speed 截取 0,1 之间的数值。 则 np.clip(scaled_speed, 0, 1) 只保留 scaled_speed 大于 0 的值，也就是超过 20 才有奖励，此时 reward = [-1,0]
        '''
        # reward = utils.lmap(reward, [ self.config["collision_reward"],self.config["high_speed_reward"] ],[0, 1] )
        '''   lmap 的用法 :  def lmap(  v: float,  x: Interval,  y: Interval) -> float:  """Linear map of value v with range x to desired range y."""
             return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) = (reward - (-1)) / (1 - (-1))
        '''
        reward = 0 if not self.vehicle.on_road else reward
        return reward,lane

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

register(
    id='highway-action-perturbation-v0',
    entry_point='highway_env.envs:HighwayEnvActionPerturbation',
)
