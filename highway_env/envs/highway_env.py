import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class HighwayEnv(AbstractEnv):
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
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
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

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    # def _reward(self, action: Action) -> float:   # 源代码
    #     """
    #     The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    #     :param action: the last action performed
    #     :return: the corresponding reward
    #     """
    #     neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #         else self.vehicle.lane_index[2]
    #     # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
    #     forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
    #     scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
    #     reward = \
    #         + self.config["collision_reward"] * self.vehicle.crashed \
    #         + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
    #         + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
    #     reward = utils.lmap(reward,
    #                       [self.config["collision_reward"],
    #                        self.config["high_speed_reward"] + self.config["right_lane_reward"]],
    #                       [0, 1])
    #     reward = 0 if not self.vehicle.on_road else reward
    #     return reward
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


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)
