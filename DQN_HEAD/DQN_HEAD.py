import gym
env = gym.make('highway-v0').unwrapped

import argparse

import warnings
warnings.filterwarnings("ignore")

def get_config():
    parser = argparse.ArgumentParser(description='SAC')
# =================  NET  ===============================================================
    parser.add_argument('--e', default=0.1,   type=float)
    parser.add_argument('--LR', default=0.001, type=float)
    parser.add_argument('--GAMMA', default=0.9,   type=float)
    parser.add_argument('--input_net', default=20,    type=float)
    parser.add_argument('--neure', default=1024,  type=int)
    parser.add_argument('--output_net', default=5,     type=float)

###################   Basic   ##########################
    parser.add_argument('--right_lane_reward', default=0.1, type=float)
    parser.add_argument('--lanes_count', default=5, type=float)
    parser.add_argument('--show_trajectories', default=False, type=float)
    parser.add_argument('--screen_height', default=300, type=float)
    parser.add_argument('--screen_width', default=1600, type=float)
    parser.add_argument('--policy_frequency', default=1, type=float)  # abstract： frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])

    parser.add_argument('--reward_speed_range', default=[20, 32], type=float)   # 动态时用
    ######    在 Vehicle.controller: class MDPVehicle(ControlledVehicle): DEFAULT_TARGET_SPEEDS = np.linspace(16, 32, 9)
    parser.add_argument('--punish_speed_range', default=[16, 22], type=float)

    # parser.add_argument('--reward_speed_range', default=[3, 9], type=float)  # SV静止时用
    # #######    DEFAULT_TARGET_SPEEDS = np.linspace(3,9,7)

#############  DQN 调参  ###########################
    parser.add_argument('--simulation_frequency', default=10, type=float)  # 决定dt=1/"simulation_frequency"；即：一次step()，系统做出几次控制循环。还和 controller 比例控制 TAU_ACC = 2 有关，一并调整
    parser.add_argument('--vehicles_count', default=19, type=float)
    parser.add_argument('--duration', default=120, type=float)     # 40 步数（和环境交互次数） 决定了在不 crashed 的前提下，能有多少个 a 和 env.step(a)。直到steps计数到["duration"]，触发done，系统复位。
    parser.add_argument('--collision_reward', default=-1, type=float)
    parser.add_argument('--high_speed_reward', default=0.5, type=float)
    parser.add_argument('--punish_speed_reward', default=-2, type=float)
    parser.add_argument('--wrong_act_reward', default=-2, type=float)
    parser.add_argument('--correct_act_reward', default=0.5, type=float)

    parser.add_argument('--T_N_update_FQ', default=60, type=float)   # 60
    parser.add_argument('--BATCH_SIZE', default=20, type=float)      # 20
    parser.add_argument('--MEMORY_CAP', default=200, type=float)     # 200,
    parser.add_argument('--total_train_t', default=50, type=float)   # 20*200*25 = 10W

# 全是0========================================================================================
    parser.add_argument('--max_reward', default=0, type=float)
    parser.add_argument('--speed_array', default=[], type=float)
    parser.add_argument('--avg_speed_episode', default=0, type=float)
    parser.add_argument('--avg_speed_episode_array', default=[], type=float)

    parser.add_argument('--reward_array', default=[], type=float)
    parser.add_argument('--sum_reward_episode', default=0, type=float)
    parser.add_argument('--sum_reward_episode_array', default=[], type=float)
    parser.add_argument('--reward_sum_in_train', default=0, type=float)

    parser.add_argument('--sum_RA_episode', default=0, type=float)
    parser.add_argument('--sum_RA_episode_array', default=[], type=float)

    parser.add_argument('--loss_array', default=[], type=float)
    parser.add_argument('--avg_loss_episode', default=0, type=float)
    parser.add_argument('--avg_loss_episode_array', default=[], type=float)

    parser.add_argument('--WRO_ACT_TR', default=0, type=float)
    parser.add_argument('--CORRECT_ACT_TR', default=0, type=float)

    parser.add_argument('--new_reward', default=0, type=float)
    parser.add_argument('--reward', default=0, type=float)

    parser.add_argument('--S_tm_ep', default=0, type=float)
    parser.add_argument('--E_tm_ep', default=0, type=float)
    parser.add_argument('--tm_ep', default=0, type=float)
    parser.add_argument('--tm_ep_array', default=[], type=float)

    parser.add_argument('--correct_act_cnt_in_train', default=0, type=float)
    parser.add_argument('--total_step_times', default=0, type=float)
    parser.add_argument('--trained_times', default=0, type=float)
    parser.add_argument('--WRONG_ACT_CNT_in_train', default=0, type=float)
    parser.add_argument('--lane_old', default=0, type=float)
    parser.add_argument('--start_train_TR', default=0, type=float)
    parser.add_argument('--total_time', default=0, type=float)
    parser.add_argument('--memory_cnt', default=0, type=float)
    parser.add_argument('--done_in_train', default=0, type=float)
    parser.add_argument('--episode', default=0, type=float)
    parser.add_argument('--success_done', default=0, type=float)
    parser.add_argument('--First_finish_time', default=0, type=float)

    parser.add_argument('--nx_cord', default=0, type=float)
    parser.add_argument('--nloss', default=0, type=float)
    parser.add_argument('--nspeed', default=0, type=float)
    parser.add_argument('--nreward', default=0, type=float)
    parser.add_argument('--nRA', default=0, type=float)
    parser.add_argument('--ntm', default=0, type=float)
    return parser
