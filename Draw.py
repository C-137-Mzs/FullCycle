import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd

def data_process(data, length, num):
    data = data[:length]
    average_arr = []
    for i in range(0, len(data), num):
        chunk = data[i:i + num]
        average = sum(chunk) / len(chunk)
        average_arr.append(average)
    data = np.array(average_arr)
    iters = list(range(len(data) - num + 1))
    # 创建一个空的二维数组，大小为 (num, len(data)-num+1)
    rows, cols = num, len(data) - num + 1
    matrix = np.zeros((rows, cols))
    # 填充矩阵
    for i in range(rows):
        matrix[i, :] = data[i:i + cols]
    return np.array(matrix), iters

def draw_line(name_of_alg, datas, color, iters, attack):
    avg = np.mean(datas, axis=0)
    std = np.std(datas, axis=0)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  #上方差
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  #下方差
    if color == "#77AE43":
        plt.plot(iters, avg, color=color, label=attack, linewidth=4.5)
    else:
        plt.plot(iters, avg, color=color, label=attack, linewidth=2.5)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

def Draw(total, num, look, exp, color, attack):
    xlsx_file = pd.read_excel('xlsx_' + exp + '/'+look+'.xlsx', engine='openpyxl')
    data = xlsx_file.iloc[:total, 0].astype(float).tolist()
    matrix, iters = data_process(data, total, num)
    draw_line(exp, matrix, color, iters, attack)

def Draw_Fig(look,model):
    plt.style.use('seaborn-paper')
    plt.rcParams['font.family'] = 'Times New Roman'
    print(look,model)
    fig = plt.figure(figsize=(12, 9))
    total = 4000
    num = 14

    exp_list = ["no_20w_"+model,"rew_new_20w_"+model,"obs_20w_"+model,"act_20w_"+model,"oar_new_20w_"+model,]
    attack_list = ["No Attack","Reward Attack","Observation Attack","Action Attack","FullCycle Attack","No Attack"]
    color_list = ['#77AE43','#edb021','#1072BD','#d7592c','#7f318d',]

    plt.xlim(left=0, right=200)
    ylabel_value = look[0].upper()+look[1:]

    if ylabel_value == "Reward" :
        plt.ylim(bottom=0, top=60)
    elif ylabel_value == "Loss":
        plt.ylim(bottom=0.00, top=0.06)
    else:
        plt.ylim(bottom=22, top=32)

    if model=="dqn":
        title = ylabel_value + " Trend on " + "DQN"
    elif model=="ddqn":
        title = ylabel_value + " Trend on " + "DDQN"
    else:
        title = ylabel_value + " Trend on " + "DuDQN"

    legend = 0
    if look == "loss" and model == "dqn":
        legend = 1
    xlabel = 0
    if look == "speed":
        xlabel = 1
    ylabel = 0
    if model == "dqn":
        ylabel = 1

    for exp, color, attack in zip(exp_list, color_list, attack_list):
        Draw(total, num, look, exp, color, attack)
        print(exp, color, attack)

    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')

    if ylabel==1:
        plt.ylabel(ylabel_value, fontsize=36,fontweight='bold')
    if xlabel==1:
        plt.xlabel('Episodes(×20)', fontweight='bold', fontsize=36)
    if look=="loss" and legend==1:
        legend = plt.legend(loc='upper right', fontsize=32)
        # 加粗图例和边框
        for text in legend.get_texts():
            text.set_fontweight('bold')
        legend.get_frame().set_linewidth(2.0)

    #plt.title(title, fontweight='bold', fontsize=34)
    plt.grid(color='lightgray', alpha=1, linewidth=2)

    # 加粗边框
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.0)  # 设置边框宽度为2.0

    plt.tight_layout(rect=[0.00, 0.01, 1.00, 0.99])  # [left, bottom, right, top]

    if look=="reward":
        if model == "dqn":
            plt.savefig("1_1.png")
        elif model == "ddqn":
            plt.savefig("1_2.png")
        else:
            plt.savefig("1_3.png")
    elif look=="loss":
        if model == "dqn":
            plt.savefig("2_1.png")
        elif model == "ddqn":
            plt.savefig("2_2.png")
        else:
            plt.savefig("2_3.png")
    else :
        if model == "dqn":
            plt.savefig("3_1.png")
        elif model == "ddqn":
            plt.savefig("3_2.png")
        else:
            plt.savefig("3_3.png")

look = ["reward","loss","speed"]
algorithm = ["dqn","ddqn","dudqn"]

for l in look:
    for a in algorithm:
        Draw_Fig(l, a)
