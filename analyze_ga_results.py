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




import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from customized_utils import  get_distinct_data_points, check_bug, filter_critical_regions, get_sorted_subfolders, load_data, get_picklename, is_distinct_vectorized
from ga_fuzzing import default_objectives

def draw_hv(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)
    hv = res['hv']
    n_evals = res['n_evals'].tolist()

    # hv = [0] + hv
    # n_evals = [0] + n_evals


    # visualze the convergence curve
    plt.plot(n_evals, hv, '-o')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(os.path.join(save_folder, 'hv_across_generations'))
    plt.close()



def draw_performance(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    time_bug_num_list = res['time_bug_num_list']

    t_list = []
    n_list = []
    for t, n in time_bug_num_list:
        t_list.append(t)
        n_list.append(n)
    print(t_list)
    print(n_list)
    plt.plot(t_list, n_list, '-o')
    plt.title("Time VS Number of Bugs")
    plt.xlabel("Time")
    plt.ylabel("Number of Bugs")
    plt.savefig(os.path.join(save_folder, 'bug_num_across_time'))
    plt.close()


def analyze_causes(folder, save_folder, total_num, pop_size):



    avg_f = [0 for _ in range(int(total_num // pop_size))]

    causes_list = []
    counter = 0
    for sub_folder_name in os.listdir(folder):
        sub_folder = os.path.join(folder, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]

                    ego_linear_speed = float(bug['ego_linear_speed'])
                    causes_list.append((sub_folder_name, ego_linear_speed, bug['offroad_dist'], bug['is_wrong_lane'], bug['is_run_red_light'], bug['status'], bug['loc'], bug['object_type']))

                    ind = int(int(sub_folder_name) // pop_size)
                    avg_f[ind] += (ego_linear_speed / pop_size)*-1

    causes_list = sorted(causes_list, key=lambda t: int(t[0]))
    for c in causes_list:
        print(c)
    print(avg_f)

    plt.plot(np.arange(len(avg_f)), avg_f)
    plt.title("average objective value across generations")
    plt.xlabel("Generations")
    plt.ylabel("average objective value")
    plt.savefig(os.path.join(save_folder, 'f_across_generations'))

    plt.close()

def show_gen_f(bug_res_path):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    val = res['val']
    plt.plot(np.arange(len(val)), val)
    plt.show()

def plot_each_bug_num_and_objective_num_over_generations(generation_data_paths):
    # X=X, y=y, F=F, objectives=objectives, time=time_list, bug_num=bug_num_list, labels=labels, hv=hv
    pop_size = 100
    data_list = []
    for generation_data_path in generation_data_paths:
        data = []
        with open(generation_data_path[1], 'r') as f_in:
            for line in f_in:
                tokens = line.split(',')
                if len(tokens) == 2:
                    pass
                else:
                    tokens = [float(x.strip('\n')) for x in line.split(',')]
                    num, has_run, time, bugs, collisions, offroad_num, wronglane_num, speed, min_d, offroad, wronglane, dev = tokens[:12]
                    out_of_road = offroad_num + wronglane_num
                    data.append(np.array([num/pop_size, has_run, time, bugs, collisions, offroad_num, wronglane_num, out_of_road, speed, min_d, offroad, wronglane, dev]))

        data = np.stack(data)
        data_list.append(data)

    labels = [generation_data_paths[i][0] for i in range(len(data_list))]
    data = np.concatenate([data_list[1], data_list[2]], axis=0)

    for i in range(len(data_list[1]), len(data_list[1])+len(data_list[2])):
        data[i] += data_list[1][-1]
    data_list.append(data)

    labels.append('collision+out-of-road')

    fig = plt.figure(figsize=(15, 9))


    plt.suptitle("values over time", fontsize=14)


    info = [(1, 3, 'Bug Numbers'), (6, 4, 'Collision Numbers'), (7, 5, 'Offroad Numbers'), (8, 6, 'Wronglane Numbers'), (9, 7, 'Out-of-road Numbers'), (11, 8, 'Collision Speed'), (12, 9, 'Min object distance'), (13, 10, 'Offroad Directed Distance'), (14, 11, 'Wronglane Directed Distance'), (15, 12, 'Max Deviation')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(3, 5, loc)
        for i in [0, 3, 1, 2]:
            if loc < 11 or i < 3:
                label = labels[i]
                if loc >= 11:
                    y = []
                    for j in range(data_list[i].shape[0]):
                        y.append(np.mean(data_list[i][:j+1, ind]))
                else:
                    y = data_list[i][:, ind]
                ax.plot(data_list[i][:, 0], y, label=label)
        if loc == 1:
            ax.legend()
        plt.xlabel("Generations")
        plt.ylabel(ylabel)
    plt.savefig('bug_num_and_objective_num_over_generations')







# list bug types and their run numbers
def list_bug_categories_with_numbers(folder_path):
    l = []
    for sub_folder_name in os.listdir(folder_path):
        sub_folder = os.path.join(folder_path, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]
                    if bug['ego_linear_speed'] > 0:
                        cause_str = 'collision'
                    elif bug['is_offroad']:
                        cause_str = 'offroad'
                    elif bug['is_wrong_lane']:
                        cause_str = 'wronglane'
                    else:
                        cause_str = 'unknown'
                    l.append((sub_folder_name, cause_str))


    for n,s in sorted(l, key=lambda t: int(t[0])):
        print(n,s)



# list pickled data
def analyze_data(pickle_path):
    with open(pickle_path, 'rb') as f_out:
        d = pickle.load(f_out)
        X = d['X']
        y = d['y']
        F = d['F']
        objectives = d['objectives']
        print(np.sum(X[10,:]-X[11,:]))
        filter_critical_regions(X, y)
        # TBD: tree diversity



def unique_bug_num(all_X, all_y, mask, xl, xu, cutoff):
    if cutoff == 0:
        return 0, []
    X = all_X[:cutoff]
    y = all_y[:cutoff]

    bug_inds = np.where(y>0)
    bugs = X[bug_inds]


    p = 0
    c = 0.15
    th = int(len(mask)*0.5)

    # TBD: count different bugs separately
    filtered_bugs, unique_bug_inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)


    print(cutoff, len(filtered_bugs), len(bugs))
    return len(filtered_bugs), np.array(unique_bug_inds), bug_inds

# plot two tsne plots for bugs VS normal and data points across generations
def apply_tsne(path, n_gen, pop_size):
    d = np.load(path, allow_pickle=True)
    X = d['X']
    y = d['y']
    mask = d['mask']
    xl = d['xl']
    xu = d['xu']


    cutoff = n_gen * pop_size
    _, unique_bug_inds, bug_inds = unique_bug_num(X, y, mask, xl, xu, cutoff)

    y[bug_inds] = 1
    y[unique_bug_inds] = 2


    generations = []
    for i in range(n_gen):
        generations += [i for _ in range(pop_size)]


    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(figsize=(9, 9))
    plt.suptitle("tSNE of bugs and unique bugs", fontsize=14)
    ax = fig.add_subplot(111)
    scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    plt.title("bugs VS normal")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])


    # fig = plt.figure(figsize=(18, 9))
    #
    # plt.suptitle("tSNE of sampled/generated data points", fontsize=14)
    #
    #
    # ax = fig.add_subplot(121)
    # scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    # plt.title("bugs VS normal")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    # plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])
    #
    # ax = fig.add_subplot(122)
    # scatter_gen = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=generations, cmap=plt.cm.rainbow)
    # plt.title("different generations")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    # plt.legend(handles=scatter_gen.legend_elements()[0], labels=[str(i) for i in range(n_gen)])

    plt.savefig('tsne')




def get_bug_num(cutoff, X, y, mask, xl, xu, p=0, c=0.15, th=0.5):

    if cutoff == 0:
        return 0, 0, 0
    p = p
    c = c
    th = int(len(mask)*th)

    def process_specific_bug(bug_ind):
        chosen_bugs = y == bug_ind
        specific_bugs = X[chosen_bugs]
        unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(specific_bugs, mask, xl, xu, p, c, th)

        return len(unique_specific_bugs)

    unique_collision_num = process_specific_bug(1)
    unique_offroad_num = process_specific_bug(2)
    unique_wronglane_num = process_specific_bug(3)
    unique_redlight_num = process_specific_bug(3)

    return unique_collision_num, unique_offroad_num, unique_wronglane_num, unique_redlight_num


def unique_bug_num_seq_partial_objectives(path_list):


    all_X_list = []
    all_y_list = []

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)

        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_X = d['X'][inds]
        all_y = d['y'][inds]
        objectives = objectives[inds]

        all_X_list.append(all_X)
        all_y_list.append(all_y)

    all_X = np.concatenate(all_X_list)
    all_y = np.concatenate(all_y_list)

    collision_num, offroad_num, wronglane_num, redlight_num = get_bug_num(700, all_X, all_y, mask, xl, xu)
    print(collision_num, offroad_num, wronglane_num, redlight_num)


def analyze_objectives(path_list, filename='objectives_bug_num_over_simulations', scene_name=''):




    cutoffs = [100*i for i in range(0, 8)]
    data_list = []
    labels = []

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        labels.append(label)

        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_X = d['X'][inds]
        all_y = d['y'][inds]
        objectives = objectives[inds]


        data = []
        for cutoff in cutoffs:
            X = all_X[:cutoff]
            y = all_y[:cutoff]
            collision_num, offroad_num, wronglane_num, redlight_num = get_bug_num(cutoff, X, y, mask, xl, xu)

            if cutoff == 1400:
                print(collision_num, offroad_num, wronglane_num, redlight_num)

            speed = np.mean(objectives[:cutoff, 0])
            min_d = np.mean(objectives[:cutoff, 1])
            offroad = np.mean(objectives[:cutoff, 2])
            wronglane = np.mean(objectives[:cutoff, 3])
            dev = np.mean(objectives[:cutoff, 4])

            bug_num = collision_num+offroad_num+wronglane_num
            out_of_road_num = offroad_num+wronglane_num
            data.append(np.array([bug_num, collision_num, offroad_num, wronglane_num, out_of_road_num, speed, min_d, offroad, wronglane, dev]))

        data = np.stack(data)
        data_list.append(data)




    fig = plt.figure(figsize=(12.5, 5))

    # fig = plt.figure(figsize=(15, 9))
    # plt.suptitle("values over simulations", fontsize=14)


    # info = [(1, 0, 'Bug Numbers'), (6, 1, 'Collision Numbers'), (7, 2, 'Offroad Numbers'), (8, 3, 'Wronglane Numbers'), (9, 4, 'Out-of-road Numbers'), (11, 5, 'Collision Speed'), (12, 6, 'Min object distance'), (13, 7, 'Offroad Directed Distance'), (14, 8, 'Wronglane Directed Distance'), (15, 9, 'Max Deviation')]

    info = [(1, 1, '# unique collision'), (2, 4, '# unique out-of-road')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(1, 2, loc)
        for i in range(len(data_list)):
            label = labels[i]
            y = data_list[i][:, ind]
            ax.plot(cutoffs, y, label=label, linewidth=2, marker='o', markersize=10)
        if loc == 1:
            ax.legend(loc=2, prop={'size': 26}, fancybox=True, framealpha=0.2)

            # import pylab
            # fig_p = pylab.figure()
            # figlegend = pylab.figure(figsize=(3,2))
            # ax = fig_p.add_subplot(111)
            # lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))
            # figlegend.legend(lines, ('collision-', 'two'), 'center')
            # fig.show()
            # figlegend.show()
            # figlegend.savefig('legend.png')

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.set_xlabel("# simulations", fontsize=26)
        ax.set_ylabel(ylabel, fontsize=26)

    fig.suptitle(scene_name, fontsize=38)

    fig.tight_layout()



    plt.savefig(filename)


def ablate_thresholds(path_list, thresholds_list, cutoff):

    p = 0

    xl = None
    xu = None
    mask = None

    for c, th in thresholds_list:
        print('(', c, th, ')')
        for i, (label, pth) in enumerate(path_list):
            print(label)
            d = np.load(pth, allow_pickle=True)

            if i == 0:
                xl = d['xl']
                xu = d['xu']
                mask = d['mask']
            objectives = np.stack(d['objectives'])
            df_objectives = np.array(default_objectives)

            eps = 1e-7
            diff = np.sum(objectives - df_objectives, axis=1)


            inds = np.abs(diff) > eps

            all_X = d['X'][inds]
            all_y = d['y'][inds]
            objectives = objectives[inds]


            X = all_X[:cutoff]
            y = all_y[:cutoff]
            collision_num, offroad_num, wronglane_num, redlight_num = get_bug_num(cutoff, X, y, mask, xl, xu, p=p, c=c, th=th)

            print(collision_num+offroad_num+wronglane_num, collision_num, offroad_num, wronglane_num)


def check_unique_bug_num(folder, path1, path2):
    d = np.load(folder+'/'+path1, allow_pickle=True)
    xl = d['xl']
    xu = d['xu']
    mask = d['mask']

    d = np.load(folder+'/'+path2, allow_pickle=True)
    all_X = d['X']
    all_y = d['y']
    cutoffs = [100*i for i in range(0, 15)]


    def subroutine(cutoff):
        if cutoff == 0:
            return 0, []
        X = all_X[:cutoff]
        y = all_y[:cutoff]

        bugs = X[y>0]


        p = 0
        c = 0.15
        th = 0.5

        filtered_bugs, inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)
        print(cutoff, len(filtered_bugs), len(bugs))
        return len(filtered_bugs), inds


    num_of_unique_bugs = []
    for cutoff in cutoffs:
        num, inds = subroutine(cutoff)
        num_of_unique_bugs.append(num)
    print(inds)
    # print(bug_counters)
    # counter_inds = np.array(bug_counters)[inds] - 1
    # print(all_X[counter_inds[-2]])
    # print(all_X[counter_inds[-1]])

    plt.plot(cutoffs, num_of_unique_bugs, marker='o', markersize=10)
    plt.xlabel('# simulations')
    plt.ylabel('# unique violations')
    plt.savefig('num_of_unique_bugs')




def draw_hv_and_gd(path_list):
    from pymoo.factory import get_performance_indicator
    def is_pareto_efficient_dumb(costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
        return is_efficient

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        X = d['X']
        y = d['y']
        objectives = d['objectives'][:, :5] * np.array([-1, 1, 1, 1, -1])

        pareto_set = objectives[is_pareto_efficient_dumb(objectives)]
        # print(label, np.sum(is_pareto_efficient_dumb(objectives)))




        gd = get_performance_indicator("gd", pareto_set)
        hv = get_performance_indicator("hv", ref_point=np.array([0.01, 7.01, 7.01, 7.01, 0.01]))

        print(label)
        for j in range(16):
            cur_objectives = objectives[:(j+1)*100]
            print(j)
            print("GD", gd.calc(cur_objectives))
            print("hv", hv.calc(cur_objectives))




def calculate_pairwise_dist(path_list):
    xl = None
    xu = None
    mask = None
    for i, (label, pth) in enumerate(path_list):
        print(label)
        d = np.load(pth, allow_pickle=True)
        if i == 0:
            xl = d['xl']
            xu = d['xu']
            mask = d['mask']
            print(len(mask))

        p = 0
        c = 0.15
        th = 0.5

        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_y = d['y'][inds][:1500]
        all_X = d['X'][inds][:1500]


        # all_X, inds = get_distinct_data_points(all_X, mask, xl, xu, p, c, th)
        # all_y = all_y[inds]

        int_inds = mask == 'int'
        real_inds = mask == 'real'
        eps = 1e-8




        def pair_dist(x_1, x_2):
            int_diff_raw = np.abs(x_1[int_inds] - x_2[int_inds])
            int_diff = np.ones(int_diff_raw.shape) * (int_diff_raw > eps)

            real_diff_raw = np.abs(x_1[real_inds] - x_2[real_inds]) / (np.abs(xu[real_inds] - xl[real_inds]) + eps)

            real_diff = np.ones(real_diff_raw.shape) * (real_diff_raw > c)

            diff = np.concatenate([int_diff, real_diff])

            diff_norm = np.linalg.norm(diff, p)
            # print(diff, diff_norm)
            return diff_norm



        dist_list = []
        for i in range(len(all_X)-1):
            for j in range(i+1, len(all_X)):
                if check_bug(objectives[i]) > 0 and check_bug(objectives[j]) > 0:
                    diff = pair_dist(all_X[i], all_X[j])
                    if diff:
                        dist_list.append(diff)


        dist = np.array(dist_list) / len(mask)
        print(np.mean(dist), np.std(dist))




def draw_unique_bug_num_over_simulations(path_list, warmup_pth_list, warmup_pth_cutoff, save_filename='num_of_unique_bugs', scene_name='', legend=True, range_upper_bound=6, bug_type='collision', unique_coeffs=[[]]):
    def subroutine(prev_X, cur_X, prev_objectives, cur_objectives, cutoff):
        if cutoff == 0:
            return 0

        cutoff_start = 0
        cutoff = np.min([cutoff, len(cur_X)])
        if label == 'ga-un':
            cutoff_start += warmup_pth_cutoff
            cutoff += warmup_pth_cutoff

        bug_num = 0
        if bug_type == 'collision':
            prev_inds = prev_objectives[:, 0] > 0.1
            cur_inds = cur_objectives[cutoff_start:cutoff, 0] > 0.1
            prev_X_bug = prev_X[prev_inds]
            cur_X_bug = cur_X[cutoff_start:cutoff][cur_inds]
            inds = is_distinct_vectorized(cur_X_bug, prev_X_bug, mask, xl, xu, p, c, th, verbose=False)
            bug_num += len(inds)
            print('all bug num: ', len(cur_X_bug))

        elif bug_type == 'out-of-road':
            prev_inds = prev_objectives[:, -3] == 1
            cur_inds = cur_objectives[cutoff_start:cutoff, -3] == 1
            prev_X_bug = prev_X[prev_inds]
            cur_X_bug_1 = cur_X[cutoff_start:cutoff][cur_inds]
            inds = is_distinct_vectorized(cur_X_bug_1, prev_X_bug, mask, xl, xu, p, c, th, verbose=False)
            bug_num += len(inds)

            prev_inds = prev_objectives[:, -2] == 1
            cur_inds = cur_objectives[cutoff_start:cutoff, -2] == 1
            prev_X_bug = prev_X[prev_inds]
            cur_X_bug_2 = cur_X[cutoff_start:cutoff][cur_inds]

            inds = is_distinct_vectorized(cur_X_bug_2, prev_X_bug, mask, xl, xu, p, c, th, verbose=False)
            bug_num += len(inds)

            print('all bug num: ', len(cur_X_bug_1)+len(cur_X_bug_2))
        elif bug_type == 'all':
            prev_inds = prev_objectives[:, 0] > 0.1
            cur_inds = cur_objectives[cutoff_start:cutoff, 0] > 0.1
            prev_X_bug = prev_X[prev_inds]
            cur_X_bug_1 = cur_X[cutoff_start:cutoff][cur_inds]
            inds = is_distinct_vectorized(cur_X_bug_1, prev_X_bug, mask, xl, xu, p, c, th, verbose=False)
            bug_num += len(inds)

            prev_inds = prev_objectives[:, -3] == 1
            cur_inds = cur_objectives[cutoff_start:cutoff, -3] == 1
            prev_X_bug = prev_X[prev_inds]
            cur_X_bug_2 = cur_X[cutoff_start:cutoff][cur_inds]
            inds = is_distinct_vectorized(cur_X_bug_2, prev_X_bug, mask, xl, xu, p, c, th, verbose=False)
            bug_num += len(inds)

            prev_inds = prev_objectives[:, -2] == 1
            cur_inds = cur_objectives[cutoff_start:cutoff, -2] == 1
            prev_X_bug = prev_X[prev_inds]
            cur_X_bug_3 = cur_X[cutoff_start:cutoff][cur_inds]

            inds = is_distinct_vectorized(cur_X_bug_3, prev_X_bug, mask, xl, xu, p, c, th, verbose=False)
            bug_num += len(inds)

            # prev_inds = prev_objectives[:, -1] == 1
            # cur_inds = cur_objectives[cutoff_start:cutoff, -1] == 1
            # prev_X_bug = prev_X[prev_inds]
            # cur_X_bug_4 = cur_X[cutoff_start:cutoff][cur_inds]
            #
            # inds = is_distinct_vectorized(cur_X_bug_4, prev_X_bug, mask, xl, xu, p, c, th, verbose=False)
            # bug_num += len(inds)

            print('collision:', len(cur_X_bug_1))
            print('wronglane:', len(cur_X_bug_2))
            print('off-road:', len(cur_X_bug_3))
            # print('red-light:', len(cur_X_bug_4))
            print('all bug num:', len(cur_X_bug_1)+len(cur_X_bug_2)+len(cur_X_bug_3))



        print(cutoff, bug_num)
        # print(inds)
        return bug_num


    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    line_style = ['-', ':', '--', '-.', '-', ':', '--', '-.']

    p = 0

    cutoffs = [50*i for i in range(0, range_upper_bound)]




    for i, (label, pth_list) in enumerate(path_list):
        if len(unique_coeffs) == 0:
            c = 0.1
            th = 0.5
        else:
            c, th = unique_coeffs[i]

        if len(warmup_pth_list) == 1:
            warmup_pth = warmup_pth_list[0]
        else:
            warmup_pth = warmup_pth_list[i]

        subfolders = get_sorted_subfolders(warmup_pth)
        prev_X, _, prev_objectives, _, _ = load_data(subfolders)
        prev_X = np.array(prev_X)[:warmup_pth_cutoff]
        prev_objectives = prev_objectives[:warmup_pth_cutoff]


        pickle_filename = get_picklename(warmup_pth)
        with open(pickle_filename, 'rb') as f_in:
            d = pickle.load(f_in)
            xl = d['xl']
            xu = d['xu']
            mask = d['mask']


        print('-'*30, label, '-'*30)
        print('prev_X.shape', prev_X.shape)
        num_of_unique_bugs_list = []
        for pth in pth_list:
            if 'dt' in label:
                cur_X = []
                cur_objectives = []
                for filename in os.listdir(pth):
                    filepath = os.path.join(pth, filename)
                    if os.path.isdir(filepath):
                        subfolders = get_sorted_subfolders(filepath)
                        tmp_X, _, tmp_objectives, _, _ = load_data(subfolders)
                        if len(tmp_X) > 0:
                            cur_X.append(tmp_X)
                            cur_objectives.append(tmp_objectives)
                cur_X = np.concatenate(cur_X)
                cur_objectives = np.concatenate(cur_objectives)
            else:
                subfolders = get_sorted_subfolders(pth)
                cur_X, _, cur_objectives, _, _ = load_data(subfolders)
                cur_X = np.array(cur_X)




            num_of_unique_bugs = []
            for cutoff in cutoffs:
                num = subroutine(prev_X, cur_X, prev_objectives, cur_objectives, cutoff)
                num_of_unique_bugs.append(num)

            num_of_unique_bugs_list.append(num_of_unique_bugs)
        num_of_unique_bugs_list = np.array(num_of_unique_bugs_list)
        # print(num_of_unique_bugs_list.shape)

        if len(pth_list) == 1:
            axes.plot(cutoffs, num_of_unique_bugs_list.squeeze(), label=label, linewidth=2, linestyle=line_style[i], markersize=5, marker='.')
        else:
            num_of_unique_bugs_std = np.std(num_of_unique_bugs_list, axis=0)
            num_of_unique_bugs_mean = np.mean(num_of_unique_bugs_list, axis=0)

            axes.errorbar(cutoffs, num_of_unique_bugs, yerr=num_of_unique_bugs_std, label=label, linewidth=2, linestyle=line_style[i], capsize=5)



    axes.set_title(scene_name, fontsize=26)
    if legend:
        axes.legend(loc=2, prop={'size': 18}, fancybox=True, framealpha=0.2)
    axes.set_xlabel('# simulations', fontsize=26)
    axes.set_ylabel('# unique violations', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout()
    fig.savefig(save_filename)

if __name__ == '__main__':
    town07_path_list = [
    ('ga-un-nn-grad', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_17_22_40_12,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1900_500nsga2initial/2021_02_20_12_57_59,50_80_adv_nn_1900_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_500nsga2initial/2021_02_24_09_41_57,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm-un-a', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_19_00_23_01,50_20_regression_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1900_500nsga2initial/2021_02_20_11_27_47,50_80_regression_nn_1900_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_500nsga2initial/2021_02_24_09_41_29,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm', ['run_results/nsga2/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_500nsga2initial_1000/2021_02_18_21_53_03,50_20_regression_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 'run_results/nsga2/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1900_500nsga2initial/2021_02_20_11_27_54,50_80_regression_nn_1900_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 'run_results/nsga2/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_500nsga2initial_700/2021_02_24_09_41_38,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01']),
    # ('nsga2-dt', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_19_17_56_00', 'run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1900_500nsga2initial/2021_02_20_11_52_50']),
    ('nsga2-dt', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_new_0.1_0.5_1000_500nsga2initial/2021_02_22_23_55_00', 'run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_new_0.1_0.5_1000_500nsga2initial_2/2021_02_23_09_12_59', 'run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_new_0.1_0.5_1000_500nsga2initial_3/2021_02_24_09_41_44'])
    ]

    town01_path_list = [
    ('ga-un-nn-grad', ['run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_18_16_32_41,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial_2/2021_02_23_09_14_24,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm-un-a', ['run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_19_10_45_59,52_20_regression_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial_2/2021_02_23_00_19_26,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial_3/2021_02_23_13_07_35,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm', ['run_results/nsga2/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_19_10_46_11,50_20_regression_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 'run_results/nsga2/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial_2/2021_02_23_18_41_01,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01']),
    # ('nsga2-dt', ['run_results/nsga2-dt/town01_left_0/turn_left_town01/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_19_17_55_52']),
    ('nsga2-dt', ['run_results/nsga2-dt/town01_left_0/turn_left_town01/lbc/new_new_0.1_0.5_1000_500nsga2initial/2021_02_23_08_49_48', 'run_results/nsga2-dt/town01_left_0/turn_left_town01/lbc/new_new_0.1_0.5_1000_500nsga2initial_2/2021_02_23_13_53_07'])
    ]

    town03_out_of_road_path_list = [
    ('ga-un-nn-grad', ['run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road/2021_02_20_23_50_02,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road_2/2021_02_22_11_30_44,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm-un-a', ['run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road/2021_02_21_16_22_51,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road_2/2021_02_22_11_30_32,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road_3/2021_02_24_14_57_40,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm', ['run_results/nsga2/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road/2021_02_21_16_22_16,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 'run_results/nsga2/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road_2/2021_02_22_11_30_37,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 'run_results/nsga2/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road_3/2021_02_24_14_57_48,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01']),
    # ('nsga2-dt', ['run_results/nsga2-dt/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road/2021_02_21_10_57_24', 'run_results/nsga2-dt/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial_out_of_road_2/2021_02_22_00_22_22']),
    ('nsga2-dt', ['run_results/nsga2-dt/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_new_0.1_0.5_1000_500nsga2initial_out_of_road/2021_02_22_22_05_51', 'run_results/nsga2-dt/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_new_0.1_0.5_1000_500nsga2initial_out_of_road_2/2021_02_23_09_12_57', 'run_results/nsga2-dt/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_new_0.1_0.5_1000_500nsga2initial_out_of_road_3/2021_02_24_14_57_55']),

    ]

    town05_out_of_road_path_list = [
    ('ga-un-nn-grad', ['run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road/2021_02_20_23_49_57,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road_2/2021_02_22_00_22_13,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road_3/2021_02_23_14_43_00,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm-un-a', ['run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road/2021_02_22_00_21_51,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road_2/2021_02_22_11_30_18,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road_3/2021_02_23_14_42_28,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm', ['run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road/2021_02_22_00_22_02,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 'run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road_2/2021_02_22_11_30_24,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 'run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500nsga2initial_700_out_of_road_3']),
    # ('nsga2-dt', ['run_results/nsga2-dt/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_700_500nsga2initial_out_of_road/2021_02_20_23_49_52']),
    ('nsga2-dt', ['run_results/nsga2-dt/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_new_0.1_0.5_700_500nsga2initial_out_of_road/2021_02_22_22_05_46', 'run_results/nsga2-dt/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_new_0.1_0.5_700_500nsga2initial_out_of_road_2/2021_02_23_09_13_04', 'run_results/nsga2-dt/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_new_0.1_0.5_700_500nsga2initial_out_of_road_3/2021_02_23_14_42_39']),
    ]

    town07_ablation_path_list = [
    ('ga-un-nn-grad(eps=1)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_17_22_40_12,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn-grad(eps=0.3)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_21_10_57_47,50_22_adv_nn_700_100_0.3_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_0.3']),
    ('ra-un-nn-grad', ['run_results/random-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_500nsga2initial/2021_02_20_19_22_41,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_17_22_40_59,50_20_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un', ['run_results/seeds/nsga2_un_1500/town07_front_0/2021_02_17_11_54_52,50_30_none_1500_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga', ['run_results/nsga2/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_500nsga2initial_1000/2021_02_21_13_58_17,50_22_none_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01']),
    ('ra', ['run_results/random/town07_front_0/go_straight_town07/lbc/0.1_0.5_700_500nsga2initial/2021_02_21_13_58_49,50_22_none_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01'])
    ]

    town07_seeds_ablation_path_list = [
    ('ga-un-nn-grad(eps=1)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_17_22_40_12,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'])

    ]

    town07_unique_ablation_path_list = [
    ('ga-un-nn-grad(0.05, 0.25)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_21_19_08_51,50_22_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.05_0.25__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn-grad(0.1, 0.25)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_21_21_01_13,50_22_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.1_0.25__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn-grad(0.2, 0.25)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_21_19_25_57,50_10_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.2_0.25__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn-grad(0.05, 0.5)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_21_20_28_14,50_10_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.05_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn-grad(0.1, 0.5)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_17_22_40_12,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn-grad(0.2, 0.5)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_21_20_40_13,50_12_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.2_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('ga-un-nn-grad(0.05, 0.75)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_21_22_02_57,50_12_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.05_0.75__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    # ('ga-un-nn-grad(0.1, 0.75)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_22_22_21_35,50_12_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.1_0.75__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    # ('ga-un-nn-grad(0.2, 0.75)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_22_23_09_14,50_12_adv_nn_300_100_1.01_-4_0.9_coeff_0.0_0.2_0.75__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'])
    ]

    town07_regression_nn_unique_ablation_path_list = [
    ('regression-un-nn(0.05, 0.25)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th_regression_nn/2021_02_23_14_08_07,50_12_regression_nn_300_100_1.01_-4_0.9_coeff_0.0_0.05_0.25__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('regression-un-nn(0.1, 0.25)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th_regression_nn/2021_02_23_16_18_09,50_12_regression_nn_300_100_1.01_-4_0.9_coeff_0.0_0.1_0.25__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('regression-un-nn(0.2, 0.25)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th_regression_nn/2021_02_23_16_47_48,50_12_regression_nn_300_100_1.01_-4_0.9_coeff_0.0_0.2_0.25__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('regression-un-nn(0.05, 0.5)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th_regression_nn/2021_02_23_18_43_32,50_12_regression_nn_300_100_1.01_-4_0.9_coeff_0.0_0.05_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('regression-un-nn(0.1, 0.5)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_19_00_23_01,50_20_regression_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('regression-un-nn(0.2, 0.5)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th_regression_nn/2021_02_23_20_05_46,50_12_regression_nn_300_100_1.01_-4_0.9_coeff_0.0_0.2_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('regression-un-nn(0.05, 0.75)', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th_regression_nn/2021_02_23_20_55_39,50_12_regression_nn_300_100_1.01_-4_0.9_coeff_0.0_0.05_0.75__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'])
    ]

    town07_nsga2_dt_unique_ablation_path_list = [
    ('nsga2-dt(0.05, 0.25)', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_23_21_40_53_0.05_0.25']),
    ('nsga2-dt(0.1, 0.25)', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_23_22_30_01_0.1_0.25']),
    ('nsga2-dt(0.2, 0.25)', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_23_23_00_23_0.2_0.25']),
    ('nsga2-dt(0.05, 0.5)', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_23_23_57_47_0.05_0.5']),
    ('nsga2-dt(0.1, 0.5)', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_19_17_56_00']),
    ('nsga2-dt(0.2, 0.5)', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_23_22_45_46_0.2_0.5']),
    ('nsga2-dt(0.05, 0.75)', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_300_500nsga2initial_diff_th/2021_02_23_23_58_25_0.05_0.75'])
    ]

    town07_path_list_100initial = [
    ('ga-un-nn-grad', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_100nsga2initial/2021_02_24_01_04_14,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm-un-a', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_100nsga2initial/2021_02_24_01_03_59,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-dt', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_100nsga2initial/2021_02_24_01_04_06'])
    ]

    town07_path_list_500randominitial = [
    ('ga-un-nn-grad', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500randominitial/2021_02_16_01_05_59,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_use_alternate_nn_0_2_1_60_60_diversity_mode_none_uncertainty_exploration_confidence_100_100_1']),
    ('nsga2-sm-un-a', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_1000_500randominitial/2021_02_24_09_00_15,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-dt', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_500randominitial/2021_02_24_09_00_26'])
    ]


    town07_path_list_1000initial = [
    ('ga-un-nn-grad', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_1000nsga2initial/2021_02_24_11_44_17,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-sm-un-a', ['run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_1000nsga2initial/2021_02_24_11_44_07,50_40_regression_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('nsga2-dt', ['run_results/nsga2-dt/town07_front_0/go_straight_town07/lbc/new_0.1_0.5_700_1000nsga2initial/2021_02_24_11_44_12'])
    ]


    town05_controllers_path_list = [
    ('lbc', ['run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/new_0.1_0.5_500initial_1000/2021_02_16_11_43_05,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_100_100_only_unique_1_eps_1.01']),
    ('pid-1', ['run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/auto_pilot/0.1_0.5_700_500nsga2initial/2021_02_24_08_53_29,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01']),
    ('pid-2', ['run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/pid_agent/0.1_0.5_700_500nsga2initial/2021_02_24_09_00_47,50_40_adv_nn_700_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'])
    ]





    warmup_pth_town07 = 'run_results/seeds/nsga2_un_1500/town07_front_0/2021_02_17_11_54_52,50_30_none_1500_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'
    warmup_pth_town01 = 'run_results/seeds/nsga2_un_1500/town01_left_0/2021_02_17_22_39_22,50_30_none_1500_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'
    warmup_pth_town03_out_of_road = 'run_results/seeds/nsga2_un_1500_out_of_road/town03_front_1/2021_02_19_23_28_03,50_30_none_1500_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_250_250_only_unique_1_eps_1.01'
    warmup_pth_town05_out_of_road = 'run_results/seeds/nsga2_un_1500_out_of_road/town05_right_0/2021_02_19_23_28_04,50_30_none_1500_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_250_250_only_unique_1_eps_1.01'
    warmup_pth_town05_collision = 'run_results/seeds/nsga2_un_1500/town05_right_0/2021_02_17_11_54_58,50_30_none_1500_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'
    warmup_pth_cutoff = 500


    # town_path_lists = [town07_path_list, town01_path_list, town03_out_of_road_path_list, town05_out_of_road_path_list, town07_ablation_path_list, town07_seeds_ablation_path_list]
    # warmup_pths = [warmup_pth_town07, warmup_pth_town01, warmup_pth_town03_out_of_road, warmup_pth_town05_out_of_road, warmup_pth_town07, warmup_pth_town07]
    # bug_types = ['collision', 'collision', 'out-of-road', 'out-of-road', 'collision', 'collision']
    # towns = ['town07', 'town01', 'town03', 'town05', 'town07_ablation', 'town07_seeds_ablation']
    # range_upper_bounds = [15, 15, 15, 15, 15, 7]
    # unique_coeffs_list = [[], [], [], [], [], []]

    town_path_lists = [town07_path_list, town01_path_list, town03_out_of_road_path_list, town05_out_of_road_path_list]
    warmup_pths = [warmup_pth_town07, warmup_pth_town01, warmup_pth_town03_out_of_road, warmup_pth_town05_out_of_road]
    bug_types = ['collision', 'collision', 'out-of-road', 'out-of-road']
    towns = ['town07', 'town01', 'town03', 'town05']
    range_upper_bounds = [15, 15, 15, 15]
    unique_coeffs_list = [[], [], [], []]

    # town_path_lists = [town07_path_list_1000initial]
    # warmup_pths = [warmup_pth_town07]
    # bug_types = ['collision']
    # towns = ['town07 (1000 ga-un)']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]


    # town_path_lists = [town05_controllers_path_list]
    # warmup_pths = [warmup_pth_town05_collision]
    # bug_types = ['all']
    # towns = ['town05']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]


    # town_path_lists = [town07_nsga2_dt_unique_ablation_path_list]
    # warmup_pths = [warmup_pth_town07]
    # bug_types = ['collision']
    # towns = ['town07_unique_ablation']
    # range_upper_bounds = [7]
    # unique_coeffs_list = [[[0.05, 0.25], [0.1, 0.25], [0.2, 0.25], [0.05, 0.5], [0.1, 0.5], [0.2, 0.5], [0.05, 0.75]]]

    def draw_simulation_wrapper(town_path_list, warmup_pth, bug_type, town, range_upper_bound, unique_coeffs):
        # 'collision', 'out-of-road'
        save_filename = 'num_of_unique_bugs_'+town+'_'+bug_type+'.pdf'
        scene_name = town + ' ' + bug_type
        print('-'*20, scene_name, '-'*20)
        draw_unique_bug_num_over_simulations(town_path_list, [warmup_pth], warmup_pth_cutoff, save_filename=save_filename, scene_name=scene_name, legend=True, range_upper_bound=range_upper_bound, bug_type=bug_type, unique_coeffs=unique_coeffs)


    for i in range(len(town_path_lists)):
        town_path_list = town_path_lists[i]
        warmup_pth = warmup_pths[i]
        bug_type = bug_types[i]
        town = towns[i]
        range_upper_bound = range_upper_bounds[i]
        unique_coeffs = unique_coeffs_list[i]
        draw_simulation_wrapper(town_path_list, warmup_pth, bug_type, town, range_upper_bound, unique_coeffs)
