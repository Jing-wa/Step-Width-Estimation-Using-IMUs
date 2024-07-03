import time
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import gridspec
from scipy.stats import spearmanr, f_oneway, shapiro, wilcoxon
# from SharedProcessors.const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES, FONT_SIZE_SMALL
# from Drawer import format_plot, save_fig
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error
from matplotlib import rc


FONT_SIZE_LARGE = 16
FONT_SIZE_LARGE_MORE = 20
FONT_SIZE = 10
FONT_SIZE_Mid = 12
FONT_SIZE_SMALL = 8
FONT_DICT = {'fontsize': FONT_SIZE, 'fontname': 'Arial'}
FONT_DICT_Mid = {'fontsize': FONT_SIZE_Mid, 'fontname': 'Arial'}
FONT_DICT_LARGE = {'fontsize': FONT_SIZE_LARGE, 'fontname': 'Arial'}
FONT_DICT_LARGE_MORE = {'fontsize': FONT_SIZE_LARGE_MORE, 'fontname': 'Arial'}
FONT_DICT_SMALL = {'fontsize': FONT_SIZE_SMALL, 'fontname': 'Arial'}
FONT_DICT_X_SMALL = {'fontsize': 8, 'fontname': 'Arial'}
LINE_WIDTH = 2
LINE_WIDTH_THICK = 3


def format_plot():
    mpl.rcParams['hatch.linewidth'] = LINE_WIDTH  # previous svg hatch linewidth
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=LINE_WIDTH)
    ax.yaxis.set_tick_params(width=LINE_WIDTH)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)


def save_fig(name, dpi=600):
    plt.savefig('exports/t/' + name + '.pdf', dpi=dpi)


def set_example_bar_ticks(min_val, max_val):
    ax = plt.gca()
    ax.set_xlim(min_val, max_val)
    ax.set_xticks(np.arange(int(min_val)-1, int(max_val) + 1, 10), fontdict=FONT_DICT_LARGE)
    # ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontdict=FONT_DICT_SMALL)
    ax.set_xlabel('Ground Truth', fontdict=FONT_DICT_LARGE)

    ax.set_ylim(min_val, max_val)
    ax.set_yticks(np.arange(int(min_val)-1, int(max_val) + 1, 10), fontdict=FONT_DICT_LARGE)
    # ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontdict=FONT_DICT_SMALL)
    ax.set_ylabel('Model Estimation', fontdict=FONT_DICT_LARGE)


def plot_mae_sara_speed():
    from utils import finder
    from sklearn.metrics import mean_absolute_error as mae
    from scipy.stats import ttest_ind
    from matplotlib import rc
    import matplotlib.lines as lines

    def format_ticks(x_locs):
        ax = plt.gca()
        ax.set_ylabel('Mean Absolute Error (cm)', fontdict=FONT_DICT_LARGE)
        ax.text(1.3, -0.8, 'SARA Score                               Walking Speed (m/s)', fontdict=FONT_DICT_LARGE)
        ax.xaxis.set_label_coords(0.47, -0.08)
        ax.set_ylim(0, 7.0)
        ax.set_yticks(range(0, 7, 1))
        ax.set_yticklabels(range(0, 7, 1), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-0.1, 8)
        ax.set_xticks(x_locs)
        ax.set_xticklabels(['Mild', 'Moderate', 'Severe', '', 'Slow', 'Medium', 'Fast'], fontdict=FONT_DICT_LARGE)
        ax.xaxis.set_tick_params(width=0)

    def format_errorbar_cap(caplines, size=15):
        for i_cap in range(1):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(size)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    def draw_paired_signifi_sign(mean_, std_, bar_locs, p_between_pattern, ylim):
        color_0, color_1 = np.array([90, 140, 20]) / 255, np.array([0, 103, 137]) / 255
        one_two, two_three, one_three = p_between_pattern
        y_top = max([a + b for a, b in zip(mean_, std_)])
        top_lines = [y_top + 0.07 * ylim, y_top + 0.2 * ylim, y_top + 0.33 * ylim]
        for i_pair, [pair, loc_0, loc_1] in enumerate(zip([one_two, two_three, one_three], [0, 1, 0], [1, 2, 2])):
            if not pair[0] and not pair[1]: continue
            top_line = top_lines.pop(0)
            if loc_0 == 0 and loc_1 == 2:
                coe_0, coe_1 = 0.53, 0.47
            else:
                coe_0, coe_1 = 0.56, 0.44
            if pair[0]:
                plt.plot([bar_locs[2 * loc_0], bar_locs[2 * loc_1]], [top_line, top_line], color=color_0,
                         linewidth=LINE_WIDTH)
            if pair[1]:
                plt.plot([bar_locs[2 * loc_0 + 1], bar_locs[2 * loc_1 + 1]],
                         [top_line - 0.025 * ylim, top_line - 0.025 * ylim],
                         color=color_1, linewidth=LINE_WIDTH)
            plt.text(bar_locs[2 * loc_0] * coe_0 + bar_locs[2 * loc_1 + 1] * coe_1, top_line - 0.097 * ylim, '*',
                     fontdict={'fontname': 'Times New Roman'}, size=32, zorder=20)
            rect = patches.Rectangle(
                (bar_locs[2 * loc_0] * coe_0 + bar_locs[2 * loc_1 + 1] * coe_1, top_line - 0.097 * ylim), 0.4,
                0.15 * ylim, linewidth=0, color='white', zorder=10)
            ax.add_patch(rect)

    def draw_signifi_sign(mean_, std_, bar_locs, one_two=True, two_three=True, one_three=True):
        x_offset = 0.3
        lo = [0. for i in range(3)]  # line offset
        for i_bar, there_are_two_lines in enumerate(
                [one_two and one_three, one_two and two_three, two_three and one_three]):
            if there_are_two_lines:
                lo[i_bar] = 0.1
        y_top = max([a + b for a, b in zip(mean_, std_)])

        for pair, loc_0, loc_1 in zip([one_two, two_three, one_three], [0, 1, 0], [1, 2, 2]):
            if not pair: continue
            if loc_0 == 0 and loc_1 == 2:
                lo_sign = -1
                if not one_two and not two_three:
                    top_line = y_top + 0.5
                else:
                    top_line = y_top + 1.0
                coe_0, coe_1 = 0.65, 0.35
            else:
                lo_sign = 1
                top_line = y_top + 0.5
                coe_0, coe_1 = 0.77, 0.23
            diff_line_0x = [bar_locs[loc_0] + lo_sign * lo[loc_0], bar_locs[loc_0] + lo_sign * lo[loc_0],
                            bar_locs[loc_1] - lo_sign * lo[loc_1], bar_locs[loc_1] - lo_sign * lo[loc_1]]
            diff_line_0y = [mean_[loc_0] + std_[loc_0] + x_offset, top_line, top_line,
                            mean_[loc_1] + std_[loc_1] + x_offset]
            plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
            plt.text((bar_locs[loc_0] + lo_sign * lo[loc_0]) * coe_0 + (bar_locs[loc_1] - lo_sign * lo[loc_1]) * coe_1 + 0.05,
                     top_line - 0.15,
                     '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=40)

    # path
    path = 'exports/new_results/'

    # data prepare #########################################################
    mild_subjects = ['3107202315', '3107202314', '28062023', '507202310', '507202311']
    moderate_subjects = ['26062023', '1207202310', '1207202311', '2607202310', '2607202311']
    severe_subjects = ['2407202313', '2407202314', '208202310']
    slow_subjects = ['1207202311', '2407202314', '208202310']
    medium_subjects = ['28062023', '507202311', '1207202310', '2407202313', '2607202310', '3107202315']
    fast_subjects = ['26062023', '507202310', '2607202311', '3107202314']

    # disease
    disease_level_list = ['mild', 'moderate', 'severe']
    sara = {key: [] for key in disease_level_list}
    p_sara = []

    for i, disease_level in enumerate(disease_level_list):
        for subject in eval(disease_level + '_subjects'):
            sub_paths = finder(subject, path)
            for sub_path in sub_paths:
                sub_data = pd.read_csv(sub_path, header=None)
                sub_y_true = sub_data.iloc[:, 0]
                sub_y_pred = sub_data.iloc[:, 1]
                MAE = mae(sub_y_true, sub_y_pred)
                sara[disease_level].append(MAE)

    for i, pair in enumerate([(0, 1), (1, 2), (0, 2)]):
        p_sara.append(ttest_ind(sara[disease_level_list[pair[0]]], sara[disease_level_list[pair[1]]]).pvalue)

    # speed
    speed_list = ['slow', 'medium', 'fast']
    speeds = {key: [] for key in speed_list}
    p_speed = []

    for i, speed in enumerate(speeds):
        for subject in eval(speed + '_subjects'):
            sub_paths = finder(subject, path)
            for sub_path in sub_paths:
                sub_data = pd.read_csv(sub_path, header=None)
                sub_y_true = sub_data.iloc[:, 0]
                sub_y_pred = sub_data.iloc[:, 1]
                MAE = mae(sub_y_true, sub_y_pred)
                speeds[speed].append(MAE)

    for i, pair in enumerate([(0, 1), (1, 2), (0, 2)]):
        p_speed.append(ttest_ind(speeds[speed_list[pair[0]]], speeds[speed_list[pair[1]]]).pvalue)

    # plot ################################################################
    rc('font', family='Arial')
    # fig = plt.figure(figsize=(9, 6))
    fig, ax = plt.subplots(figsize=(8, 6))
    format_plot()
    x_locs = [1, 2, 3, 4, 5, 6, 7]

    _mean = [np.mean(sara['mild']), np.mean(sara['moderate']), np.mean(sara['severe']),
                np.mean(speeds['slow']), np.mean(speeds['medium']), np.mean(speeds['fast'])]
    _std = [np.std(sara['mild']), np.std(sara['moderate']), np.std(sara['severe']),
            np.std(speeds['slow']), np.std(speeds['medium']), np.std(speeds['fast'])]

    ax.bar([1, 2, 3, 5, 6, 7], _mean, align='center', alpha=0.7, ecolor='blue', capsize=10, width=0.5)
    ebar, caplines, barlinecols = ax.errorbar([1, 2, 3, 5, 6, 7], _mean, _std, capsize=0, ecolor='black', fmt='none',
                                              lolims=True, elinewidth=LINE_WIDTH_THICK)
    format_errorbar_cap(caplines, 10)

    format_ticks(x_locs)

    # draw significance sign
    draw_signifi_sign(_mean[:3], _std[:3], x_locs[:3], one_two=False)
    draw_signifi_sign(_mean[3:], _std[3:], x_locs[4:], one_three=False)

    plt.tight_layout(rect=[0.0, 0.0, 1, 1])
    l2 = lines.Line2D([0.53, 0.53], [0.05, 0.85], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])
    plt.savefig('exports/results/' + 'mae_sara_speed_2' + '.pdf', dpi=800)
    plt.show()
    print(p_sara, p_speed)
    print(_mean, _std)
    return


def plot_scatter_bland_altman():
    # each fold
    from utils import finder
    from matplotlib import rc
    import matplotlib.gridspec as gridspec

    def bland_altman(ax, fold_y_true, fold_y_pred, var1, var2, minlimit, maxlimit):
        """
        Bland-Altman Plot
        :param data: Pandas dataframe containing data
        :param var1: String containing name of first variable
        :param var2: String containing name of second variable
        :param minlimit: Minimum limit of plot
        :param maxlimit: Maximum limit of plot
        :return: None
        """
        # determine mean and difference
        ba_y_true = np.array([item for sublist in fold_y_true.values() for item in sublist])
        ba_y_pred = np.array([item for sublist in fold_y_pred.values() for item in sublist])
        data = pd.DataFrame({'true': ba_y_true, 'pred': ba_y_pred})
        mean = (data.iloc[:, 0] + data.iloc[:, 1]) / 2
        diff = data.iloc[:, 0] - data.iloc[:, 1]

        # get mean and standard deviation of the differences
        mean_diff = diff.mean()
        std_diff = diff.std()

        # determine 95% limits of agreement
        LoA_plus = mean_diff + 1.96 * std_diff
        LoA_neg = mean_diff - 1.96 * std_diff

        color = ['orange', 'green', 'blue', 'purple']
        for i, [key, value] in enumerate(fold_y_true.items()):
            length = len(value)
            start = sum([len(value) for value in list(fold_y_true.values())[:i]])
            end = start + length
            ax.scatter(mean[start:end], diff[start:end], s=40, marker='.', alpha=0.7, edgecolors='none', color=color[i])

        minlimit = min(min(ba_y_true), min(ba_y_pred)) - 3
        maxlimit = max(max(ba_y_true), max(ba_y_pred)) + 3
        ax.set_xlim((minlimit, maxlimit))
        xval = ax.get_xlim()
        ax.plot(xval, (mean_diff, mean_diff))
        ax.plot(xval, (LoA_plus, LoA_plus), 'r--')
        ax.plot(xval, (LoA_neg, LoA_neg), 'r--')
        # set text
        # for ataxia patients
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_plus + xval[1] * 0.015, 'Mean+1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_neg + xval[1] * 0.015, 'Mean-1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff + xval[1] * 0.015, 'Mean', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_plus - xval[1] * 0.035, round(LoA_plus, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_neg - xval[1] * 0.035, round(LoA_neg, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff - xval[1] * 0.035, round(mean_diff, 2), fontdict=FONT_DICT_LARGE)

        # for healthy subjects
        ax.text(xval[1] - (xval[1] / 8) - 5, LoA_plus + xval[1] * 0.01, 'Mean+1.96SD', fontdict=FONT_DICT_LARGE)
        ax.text(xval[1] - (xval[1] / 8) - 5, LoA_neg + xval[1] * 0.01, 'Mean-1.96SD', fontdict=FONT_DICT_LARGE)
        ax.text(xval[1] - (xval[1] / 8), mean_diff + xval[1] * 0.01, 'Mean', fontdict=FONT_DICT_LARGE)
        ax.text(xval[1] - (xval[1] / 8), LoA_plus - xval[1] * 0.025, round(LoA_plus, 2), fontdict=FONT_DICT_LARGE)
        ax.text(xval[1] - (xval[1] / 8), LoA_neg - xval[1] * 0.025, round(LoA_neg, 2), fontdict=FONT_DICT_LARGE)
        ax.text(xval[1] - (xval[1] / 8), mean_diff - xval[1] * 0.025, round(mean_diff, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('Mean of ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        ax.set_ylabel('Difference between ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        format_plot()

    def show_each_pair(ax, y_true, y_pred, sub_id):
        si_true = [item for sublist in fold_y_true.values() for item in sublist]
        si_pred = [item for sublist in fold_y_pred.values() for item in sublist]

        format_plot()
        color = ['orange', 'green', 'blue', 'purple']
        if len(y_true.keys()) == 2:
            sc = [0., 0.]
        elif len(y_true.keys()) == 3:
            sc = [0., 0., 0.]
        for i, [key, value] in enumerate(y_true.items()):
            sc[i] = ax.scatter(value, y_pred[key], s=40, marker='.', alpha=0.7, edgecolors='none',
                               color=color[i])

        min_value = min(min(si_true), min(si_pred))
        max_value = max(max(si_true), max(si_pred))
        set_example_bar_ticks(min_value, max_value)

        coef = np.polyfit(si_true, si_pred, 1)
        poly1d_fn = np.poly1d(coef)
        black_line, = plt.plot([min_value, max_value], poly1d_fn([min_value, max_value]), color='black',
                               linewidth=LINE_WIDTH)
        # ax = plt.gca()
        scc = spearmanr(si_true, si_pred)[0]
        mae = mean_absolute_error(si_true, si_pred)
        # rmse = 100 * np.sqrt(mean_squared_error(si_true, si_pred))
        # R2 = r2_score(si_true, si_pred)
        ax.text(0.6, 0.135, 'MAE = {:4.2f}'.format(mae), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.6, 0.08, 'SCC = {:4.2f}'.format(scc), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        # ax.text(0.6, 0.08, '$R^2$ = {:4.2f}'.format(R2), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.6, 0.03, '$y$ = {:4.2f}$x$ + {:4.2f}'.format(coef[0], coef[1]),
                fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        plt.tight_layout(rect=[0, 0, 1, 1])
        ax.legend([sc[0], sc[1], black_line], ['Subject A', 'Subject B', 'Regression Line'], fontsize=FONT_SIZE_Mid,
                   frameon=False, bbox_to_anchor=(0.5, 0.99))
        print('Fold {}, SCC: {:.2f}, R2: {:.2f}'.format(sub_id, scc, mae))
        # print('The RMSE and $R^2$ of this subject were {:.1f}\% and {:.2f}'.format(rmse, R2))
        # t0 = time.time()
        # save_fig('f1' + str(sub_id) + '--' + str(t0), 600)

    # path
    path = 'exports/new_results/new_healthy_r_t'
    folds = [0, 1, 2, 3, 4, 5, 6, 7]
    epoch = 30
    for i_fold in folds:
        i_fold = 5
        pattern = str(i_fold) + '-' + str(epoch)
        sub_paths = finder(pattern, path)
        fold_y_true = {}
        fold_y_pred = {}
        for sub_path in sub_paths:
            sub_data = pd.read_csv(sub_path, header=None)
            sub_y_true = sub_data.iloc[:, 0]
            sub_y_pred = sub_data.iloc[:, 1]

            fold_y_true[sub_path] = list(sub_y_true)
            fold_y_pred[sub_path] = list(sub_y_pred)

        # plot bland-altman at the left and plot show_each_pair at the right, in one figure
        rc('font', family='Arial')
        fig = plt.figure(figsize=(12, 6))
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6])

        # draw scatter
        show_each_pair(fig.add_subplot(gs[1, 1]), fold_y_true, fold_y_pred, i_fold)

        # draw bland-altman
        ba_y_true = [item for sublist in fold_y_true.values() for item in sublist]
        ba_y_pred = [item for sublist in fold_y_pred.values() for item in sublist]
        bland_altman(fig.add_subplot(gs[1, 0]), fold_y_true, fold_y_pred,
                     'Ground Truth', 'Model Estimation', -10, 40)

        plt.tight_layout(rect=[0., -0.02, 1., 1.1], w_pad=3)
        # plt.savefig('exports/results/healthy_f' + str(i_fold) + '.pdf', dpi=800)
        plt.show()
        break


def plot_scatter_bland_altman_ieee_one_sample():
    # each sample
    from utils import finder
    from matplotlib import rc
    import matplotlib.gridspec as gridspec

    def bland_altman_ieee(ax, fold_y_true, fold_y_pred, var1, var2, minlimit, maxlimit):
        """
        Bland-Altman Plot
        :param data: Pandas dataframe containing data
        :param var1: String containing name of first variable
        :param var2: String containing name of second variable
        :param minlimit: Minimum limit of plot
        :param maxlimit: Maximum limit of plot
        :return: None
        """
        # determine mean and difference
        # ba_y_true = np.array([item for sublist in fold_y_true.values() for item in sublist])
        # ba_y_pred = np.array([item for sublist in fold_y_pred.values() for item in sublist])
        data = pd.DataFrame({'true': np.array(fold_y_true), 'pred': np.array(fold_y_pred)})
        mean = (data.iloc[:, 0] + data.iloc[:, 1]) / 2
        diff = data.iloc[:, 0] - data.iloc[:, 1]

        # get mean and standard deviation of the differences
        mean_diff = diff.mean()
        std_diff = diff.std()

        # determine 95% limits of agreement
        LoA_plus = mean_diff + 1.96 * std_diff
        LoA_neg = mean_diff - 1.96 * std_diff

        ax.scatter(mean, diff, s=40, marker='.', alpha=0.7, edgecolors='none', color='green')

        minlimit = min(min(fold_y_true), min(fold_y_pred)) - 3
        maxlimit = max(max(fold_y_true), max(fold_y_pred)) + 3
        ax.set_xlim((minlimit, maxlimit))
        xval = ax.get_xlim()
        ax.plot(xval, (mean_diff, mean_diff))
        ax.plot(xval, (LoA_plus, LoA_plus), 'r--')
        ax.plot(xval, (LoA_neg, LoA_neg), 'r--')
        # set text
        # for ataxia patients
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_plus + xval[1] * 0.015, 'Mean+1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_neg + xval[1] * 0.015, 'Mean-1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff + xval[1] * 0.015, 'Mean', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_plus - xval[1] * 0.035, round(LoA_plus, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_neg - xval[1] * 0.035, round(LoA_neg, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff - xval[1] * 0.035, round(mean_diff, 2), fontdict=FONT_DICT_LARGE)

        # for healthy subjects
        ax.text(xval[1] - (xval[1] / 8) - 2, LoA_plus + xval[1] * 0.01, 'Mean+1.96SD', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) - 2, LoA_neg + xval[1] * 0.01, 'Mean-1.96SD', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, mean_diff + xval[1] * 0.01, 'Mean', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, LoA_plus - xval[1] * 0.025, round(LoA_plus, 2), fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, LoA_neg - xval[1] * 0.025, round(LoA_neg, 2), fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, mean_diff - xval[1] * 0.025, round(mean_diff, 2), fontdict=FONT_DICT_Mid)
        ax.set_xlabel('Mean of ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        ax.set_ylabel('Difference between ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        format_plot()

    def show_each_pair_ieee(ax, y_true, y_pred, sub_id):
        si_true = y_true
        si_pred = y_pred

        format_plot()

        sc = [0.]
        sc[0] = ax.scatter(si_true, si_pred, s=40, marker='.', alpha=0.7, edgecolors='none', color='green')

        min_value = min(min(si_true), min(si_pred))
        max_value = max(max(si_true), max(si_pred))
        set_example_bar_ticks(min_value, max_value)

        coef = np.polyfit(si_true, si_pred, 1)
        poly1d_fn = np.poly1d(coef)
        black_line, = plt.plot([min_value, max_value], poly1d_fn([min_value, max_value]), color='black',
                               linewidth=LINE_WIDTH)
        # ax = plt.gca()
        scc = spearmanr(si_true, si_pred)[0]
        mae = mean_absolute_error(si_true, si_pred)
        # rmse = 100 * np.sqrt(mean_squared_error(si_true, si_pred))
        # R2 = r2_score(si_true, si_pred)
        ax.text(0.7, 0.135, 'MAE = {:4.2f}'.format(mae), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.7, 0.08, 'SCC = {:4.2f}'.format(scc), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        # ax.text(0.6, 0.08, '$R^2$ = {:4.2f}'.format(R2), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.7, 0.03, '$y$ = {:4.2f}$x$ + {:4.2f}'.format(coef[0], coef[1]),
                fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        plt.tight_layout(rect=[0, 0, 1, 1])
        # ax.legend([sc[0], sc[1], black_line], ['Subject A', 'Subject B', 'Regression Line'], fontsize=FONT_SIZE_Mid,
        #            frameon=False, bbox_to_anchor=(0.5, 0.99))
        ax.legend([sc[0], black_line], ['Healthy', 'Regression Line'], fontsize=FONT_SIZE_LARGE,
                   frameon=False, bbox_to_anchor=(0.51, 0.99))
        print('Fold {}, SCC: {:.2f}, R2: {:.2f}'.format(sub_id, scc, mae))
        # print('The RMSE and $R^2$ of this subject were {:.1f}\% and {:.2f}'.format(rmse, R2))
        # t0 = time.time()
        # save_fig('f1' + str(sub_id) + '--' + str(t0), 600)

    # path
    path = 'exports/new_results/new_healthy_r_t'
    # path = 'exports/new_results/new_results without pretrained weights'

    sub_paths = finder('.csv', path)[:10]
    for sub_path in sub_paths:
        # sub_path = 'exports/new_results/new_healthy_r_t/10-5-30.csv'
        sub_data = pd.read_csv(sub_path, header=None)
        sub_y_true = list(sub_data.iloc[:, 0])
        sub_y_pred = list(sub_data.iloc[:, 1])

        # plot bland-altman at the left and plot show_each_pair at the right, in one figure
        rc('font', family='Arial')
        fig = plt.figure(figsize=(12, 6))
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6])

        # draw scatter
        show_each_pair_ieee(fig.add_subplot(gs[1, 0]), sub_y_true, sub_y_pred, sub_path)

        # draw bland-altman
        bland_altman_ieee(fig.add_subplot(gs[1, 1]), sub_y_true, sub_y_pred,
                     'Ground Truth', 'Model Estimation', -10, 40)

        plt.tight_layout(rect=[0., -0.02, 1., 1.1], w_pad=3)
        # plt.savefig('exports/results/ataxia_f' + str('2607202311-4-150') + '.pdf', dpi=800)
        plt.show()
        # break


def plot_scatter_bland_altman_ieee_all_sample():
    # each sample
    from utils import finder
    from matplotlib import rc
    import matplotlib.gridspec as gridspec

    def bland_altman_ieee(ax, fold_y_true, fold_y_pred, var1, var2, minlimit, maxlimit):
        """
        Bland-Altman Plot
        :param data: Pandas dataframe containing data
        :param var1: String containing name of first variable
        :param var2: String containing name of second variable
        :param minlimit: Minimum limit of plot
        :param maxlimit: Maximum limit of plot
        :return: None
        """
        # determine mean and difference
        # ba_y_true = np.array([item for sublist in fold_y_true.values() for item in sublist])
        # ba_y_pred = np.array([item for sublist in fold_y_pred.values() for item in sublist])
        data = pd.DataFrame({'true': np.array(fold_y_true), 'pred': np.array(fold_y_pred)})
        mean = (data.iloc[:, 0] + data.iloc[:, 1]) / 2
        diff = data.iloc[:, 0] - data.iloc[:, 1]

        # get mean and standard deviation of the differences
        mean_diff = diff.mean()
        std_diff = diff.std()

        # determine 95% limits of agreement
        LoA_plus = mean_diff + 1.96 * std_diff
        LoA_neg = mean_diff - 1.96 * std_diff

        ax.scatter(mean, diff, s=40, marker='.', alpha=0.7, edgecolors='none', color='green')

        minlimit = min(min(fold_y_true), min(fold_y_pred)) - 3
        maxlimit = max(max(fold_y_true), max(fold_y_pred)) + 3

        # set_example_bar_ticks(minlimit, maxlimit)
        minlimit = 0
        maxlimit = 42

        # ax.set_xlim((minlimit, maxlimit))
        ax.set_xticks(np.arange(minlimit, maxlimit, step=10))
        # ax.set_yticks(np.arange(-22, 22, step=10))
        ax.set_ylim((-19, 19))
        xval = ax.get_xlim()
        ax.plot(xval, (mean_diff, mean_diff))
        ax.plot(xval, (LoA_plus, LoA_plus), 'r--')
        ax.plot(xval, (LoA_neg, LoA_neg), 'r--')
        # set text
        # for ataxia patients
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_plus + xval[1] * 0.015, 'Mean+1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_neg + xval[1] * 0.015, 'Mean-1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff + xval[1] * 0.015, 'Mean', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_plus - xval[1] * 0.035, round(LoA_plus, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_neg - xval[1] * 0.035, round(LoA_neg, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff - xval[1] * 0.035, round(mean_diff, 2), fontdict=FONT_DICT_LARGE)

        # for healthy subjects
        ax.text(xval[1] - (xval[1] / 8) - 2, LoA_plus + xval[1] * 0.01, 'Mean+1.96SD', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) - 2, LoA_neg + xval[1] * 0.01, 'Mean-1.96SD', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, mean_diff + xval[1] * 0.01, 'Mean', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, LoA_plus - xval[1] * 0.025, round(LoA_plus, 2), fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, LoA_neg - xval[1] * 0.025, round(LoA_neg, 2), fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8) + 2, mean_diff - xval[1] * 0.025, round(mean_diff, 2), fontdict=FONT_DICT_Mid)
        ax.set_xlabel('Mean of ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        ax.set_ylabel('Difference between ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        format_plot()

    def show_each_pair_ieee(ax, y_true, y_pred, sub_id):
        si_true = y_true
        si_pred = y_pred

        format_plot()

        sc = [0.]
        sc[0] = ax.scatter(si_true, si_pred, s=40, marker='.', alpha=0.7, edgecolors='none', color='green')

        min_value = min(min(si_true), min(si_pred))
        max_value = max(max(si_true), max(si_pred))
        min_value = 1
        max_value = 41
        set_example_bar_ticks(min_value, max_value)

        coef = np.polyfit(si_true, si_pred, 1)
        poly1d_fn = np.poly1d(coef)
        black_line, = plt.plot([min_value, max_value], poly1d_fn([min_value, max_value]), color='black',
                               linewidth=LINE_WIDTH)
        # ax = plt.gca()
        scc = spearmanr(si_true, si_pred)[0]
        mae = mean_absolute_error(si_true, si_pred)
        # rmse = 100 * np.sqrt(mean_squared_error(si_true, si_pred))
        # R2 = r2_score(si_true, si_pred)
        ax.text(0.7, 0.135, 'MAE = {:4.1f}'.format(2.9), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.7, 0.08, 'SCC = {:4.2f}'.format(0.85), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        # ax.text(0.6, 0.08, '$R^2$ = {:4.2f}'.format(R2), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.7, 0.03, '$y$ = {:4.1f}$x$ + {:4.1f}'.format(coef[0], coef[1]),
                fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        plt.tight_layout(rect=[0, 0, 1, 1])
        # ax.legend([sc[0], sc[1], black_line], ['Subject A', 'Subject B', 'Regression Line'], fontsize=FONT_SIZE_Mid,
        #            frameon=False, bbox_to_anchor=(0.5, 0.99))
        ax.legend([sc[0], black_line], ['Healthy', 'Regression Line'], fontsize=FONT_SIZE_LARGE,
                   frameon=False, bbox_to_anchor=(0.51, 0.99))
        print('Fold {}, SCC: {:.2f}, R2: {:.2f}'.format(sub_id, scc, r2_score(si_true, si_pred)))
        # print('The RMSE and $R^2$ of this subject were {:.1f}\% and {:.2f}'.format(rmse, R2))
        # t0 = time.time()
        # save_fig('f1' + str(sub_id) + '--' + str(t0), 600)

    # path
    path = 'exports/new_results/new_healthy_r_t'
    # path = 'exports/new_results/new_results without pretrained weights'
    sub_y_true = []
    sub_y_pred = []

    sub_paths = finder('.csv', path)   # [:10]
    # sub_paths = finder('146.csv', path)   # [:10]
    for sub_path in sub_paths:
        print(sub_path)
        # sub_path = 'exports/new_results/new_healthy_r_t/10-5-30.csv'
        sub_data = pd.read_csv(sub_path, header=None)
        sub_y_true += list(sub_data.iloc[:, 0])
        sub_y_pred += list(sub_data.iloc[:, 1])
        print(len(sub_y_true))

    # remove negative values
    negative_index = []
    for k in range(len(sub_y_true)):
        if sub_y_true[k] < 0 or sub_y_pred[k] < 0:
            negative_index.append(k)
    sub_y_true = [sub_y_true[i] for i in range(len(sub_y_true)) if i not in negative_index]
    sub_y_pred = [sub_y_pred[i] for i in range(len(sub_y_pred)) if i not in negative_index]


    # plot bland-altman at the right and plot show_each_pair at the left, in one figure
    rc('font', family='Arial')
    fig = plt.figure(figsize=(12, 6))
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6])

    # draw scatter
    show_each_pair_ieee(fig.add_subplot(gs[1, 0]), sub_y_true, sub_y_pred, sub_path)

    # draw bland-altman
    bland_altman_ieee(fig.add_subplot(gs[1, 1]), sub_y_true, sub_y_pred,
                 'Ground Truth', 'Model Estimation', -10, 40)

    plt.tight_layout(rect=[0., -0.02, 1., 1.1], w_pad=3)
    plt.savefig('exports/results/healthy_all' + '.pdf', dpi=800)
    plt.show()
    # break


def plot_box_of_each_subject():
    # plot step width of each subject including ground truth and model estimation, and plot the significance sign
    from utils import finder
    from matplotlib import rc
    import matplotlib.lines as lines
    import matplotlib.patches as mpatches
    from scipy.stats import ttest_rel, f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    def one_way_anova(y_true, y_pred):
        # perform one-way ANOVA
        f_oneway(y_true, y_pred)

        df = pd.DataFrame({'score': list(y_true) + list(y_pred),
                           'group': np.repeat(['y_true', 'y_pred'], repeats=len(y_true))})

        # perform Tukey's test
        tukey = pairwise_tukeyhsd(endog=df['score'],
                                  groups=df['group'],
                                  alpha=0.05)

        # display results
        return tukey

    def format_ticks(min_ylim, max_ylim):
        ax = plt.gca()
        ax.set_ylabel('Step Width (cm)', fontdict=FONT_DICT_LARGE_MORE, labelpad=5)
        ax.set_xlabel('Healthy Subjects', fontdict=FONT_DICT_LARGE_MORE, labelpad=5)
        ax.set_ylim(int(min_ylim-5), int(max_ylim+5))
        ax.set_yticks(np.arange(int(min_ylim), int(max_ylim), 10))
        # ax.set_yticklabels(['2', '8', '14', '20', '26'], fontdict=FONT_DICT_SMALL)
        ax.set_xlim(-0.3, 8.6)
        ax.set_xticks(bar_locs+0.1)
        ax.set_xticklabels(['P' + str(x) for x in list(range(1, 18))], fontdict=FONT_DICT_LARGE, linespacing=2.2)

    # data preparation
    path = 'exports/new_results/healthy_r_t'
    folds = [0, 1, 2, 3, 4, 5, 6, 7]
    epoch = 30
    all_y_true = {}
    all_y_pred = {}
    min_value = 100
    max_value = 0
    p = []
    t = []
    for i_fold in folds:
        pattern = str(i_fold) + '-' + str(epoch)
        sub_paths = finder(pattern, path)
        for sub_path in sub_paths:
            sub_data = pd.read_csv(sub_path, header=None)
            sub_y_true = sub_data.iloc[:, 0]
            sub_y_pred = sub_data.iloc[:, 1]
            all_y_true[sub_path] = list(sub_y_true)
            all_y_pred[sub_path] = list(sub_y_pred)
            min_value = min(min_value, min(sub_y_true), min(sub_y_pred))
            max_value = max(max_value, max(sub_y_true), max(sub_y_pred))
            p.append(ttest_rel(sub_y_true, sub_y_pred).pvalue)
            t.append(one_way_anova(sub_y_true, sub_y_pred).pvalues[0])

    # plot step width box
    rc('font', family='Arial')
    fig = plt.figure(figsize=(15, 9))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    format_plot()
    bar_locs = np.arange(0, 0.5*17, 0.5)

    for i, [key, value] in enumerate(all_y_true.items()):
        #plot box
        box_ = plt.boxplot(value, positions=[bar_locs[i]], widths=[0.1], patch_artist=True, notch=True,
                           boxprops=dict(facecolor='lightblue'),  # Setting box color
                           whiskerprops=dict(color='black'),  # Setting whisker color
                           capprops=dict(color='black'),  # Setting cap color
                           medianprops=dict(color='yellow')  # Setting median line color
                           )
        box_ = plt.boxplot(all_y_pred[key], positions=[bar_locs[i]+0.2], widths=[0.1], patch_artist=True, notch=True,
                           boxprops=dict(facecolor='lightgreen'),  # Setting box color
                           whiskerprops=dict(color='black'),  # Setting whisker color
                           capprops=dict(color='black'),  # Setting cap color
                           medianprops=dict(color='yellow')  # Setting median line color
                           )
    ground_truth_patch = mpatches.Patch(color='lightblue', label='Ground Truth')
    model_estimation_patch = mpatches.Patch(color='lightgreen', label='Model Estimation')

    plt.legend(handles=[ground_truth_patch, model_estimation_patch],
              labels=['Ground Truth', 'Model Estimation'],
              fontsize=FONT_SIZE_LARGE_MORE, frameon=False, bbox_to_anchor=(0.8, 1.01), ncol=2)

    plt.tight_layout(rect=[0.04, 0.03, 1, 1])
    print(p)
    print(t)
    format_ticks(min_value, max_value)
    plt.savefig('exports/results/healthy_box_of_each_subject.pdf', dpi=800)
    plt.show()


def plot_step_width_variability():
    from utils import finder
    from matplotlib import rc

    # data preparation
    # path = 'exports/new_results/new_healthy_r_t'
    path = 'exports/new_results/new_results without pretrained weights'
    folds = [0, 1, 2, 3, 4, 5]
    # folds = [0, 1, 2, 3, 4, 5, 6, 7]  # [0, 1, 2, 3, 4, 5]
    epoch = 150
    # epoch = 30  # 150
    variability_true = []
    variability_pred = []
    for i_fold in folds:
        pattern = str(i_fold) + '-' + str(epoch)
        sub_paths = finder(pattern, path)
        for sub_path in sub_paths:
            # print(sub_path)
            sub_data = pd.read_csv(sub_path, header=None)
            sub_y_true = sub_data.iloc[:, 0]
            sub_y_pred = sub_data.iloc[:, 1]
            print(len(sub_y_true))
            variability_true.append(np.std(sub_y_true))
            variability_pred.append(np.std(sub_y_pred))
            # variability_true.append(np.std(sub_y_true) / np.mean(sub_y_true) * 100)
            # variability_pred.append(np.std(sub_y_pred) / np.mean(sub_y_pred) * 100)

    print(np.mean(np.array(variability_true)), np.std(np.array(variability_true)))


    # plot step width variability
    rc('font', family='Arial')
    # fig = plt.figure(figsize=(4, 3.5))
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    format_plot()

    # Define a list of colors, one for each point
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'magenta',
              'lightgreen', 'forestgreen', 'darkcyan', 'greenyellow', 'crimson'
              ]
    # Ensure that the length of this list is at least as long as the length of your data

    # Plot each point individually
    # for i in range(len(variability_true)):
    #     plt.scatter(variability_true[i], variability_pred[i], s=40, marker='.', alpha=0.7, color=colors[i], label=f'P{i + 1}')

    sc = plt.scatter(variability_true, variability_pred, s=40, marker='.', alpha=0.7, color='green')

    # Create the legend
    # plt.legend(bbox_to_anchor=(1.01, 0.99))
    # sc = plt.scatter(variability_true, variability_pred, s=40, marker='.', alpha=0.7, edgecolors='none', color='blue')

    min_value = min(min(variability_true), min(variability_pred))
    max_value = max(max(variability_true), max(variability_pred))

    ax.set_xlim(min_value-1, max_value+1)
    ax.set_xticks(np.round(np.arange(min_value - 1, max_value + 1, 2), 0), fontdict=FONT_DICT_Mid)
    ax.set_xlabel('Ground Truth', fontdict=FONT_DICT_LARGE)

    ax.set_ylim(min_value-1, max_value+1)
    ax.set_yticks(np.round(np.arange(min_value - 1, max_value + 1, 2), 0), fontdict=FONT_DICT_Mid)
    ax.set_ylabel('Model Estimation', fontdict=FONT_DICT_LARGE)

    variability_true = np.array(variability_true)
    variability_pred = np.array(variability_pred)
    coef = np.polyfit(variability_true, variability_pred, 1)
    poly1d_fn = np.poly1d(coef)
    black_line, = plt.plot([min_value, max_value], poly1d_fn([min_value, max_value]), color='black',
                           linewidth=LINE_WIDTH)
    R2 = r2_score(variability_true, variability_pred)

    RMSE = np.sqrt(mse(variability_true, variability_pred, multioutput='uniform_average'))
    print(R2)
    print(RMSE)
    mae = mean_absolute_error(variability_true, variability_pred)
    scc = spearmanr(variability_true, variability_pred)[0]
    ax.text(0.7, 0.08, 'SCC = {:4.2f}'.format(scc), fontdict=FONT_DICT, transform=ax.transAxes)
    ax.text(0.7, 0.13, 'MAE = {:4.1f}'.format(mae), fontdict=FONT_DICT, transform=ax.transAxes)
    ax.text(0.7, 0.03, '$y$ = {:4.2f}$x$ + {:4.2f}'.format(coef[0], coef[1]),
            fontdict=FONT_DICT, transform=ax.transAxes)
    plt.tight_layout(rect=[0, 0, 1, 1])
    ax.text(0.4, 0.85, 'Healthy', fontdict=FONT_DICT_LARGE, transform=ax.transAxes)
    # ax.legend(['Healthy'], fontsize=FONT_SIZE_LARGE,
    #           frameon=False, bbox_to_anchor=(0.6, 0.99))
    plt.savefig('exports/results/healthy_cv_ieee.pdf', dpi=800)
    plt.show()


def plot_step_width_variability_ieee():
    from utils import finder
    from matplotlib import rc

    def std_bar_ticks(min_val, max_val):
        ax = plt.gca()
        ax.set_xlim(min_val, max_val)
        ax.set_xticks(np.arange(int(min_val)-1, int(max_val)+1, 2), fontdict=FONT_DICT_LARGE)
        # ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontdict=FONT_DICT_SMALL)
        ax.set_xlabel('Ground Truth', fontdict=FONT_DICT_LARGE)

        ax.set_ylim(min_val, max_val)
        ax.set_yticks(np.arange(int(min_val)-1, int(max_val)+1, 2), fontdict=FONT_DICT_LARGE)
        # ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('Model Estimation', fontdict=FONT_DICT_LARGE)

    def std_bland_altman_ieee(ax, fold_y_true, fold_y_pred, var1, var2, minlimit, maxlimit):
        """
        Bland-Altman Plot
        :param data: Pandas dataframe containing data
        :param var1: String containing name of first variable
        :param var2: String containing name of second variable
        :param minlimit: Minimum limit of plot
        :param maxlimit: Maximum limit of plot
        :return: None
        """
        # determine mean and difference
        # ba_y_true = np.array([item for sublist in fold_y_true.values() for item in sublist])
        # ba_y_pred = np.array([item for sublist in fold_y_pred.values() for item in sublist])
        data = pd.DataFrame({'true': np.array(fold_y_true), 'pred': np.array(fold_y_pred)})
        mean = (data.iloc[:, 0] + data.iloc[:, 1]) / 2
        diff = data.iloc[:, 0] - data.iloc[:, 1]

        # get mean and standard deviation of the differences
        mean_diff = diff.mean()
        std_diff = diff.std()

        # determine 95% limits of agreement
        LoA_plus = mean_diff + 1.96 * std_diff
        LoA_neg = mean_diff - 1.96 * std_diff

        ax.scatter(mean, diff, s=70, marker='.', alpha=0.7, edgecolors='none', color='green')

        minlimit = min(min(fold_y_true), min(fold_y_pred)) - 1
        maxlimit = max(max(fold_y_true), max(fold_y_pred)) + 3

        # set_example_bar_ticks(minlimit, maxlimit)
        minlimit = 3
        maxlimit = 10

        # ax.set_xlim((minlimit, maxlimit))
        ax.set_xticks(np.arange(minlimit, maxlimit, step=2))
        # ax.set_yticks(np.arange(-22, 22, step=10))
        ax.set_ylim((-5, 5))
        xval = ax.get_xlim()
        ax.plot(xval, (mean_diff, mean_diff))
        ax.plot(xval, (LoA_plus, LoA_plus), 'r--')
        ax.plot(xval, (LoA_neg, LoA_neg), 'r--')
        # set text
        # for ataxia patients
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_plus + xval[1] * 0.015, 'Mean+1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8) - 10, LoA_neg + xval[1] * 0.015, 'Mean-1.96SD', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff + xval[1] * 0.015, 'Mean', fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_plus - xval[1] * 0.035, round(LoA_plus, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), LoA_neg - xval[1] * 0.035, round(LoA_neg, 2), fontdict=FONT_DICT_LARGE)
        # ax.text(xval[1] - (xval[1] / 8), mean_diff - xval[1] * 0.035, round(mean_diff, 2), fontdict=FONT_DICT_LARGE)

        # for healthy subjects
        ax.text(xval[1] - (xval[1] / 8), LoA_plus + xval[1] * 0.01, 'Mean+1.96SD', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8), LoA_neg + xval[1] * 0.01, 'Mean-1.96SD', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8)+0.35, mean_diff + xval[1] * 0.01, 'Mean', fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8)+0.5, LoA_plus - xval[1] * 0.025-0.3, round(LoA_plus, 1), fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8)+0.5, LoA_neg - xval[1] * 0.025-0.3, round(LoA_neg, 1), fontdict=FONT_DICT_Mid)
        ax.text(xval[1] - (xval[1] / 8)+0.5, mean_diff - xval[1] * 0.025-0.3, round(mean_diff, 1), fontdict=FONT_DICT_Mid)
        ax.set_xlabel('Mean of ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        ax.set_ylabel('Difference between ' + var1 + ' and ' + var2, fontdict=FONT_DICT_LARGE)
        format_plot()

    def std_show_each_pair_ieee(ax, y_true, y_pred, sub_id):
        si_true = y_true
        si_pred = y_pred

        format_plot()

        sc = [0.]
        sc[0] = ax.scatter(si_true, si_pred, s=70, marker='.', alpha=0.7, edgecolors='none', color='green')

        min_value = min(min(si_true), min(si_pred))
        max_value = max(max(si_true), max(si_pred))
        min_value = 3
        max_value = 10.1

        std_bar_ticks(min_value, max_value)

        coef = np.polyfit(si_true, si_pred, 1)
        poly1d_fn = np.poly1d(coef)
        black_line, = plt.plot([min_value, max_value], poly1d_fn([min_value, max_value]), color='black',
                               linewidth=LINE_WIDTH)
        # ax = plt.gca()
        scc = spearmanr(si_true, si_pred)[0]
        mae = mean_absolute_error(si_true, si_pred)
        # rmse = 100 * np.sqrt(mean_squared_error(si_true, si_pred))
        # R2 = r2_score(si_true, si_pred)
        ax.text(0.7, 0.135, 'MAE = {:4.1f}'.format(0.8), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.7, 0.08, 'SCC = {:4.2f}'.format(0.63), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        # ax.text(0.6, 0.08, '$R^2$ = {:4.2f}'.format(R2), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.7, 0.03, '$y$ = {:4.1f}$x$ + {:4.1f}'.format(coef[0], coef[1]),
                fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        plt.tight_layout(rect=[0, 0, 1, 1])
        # ax.legend([sc[0], sc[1], black_line], ['Subject A', 'Subject B', 'Regression Line'], fontsize=FONT_SIZE_Mid,
        #            frameon=False, bbox_to_anchor=(0.5, 0.99))
        ax.legend([sc[0], black_line], ['Healthy', 'Regression Line'], fontsize=FONT_SIZE_LARGE,
                   frameon=False, bbox_to_anchor=(0.51, 0.99))
        # ax.text(0.1, 0.85, 'Ataxia', fontdict=FONT_DICT_LARGE, transform=ax.transAxes)
        print('Fold {}, SCC: {:.2f}, R2: {:.2f}'.format(sub_id, scc, r2_score(si_true, si_pred)))
        # print('The RMSE and $R^2$ of this subject were {:.1f}\% and {:.2f}'.format(rmse, R2))
        # t0 = time.time()
        # save_fig('f1' + str(sub_id) + '--' + str(t0), 600)


    # data preparation
    path = 'exports/new_results/new_healthy_r_t'
    # path = 'exports/new_results/new_results without pretrained weights'
    # folds = [0, 1, 2, 3, 4, 5]
    folds = [0, 1, 2, 3, 4, 5, 6, 7]  # [0, 1, 2, 3, 4, 5]
    # epoch = 150
    epoch = 30  # 150
    variability_true = []
    variability_pred = []
    for i_fold in folds:
        pattern = str(i_fold) + '-' + str(epoch)
        sub_paths = finder(pattern, path)
        for sub_path in sub_paths:
            # print(sub_path)
            sub_data = pd.read_csv(sub_path, header=None)
            sub_y_true = sub_data.iloc[:, 0]
            sub_y_pred = sub_data.iloc[:, 1]
            print(len(sub_y_true))
            variability_true.append(np.std(sub_y_true))
            variability_pred.append(np.std(sub_y_pred))
            # variability_true.append(np.std(sub_y_true) / np.mean(sub_y_true) * 100)
            # variability_pred.append(np.std(sub_y_pred) / np.mean(sub_y_pred) * 100)

    print(np.mean(np.array(variability_true)), np.std(np.array(variability_true)))

    rc('font', family='Arial')
    fig = plt.figure(figsize=(12, 6))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6])

    # draw scatter
    std_show_each_pair_ieee(fig.add_subplot(gs[1, 0]), variability_true, variability_pred, sub_path)

    # draw bland-altman
    std_bland_altman_ieee(fig.add_subplot(gs[1, 1]), variability_true, variability_pred,
                     'Ground Truth', 'Model Estimation', -10, 40)

    plt.tight_layout(rect=[0, 0, 1, 1])
    # ax.text(0.4, 0.85, 'Healthy', fontdict=FONT_DICT_LARGE, transform=ax.transAxes)
    # ax.legend(['Healthy'], fontsize=FONT_SIZE_LARGE,
    #           frameon=False, bbox_to_anchor=(0.6, 0.99))
    plt.savefig('exports/results/healthy_cv_ieee_ba.pdf', dpi=800)
    plt.show()


def plot_augmentation_mae():
    from utils import finder
    from sklearn.metrics import mean_absolute_error as mae
    from scipy.stats import ttest_rel
    from matplotlib import rc
    import matplotlib.lines as lines

    def format_ticks(x_locs):
        ax = plt.gca()
        ax.set_ylabel('Mean Absolute Error (cm)', fontdict=FONT_DICT_LARGE)
        ax.text(1.6, -0.8, 'Data Augmentation', fontdict=FONT_DICT_LARGE)
        ax.xaxis.set_label_coords(0.47, -0.08)
        ax.set_ylim(0, 7.0)
        ax.set_yticks(range(0, 7, 1))
        ax.set_yticklabels(range(0, 7, 1), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-0.1, 5)
        ax.set_xticks(x_locs)
        ax.set_xticklabels(['w/o', 'R', 'T', 'R + T'], fontdict=FONT_DICT_LARGE)
        ax.xaxis.set_tick_params(width=0)

    def format_errorbar_cap(caplines, size=15):
        for i_cap in range(1):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(size)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    def draw_signifi_sign(mean_, std_, bar_locs, args):
        x_offset = 0.2
        y_top = max([a + b for a, b in zip(mean_, std_)])
        y_top_sub = max([a + b for a, b in zip(mean_, std_)])

        # diff_line_0x = [bar_locs[0], bar_locs[0],
        #                 bar_locs[1]-0.05, bar_locs[1]-0.05]
        # diff_line_0y = [mean_[0] + std_[0] + x_offset, y_top_sub + 0.4, y_top_sub + 0.4,
        #                 mean_[1] + std_[1] + x_offset]
        # plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        # plt.text((bar_locs[0]) * 0.5 + (
        #     bar_locs[1]) * 0.5 - 0.1,
        #          y_top_sub + 0.4 - 0.1,
        #          '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=30)

        diff_line_0x = [bar_locs[1]+0.05, bar_locs[1]+0.05,
                        bar_locs[3], bar_locs[3]]
        diff_line_0y = [mean_[1] + std_[1] + x_offset * 6.5, y_top + 0.9, y_top + 0.9,
                        mean_[3] + std_[3] + x_offset * 1]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text((bar_locs[1]) * 0.5 + (
            bar_locs[3]) * 0.5 - 0.25,
                 y_top + 0.9 - 0.1,
                 '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=30)

        diff_line_0x = [bar_locs[0], bar_locs[0],
                        bar_locs[2], bar_locs[2]]
        diff_line_0y = [mean_[0] + std_[0] + x_offset, y_top + 0.3, y_top + 0.3,
                        mean_[2] + std_[2] + x_offset*1]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text((bar_locs[0]) * 0.5 + (
            bar_locs[2]) * 0.5 - 0.25,
                 y_top + 0.3 - 0.1,
                 '**', fontdict={'fontname': 'Times New Roman'}, color='black', size=30)

        diff_line_0x = [bar_locs[0], bar_locs[0],
                        bar_locs[3], bar_locs[3]]
        diff_line_0y = [mean_[0] + std_[0] + x_offset, y_top + 1.3, y_top + 1.3,
                        mean_[3] + std_[3] + x_offset * 7]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text((bar_locs[0]) * 0.5 + (
            bar_locs[3]) * 0.5 - 0.25,
                 y_top + 1.3 - 0.1,
                 '**', fontdict={'fontname': 'Times New Roman'}, color='black', size=30)

    # path
    path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'

    # data prepare #########################################################
    mae_data = pd.read_excel(path, sheet_name='MAE')
    mae_data = mae_data.iloc[:, 1:]
    without = mae_data["without"]
    time_warp = mae_data["t"]
    rotation = mae_data["r"]
    rotation_time_warp = mae_data["r+t"]

    _mean = [np.mean(without), np.mean(rotation), np.mean(time_warp), np.mean(rotation_time_warp)]
    _std = [np.std(without), np.std(rotation), np.std(time_warp), np.std(rotation_time_warp)]

    p = []
    for i, pair in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        p.append(ttest_rel(mae_data.iloc[:, pair[0]], mae_data.iloc[:, pair[1]]).pvalue)

    print(p)

    # plot ################################################################
    rc('font', family='Arial')
    # fig = plt.figure(figsize=(9, 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    format_plot()
    x_locs = [1, 2, 3, 4]

    ax.bar([1, 2, 3, 4], _mean, align='center', alpha=0.7, ecolor='blue', capsize=10, width=0.7)
    ebar, caplines, barlinecols = ax.errorbar([1, 2, 3, 4], _mean, _std, capsize=0, ecolor='black',
                                              fmt='none',
                                              lolims=True, elinewidth=LINE_WIDTH_THICK)
    format_errorbar_cap(caplines, 10)

    format_ticks(x_locs)

    # draw significance sign
    args = {"one_two": True if p[0] < 0.05 else False,
            "one_three": True if p[1] < 0.05 else False,
            "one_four": True if p[2] < 0.05 else False,
            "two_three": True if p[3] < 0.05 else False,
            "two_four": True if p[4] < 0.05 else False,
            "three_four": True if p[5] < 0.05 else False}

    draw_signifi_sign(_mean, _std, x_locs, args)

    plt.tight_layout(rect=[0.0, 0.0, 1, 1])
    plt.savefig('exports/results/' + 'new_augmentation_mae' + '.pdf', dpi=800)
    plt.show()

    return


def plot_augmentation_mae_healthy_ataxia():
    from utils import finder
    from sklearn.metrics import mean_absolute_error as mae
    from scipy.stats import ttest_rel
    from matplotlib import rc
    import matplotlib.lines as lines
    import matplotlib.patches as mpatches

    def paired_t_test(pre, post, var1, var2):
        if shapiro(pre).pvalue > 0.05 and shapiro(post).pvalue > 0.05:
            print("p value of {} vs. {}: ".format(var1, var2), ttest_rel(pre, post).pvalue)
        else:
            print("p value of {} vs. {}: ".format(var1, var2), wilcoxon(pre, post).pvalue)

    def format_errorbar_cap(caplines, size=15):
        for i_cap in range(1):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(size)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    def draw_signifi_sign(mean_, std_, bar_locs, args):
        x_offset = 0.2
        y_top = max([a + b for a, b in zip(mean_, std_)])
        y_top_sub = max([a + b for a, b in zip(mean_, std_)])

        diff_line_0x = [bar_locs[0], bar_locs[0],
                        bar_locs[1]-0.05, bar_locs[1]-0.05]
        diff_line_0y = [mean_[0] + std_[0] + x_offset, y_top_sub + 0.4, y_top_sub + 0.4,
                        mean_[1] + std_[1] + x_offset]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text((bar_locs[0]) * 0.5 + (
            bar_locs[1]) * 0.5 - 0.1,
                 y_top_sub + 0.4 - 0.1,
                 '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=30)

    def plot_mae(ax):
        # path
        path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'

        # ataxia data prepare #########################################################
        mae_data = pd.read_excel(path, sheet_name='MAE')
        mae_data = mae_data.iloc[:, 1:]
        without = mae_data["without"]
        rotation_time_warp = mae_data["r+t"]

        # healthy data prepare #########################################################
        # mae
        h_without = [2.721865, 2.088152, 2.903751, 3.242875, 4.347702, 2.505275, 2.351918, 2.406009]
        h_r_t = [3.175241, 2.885617, 3.1398, 3.029848, 3.695846, 2.576486, 2.076475, 2.479528]

        # scc
        # h_without = [0.84287, 0.871596, 0.868171, 0.83309, 0.889168, 0.783044, 0.951089, 0.904874]
        # h_r_t = [0.921462, 0.652806, 0.831309, 0.792998, 0.911725, 0.86476, 0.925347, 0.902563]

        _mean = [np.mean(without), np.mean(rotation_time_warp), 0., np.mean(h_without), np.mean(h_r_t)]
        _std = [np.std(without), np.std(rotation_time_warp), 0., np.std(h_without), np.std(h_r_t)]

        p1 = paired_t_test(without, rotation_time_warp, 'without', 'rotation_time_warp')
        p2 = paired_t_test(h_without, h_r_t, 'h_without', 'h_r_t')
        print(p1, p2)

        x_locs = [1, 2, 3, 4, 5]

        ax.bar([1, 2, 3, 4, 5], _mean, align='center', alpha=0.7,
               color=['lightblue', 'darkblue', 'blue', 'lightgreen', 'darkgreen'], capsize=10, width=0.7)

        # ax.legend(['Ataxia', 'Healthy'], fontsize=FONT_SIZE_LARGE_MORE, frameon=False, bbox_to_anchor=(0.5, 0.99))
        ebar, caplines, barlinecols = ax.errorbar([1, 2, 3, 4, 5], _mean, _std, capsize=0, ecolor='black',
                                                  fmt='none',
                                                  lolims=True, elinewidth=LINE_WIDTH_THICK)
        format_errorbar_cap(caplines, 10)

        ax.vlines([3], 0, 5.5, linestyles='dashed', colors='black')

        ax.set_ylabel('Mean Absolute Error (cm)', fontdict=FONT_DICT_LARGE)
        # ax.text(1.9, -0.8, 'Data Augmentation', fontdict=FONT_DICT_LARGE)
        # ax.xaxis.set_label_coords(0.47, -0.08)
        ax.set_ylim(0, 6.0)
        ax.set_yticks(range(0, 6, 1))
        ax.set_yticklabels(range(0, 6, 1), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(0, 6)
        ax.set_xticks(x_locs)
        # ax.set_xticklabels(['w/o', 'R + T', '', 'w/o', 'R + T'], fontdict=FONT_DICT_LARGE)
        ax.xaxis.set_tick_params(width=0)
        ax.xaxis.set_major_locator(plt.NullLocator())

        draw_signifi_sign(_mean, _std, x_locs, None)
        format_plot()

    def plot_rmse(ax):
        # path
        path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'

        # ataxia data prepare #########################################################
        mae_data = pd.read_excel(path, sheet_name='RMSE')
        mae_data = mae_data.iloc[:, 1:]
        without = mae_data["without"]
        rotation_time_warp = mae_data["r+t"]

        # healthy data prepare #########################################################
        # rmse
        h_without = [3.297281, 2.576389, 3.527732, 3.894709, 5.047336, 3.137501, 2.829655, 3.20968]
        h_r_t = [3.790714, 3.515103, 3.808829, 3.643453, 4.501465, 3.153955, 2.671232, 3.218106]

        _mean = [np.mean(without), np.mean(rotation_time_warp), 0., np.mean(h_without), np.mean(h_r_t)]
        _std = [np.std(without), np.std(rotation_time_warp), 0., np.std(h_without), np.std(h_r_t)]

        p1 = paired_t_test(without, rotation_time_warp, 'without', 'rotation_time_warp')
        p2 = paired_t_test(h_without, h_r_t, 'h_without', 'h_r_t')
        print(p1, p2)

        x_locs = [1, 2, 3, 4, 5]

        ax.bar([1, 2, 3, 4, 5], _mean, align='center', alpha=0.7,
               color=['lightblue', 'darkblue', 'blue', 'lightgreen', 'darkgreen'], capsize=10, width=0.7)

        # ax.legend(['Ataxia', 'Healthy'], fontsize=FONT_SIZE_LARGE_MORE, frameon=False, bbox_to_anchor=(0.5, 0.99))
        ebar, caplines, barlinecols = ax.errorbar([1, 2, 3, 4, 5], _mean, _std, capsize=0, ecolor='black',
                                                  fmt='none',
                                                  lolims=True, elinewidth=LINE_WIDTH_THICK)
        format_errorbar_cap(caplines, 10)

        ax.vlines([3], 0, 5.5, linestyles='dashed', colors='black')

        ax.set_ylabel('Root Mean Square Error (cm)', fontdict=FONT_DICT_LARGE)
        # ax.text(1.9, -0.8, 'Data Augmentation', fontdict=FONT_DICT_LARGE)
        # ax.xaxis.set_label_coords(0.47, -0.08)
        ax.set_ylim(0, 6.0)
        ax.set_yticks(range(0, 6, 1))
        ax.set_yticklabels(range(0, 6, 1), fontdict=FONT_DICT_LARGE)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.set_xlim(0, 6)
        ax.set_xticks(x_locs)
        # ax.set_xticklabels(['w/o', 'R + T', '', 'w/o', 'R + T'], fontdict=FONT_DICT_LARGE)
        ax.xaxis.set_tick_params(width=0)
        ax.xaxis.set_major_locator(plt.NullLocator())

        draw_signifi_sign(_mean, _std, x_locs, None)
        format_plot()

    # plot ################################################################
    rc('font', family='Arial')
    fig = plt.figure(figsize=(12, 5))
    # plt.rcParams['xtick.labelsize'] = 12
    # plt.rcParams['ytick.labelsize'] = 12
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6])

    plot_mae(fig.add_subplot(gs[1, 0]))
    plot_rmse(fig.add_subplot(gs[1, 1]))

    legend_labels = [
        mpatches.Patch(color='lightblue', label='Ataxia'),
        mpatches.Patch(color='darkblue', label='Ataxia w/ Data Augmentation'),
        mpatches.Patch(color='lightgreen', label='Healthy'),
        mpatches.Patch(color='darkgreen', label='Healthy w/ Data Augmentation')
    ]

    # Adding legend with custom labels
    fig.legend(handles=legend_labels, fontsize=FONT_SIZE_LARGE, ncol=4, frameon=False,
               bbox_to_anchor=(0.99, 0.15))

    plt.tight_layout(rect=[0, 0.10, 1, 1.15])
    plt.savefig('exports/results/' + 'healthy_ataxia_new_augmentation_rmse' + '.pdf', dpi=800)
    plt.show()

    return


def get_all_scores():
    from utils import finder
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score
    from scipy.stats import ttest_rel
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    from math import sqrt
    from sklearn.metrics import mean_absolute_percentage_error as mape
    import csv

    def _get_results(y_test, y_pred):
        R2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        RMSE = sqrt(mse(y_test, y_pred, multioutput='uniform_average'))
        MAPE = mape(y_test, y_pred)
        MAE = mae(y_test, y_pred)
        errors = y_test - y_pred
        mean_error = np.mean(errors, axis=0)

        pearsonr_cc = pearsonr(y_test, y_pred)[0]
        spearmanr_cc = spearmanr(y_test, y_pred)[0]
        return [R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc]

    # data preparation
    path = 'exports/new_results/with pretrained'
    folds = [0, 1, 2, 3, 4, 5]
    epochs = [146, 147, 148, 149, 150]
    subjects = {0: ["26062023", "28062023"], 1: ["507202310", "507202311"], 2: ["1207202310", "1207202311"],
                3: ["2407202313", "2607202310"], 4: ["2607202311", "3107202314"], 5: ["3107202315", "2407202314"]}

    results = []
    for epoch in epochs:
        for i_fold in folds:
            y_true = []
            y_pred = []
            for sub in subjects[i_fold]:
                pattern = sub + '-' + str(i_fold) + '-' + str(epoch)
                sub_paths = finder(pattern, path)
                for sub_path in sub_paths:
                    print(sub_path)
                    sub_data = pd.read_csv(sub_path, header=None)
                    sub_y_true = sub_data.iloc[:, 0]
                    sub_y_pred = sub_data.iloc[:, 1]
                    y_true.append(sub_y_true)
                    y_pred.append(sub_y_pred)
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            results.append(_get_results(y_true, y_pred))

    # save results as csv
    with open('exports/results/all_scores_pretrained.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['R2', 'RMSE', 'MAPE', 'MAE', 'mean_error', 'pearsonr_cc', 'spearmanr_cc'])
        for result in results:
            writer.writerow(result)


def compare_add_or_replacing_augmentation():
    from scipy.stats import ttest_rel, wilcoxon, shapiro

    def paired_t_test(pre, post):
        if shapiro(pre).pvalue > 0.05 and shapiro(post).pvalue > 0.05:
            print("normally distributed", ttest_rel(pre, post).pvalue)
        else:
            print("not normally distributed", wilcoxon(pre, post).pvalue)

    path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'

    # data prepare #########################################################
    metrics = ['SCC', 'R2', 'MAE', 'RMSE']

    mae_data = pd.read_excel(path, sheet_name='MAE')
    without = mae_data["without"]
    time_warp = mae_data["t"]
    rotation = mae_data["r"]
    rotation_time_warp = mae_data["r+t"]

    replacing_data = pd.read_excel(path, sheet_name='R-MAE')
    r_time_warp = replacing_data["t"]
    r_rotation = replacing_data["r"]
    r_rotation_time_warp = replacing_data["r+t"]

    # Substituting the raw data with augmented data A
    paired_t_test(without, r_rotation)
    paired_t_test(without, r_time_warp)
    paired_t_test(without, r_rotation_time_warp)

    # Supplementing the raw data with augmented data B
    paired_t_test(without, rotation)
    paired_t_test(without, time_warp)
    paired_t_test(without, rotation_time_warp)
    # print(np.mean(without), np.mean(rotation), np.mean(time_warp), np.mean(rotation_time_warp))

    # Comparing A and B
    paired_t_test(rotation, r_rotation)
    paired_t_test(time_warp, r_time_warp)
    paired_t_test(rotation_time_warp, r_rotation_time_warp)

    # paired_t_test(rotation_time_warp, time_warp)
    # paired_t_test(rotation_time_warp, rotation)
    # paired_t_test(rotation_time_warp, without)
    # paired_t_test(time_warp, rotation)
    # paired_t_test(time_warp, without)
    # paired_t_test(rotation, without)


def compare_add_or_replacing_augmentation_HEALTHY():
    from scipy.stats import ttest_rel, wilcoxon, shapiro

    def paired_t_test(pre, post):
        flag = True
        if flag:
            print(wilcoxon(pre, post).pvalue)
        else:
            if shapiro(pre).pvalue > 0.05 and shapiro(post).pvalue > 0.05:
                print("normally distributed", ttest_rel(pre, post).pvalue)
            else:
                print("not normally distributed", wilcoxon(pre, post).pvalue)

    path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'

    # data prepare #########################################################
    metrics = ['SCC', 'R2', 'MAE', 'RMSE']

    mae_data = pd.read_excel(path, sheet_name='H-RMSE')
    without = mae_data["without"]
    time_warp = mae_data["t+without"]
    rotation = mae_data["r+without"]
    rotation_time_warp = mae_data["r+t+without"]
    replacing_time_warp = mae_data["t"]
    replacing_rotation = mae_data["r"]
    replacing_rotation_time_warp = mae_data["r+t"]

    # Substituting the raw data with augmented data A
    paired_t_test(without, replacing_rotation)
    paired_t_test(without, replacing_time_warp)
    paired_t_test(without, replacing_rotation_time_warp)

    # Supplementing the raw data with augmented data B
    paired_t_test(without, rotation)
    paired_t_test(without, time_warp)
    paired_t_test(without, rotation_time_warp)
    # print(np.mean(without), np.mean(rotation), np.mean(time_warp), np.mean(rotation_time_warp))

    # Comparing A and B
    paired_t_test(rotation, replacing_rotation)
    paired_t_test(time_warp, replacing_time_warp)
    paired_t_test(rotation_time_warp, replacing_rotation_time_warp)

    # paired_t_test(rotation_time_warp, time_warp)
    # paired_t_test(rotation_time_warp, rotation)
    # paired_t_test(rotation_time_warp, without)
    # paired_t_test(time_warp, rotation)
    # paired_t_test(time_warp, without)
    # paired_t_test(rotation, without)


def compare_augmented_dataset_with_pretrained():
    from scipy.stats import ttest_rel, wilcoxon, shapiro

    def paired_t_test(pre, post):
        if shapiro(pre).pvalue > 0.05 and shapiro(post).pvalue > 0.05:
            print("normally distributed", ttest_rel(pre, post).pvalue)
        else:
            print("not normally distributed", wilcoxon(pre, post).pvalue)

    path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'

    # data prepare #########################################################
    metrics = ['SCC', 'R2', 'MAE', 'RMSE']
    for m in metrics:
        data = pd.read_excel(path, sheet_name=m)["r+t"]
        pretrained_data = pd.read_excel(path, sheet_name='pretrained')[m]
        paired_t_test(data, pretrained_data)


def icc_calculation():
    import pingouin as pg
    from utils import finder

    path = 'exports/new_results/new_results without pretrained weights'
    # path = 'exports/new_results/new_healthy_r_t'
    folds = [0, 1, 2, 3, 4, 5]
    epoch = 150
    icc = []
    for i_fold in folds:
        pattern = str(i_fold) + '-' + str(epoch)
        sub_paths = finder(pattern, path)
        fold_y_true = []
        fold_y_pred = []
        for sub_path in sub_paths:
            sub_data = pd.read_csv(sub_path, header=None)
            sub_y_true = sub_data.iloc[:, 0]
            sub_y_pred = sub_data.iloc[:, 1]

            fold_y_true.extend(list(sub_y_true))
            fold_y_pred.extend(list(sub_y_pred))

        data_true = pd.DataFrame({'Steps': list(range(len(fold_y_true))), 'Methods': ['True'] * len(fold_y_true),
                                  'Step Width': fold_y_true})
        data_pred = pd.DataFrame({'Steps': list(range(len(fold_y_pred))), 'Methods': ['Pred'] * len(fold_y_pred),
                                    'Step Width': fold_y_pred})
        data = pd.concat([data_true, data_pred])

        results = pg.intraclass_corr(data=data, targets='Steps', raters='Methods', ratings='Step Width')
        results = results.set_index('Description')
        icc.append(results.loc['Single random raters', 'ICC'])
    print(np.mean(icc), np.std(icc))

    # 0.75±0.09 for healthy subjects, 0.78±0.09 for ataxia patients


def icc_table2():
    from scipy.stats import ttest_rel, wilcoxon, shapiro

    def paired_t_test(pre, post):
        if shapiro(pre).pvalue > 0.05 and shapiro(post).pvalue > 0.05:
            print("normally distributed", ttest_rel(pre, post).pvalue)
        else:
            print("not normally distributed", wilcoxon(pre, post).pvalue)

    # data prepare #########################################################
    path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'
    icc = pd.read_excel(path, sheet_name='ATAXIA_ICC')

    without = icc["without"][4::5]
    time_warp = icc["t"][4::5]
    rotation = icc["r"][4::5]
    rotation_time_warp = icc["r+t"][4::5]

    r_time_warp = icc["t+without"][4::5]
    r_rotation = icc["r+without"][4::5]
    r_rotation_time_warp = icc["r+t+without"][4::5]

    # Substituting the raw data with augmented data A
    paired_t_test(without, r_rotation)
    paired_t_test(without, r_time_warp)
    paired_t_test(without, r_rotation_time_warp)

    # Supplementing the raw data with augmented data B
    paired_t_test(without, rotation)
    paired_t_test(without, time_warp)
    paired_t_test(without, rotation_time_warp)
    # print(np.mean(without), np.mean(rotation), np.mean(time_warp), np.mean(rotation_time_warp))

    # Comparing A and B
    paired_t_test(rotation, r_rotation)
    paired_t_test(time_warp, r_time_warp)
    paired_t_test(rotation_time_warp, r_rotation_time_warp)


def reduce_training_dataset():
    # data prepare #########################################################
    path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'
    rmse = pd.read_excel(path, sheet_name='ATAXIA_RMSE_vs_Trainingdatasize')

    rmse_mean = rmse.mean(axis=0)[1:]
    rmse_mean = np.array(rmse_mean)[::-1]

    # plot ################################################################
    rc('font', family='Arial')
    fig = plt.figure(figsize=(9, 6))
    # fig, ax = plt.subplots(figsize=(6, 6))
    format_plot()

    tick_positions = list(range(1, 12))

    plt.plot(np.array(tick_positions), rmse_mean, marker='o', markersize=6, linewidth=LINE_WIDTH, linestyle='--', color='blue')
    plt.grid(axis='y')
    plt.xlabel('Training Data Size', fontdict=FONT_DICT_LARGE)

    tick_labels = ['5%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
    plt.xticks(tick_positions, tick_labels, fontdict=FONT_DICT_LARGE)

    plt.ylabel('Root Mean Square Error (cm)', fontdict=FONT_DICT_LARGE)
    plt.yticks(fontsize=FONT_SIZE_LARGE)
    plt.ylim(3.5, 5.5)

    plt.tight_layout(rect=[0.0, 0.0, 1, 1])
    # plt.savefig('exports/results/' + 'new_augmentation_mae' + '.pdf', dpi=800)
    plt.show()

    return


def reduce_training_dataset_ataxia_and_healthy():
    # data prepare #########################################################
    path = '../peter_Israel/Meeting/new_ataxia_results_new_definition.xlsx'
    a_rmse = pd.read_excel(path, sheet_name='ATAXIA_RMSE_vs_Trainingdatasize')

    a_rmse_mean = a_rmse.mean(axis=0)[1:]
    a_rmse_mean = np.array(a_rmse_mean)[::-1]

    h_rmse = pd.read_excel(path, sheet_name='HEALTHY_RMSE_vs_Trainingdatasiz')
    h_rmse_mean = h_rmse.mean(axis=0)[1:]
    h_rmse_mean = np.array(h_rmse_mean)[::-1]

    # plot ################################################################
    rc('font', family='Arial')
    fig = plt.figure(figsize=(9, 6))
    # fig, ax = plt.subplots(figsize=(6, 6))
    format_plot()

    tick_positions = list(range(1, 8))
    used = [0, 1, 3, 5, 7, 9, 10]

    plt.plot(np.array(tick_positions), a_rmse_mean[used], marker='o', markersize=6, linewidth=LINE_WIDTH, linestyle='--', color='blue')
    plt.plot(np.array(tick_positions), h_rmse_mean[used], marker='o', markersize=6, linewidth=LINE_WIDTH, linestyle='--', color='green')

    plt.grid(axis='y')
    plt.xlabel('Training Data Size', fontdict=FONT_DICT_LARGE)

    tick_labels = ['5%', '10%', '30%', '50%', '70%', '90%', '100%']
    plt.xticks(tick_positions, tick_labels, fontdict=FONT_DICT_LARGE)

    plt.ylabel('Root Mean Square Error (cm)', fontdict=FONT_DICT_LARGE)
    plt.yticks(fontsize=FONT_SIZE_LARGE)
    plt.ylim(3.2, 5.5)
    plt.legend(['Ataxia', 'Healthy'], fontsize=FONT_SIZE_LARGE, frameon=False, bbox_to_anchor=(0.9, 0.99))

    plt.tight_layout(rect=[0.0, 0.0, 1, 1])
    plt.savefig('exports/results/' + 'reduce_training_data_size' + '.pdf', dpi=800)
    plt.show()

    return





if __name__ == "__main__":
    figure = 'f13b'
    if figure == 'f1':
        plot_mae_sara_speed()
    if figure == 'f2':
        plot_scatter_bland_altman()
    if figure == 'f2b':
        plot_scatter_bland_altman_ieee_one_sample()
    if figure == 'f2c':
        plot_scatter_bland_altman_ieee_all_sample()
    if figure == 'f3':
        plot_box_of_each_subject()
    if figure == 'f4':
        plot_step_width_variability()
    if figure == 'f4b':
        plot_step_width_variability_ieee()
    if figure == 'f5':
        plot_augmentation_mae()
    if figure == 'f6':
        get_all_scores()
    if figure == 'f7':
        compare_add_or_replacing_augmentation()
    if figure == 'f8':
        compare_augmented_dataset_with_pretrained()
    if figure == 'f9':
        icc_calculation()
    if figure == 'f10':
        plot_augmentation_mae_healthy_ataxia()
    if figure == 'f11':
        compare_add_or_replacing_augmentation_HEALTHY()
    if figure == 'f12':
        icc_table2()
    if figure == 'f13':
        reduce_training_dataset()
    if figure == 'f13b':
        reduce_training_dataset_ataxia_and_healthy()


