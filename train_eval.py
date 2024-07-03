# -*- coding: utf-8 -*-
# @Time : 2023/3/26 21:29
# @Author : Liang kairan
# @FileName: train_eval.py
# @Software: PyCharm
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from numpy import sqrt
from scipy.stats import pearsonr, spearmanr
from utils import *
# from regression_scatter_plot import show_each_pair
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_model(model, data_loader, optimizer, loss_function, args, type="train", plot=False):

    is_train = False
    if type == "train":
        model.train()
        is_train = True
    else:
        model.eval()

    return_loss = 0.0
    if type == "test":
        test_sub = args["test_set"]
        test_sub = [str(int(x)) for x in list(set(args["test_set"]))]
        y_true_list = []
        y_pred_list = []
        assert test_sub is not None
        RESULTS = {}
        for x in test_sub:
            sca_patients = False
            if sca_patients:
                if x.startswith('0'):
                    x = x[1:]
            RESULTS[x] = {"y_true": [], "y_pred": []}

    plot_path = args["plot_path"]

    with torch.set_grad_enabled(is_train):
        for index, batch in enumerate(data_loader):
            # augmentation during loop
            # inputs, labels, lens = argumentation_during_loop(np.array(batch[0]), np.array(batch[1]), np.array(batch[2]))
            # loop
            inputs = batch[0].float().to(device)
            labels = batch[1].float().to(device)
            lens = batch[2]
            # subs = batch[3]

            # if type == "test":
            #     time_start = time.time()
            y = model(inputs, lens)
            # if type == "test":
            #     time_end = time.time()
            #     # print('time cost', time_end - time_start, 's')
            #     print('Execution time per step:', (time_end - time_start) / len(y), 's')
            # y = model(inputs)  # for Sharifi_CNN

            loss = loss_function(y, labels)

            if type == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            return_loss += float(loss)

            # if plot and index == 0:
            #     fig = go.Figure()
            #     fig.add_trace(go.Scatter(x=y.cpu().detach().numpy().ravel(), y=labels.cpu().detach().numpy().ravel(),
            #                              mode='markers', name='Labels'))
            #     fig.write_html(plot_path + type + '.html', auto_open=False)

            # evaluate model
            if type == "test":
                # subs = batch[3]
                #
                # for i in range(len(subs)):
                #     RESULTS[str(int(subs[i].item()))]["y_true"].append(labels[i].cpu().detach().numpy())
                #     RESULTS[str(int(subs[i].item()))]["y_pred"].append(y[i].cpu().detach().numpy())

                y_true_list.extend(labels.cpu().detach().numpy())
                y_pred_list.extend(y.cpu().detach().numpy())

        if type == "test":
            y_true_array = np.array(y_true_list)
            y_pred_array = np.array(y_pred_list)
            R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc, icc, NAPE = _get_all_scores(y_true_array, y_pred_array, args["i_folder"])
            # temp = []
            # if args["epoch"] > 29:
            #     for key in RESULTS.keys():
            #         y_true_array_each_sub = np.array(RESULTS[key]["y_true"])
            #         y_pred_array_each_sub = np.array(RESULTS[key]["y_pred"])
            #         # t0, t1, t2, t3, t4, t5, t6 = _get_all_scores(y_true_array_each_sub, y_pred_array_each_sub)
            #         # temp.append([t0, t1, t2, t3, t4, t5, t6])
            #
            #         #save y_true_array_each_sub and y_pred_array_each_sub as two columns in a csv file, named as key + epoch
            #         fname = 'exports/new_results/new_healthy_r_t/' + key + '-' +str(args["i_folder"]) + '-' + str(args["epoch"]) + '.csv'
            #         np.savetxt(fname, np.c_[y_true_array_each_sub, y_pred_array_each_sub], delimiter=',')

            # temp = [sum(RESULTS[key]) / len(RESULTS[key]) for key in RESULTS.keys()]
            return return_loss/len(data_loader), R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc, icc, NAPE #, temp

        return_loss = return_loss / len(data_loader)
    return return_loss


def _get_all_scores(y_test, y_pred, i_folder, precision=None):
    R2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    RMSE = sqrt(mse(y_test, y_pred, multioutput='uniform_average'))
    MAPE = mape(y_test, y_pred)
    MAE = mae(y_test, y_pred)
    errors = y_test - y_pred
    mean_error = np.mean(errors, axis=0)
    # CALCULATE normalized absolute percent error (NAPE).
    NAPE = np.mean(np.abs(errors) / np.mean(y_test)) * 100


    pearsonr_cc = pearsonr(y_test, y_pred)[0]
    spearmanr_cc = spearmanr(y_test, y_pred)[0]

    # add ICC
    import pingouin as pg
    data_true = pd.DataFrame({'Steps': list(range(len(y_test))), 'Methods': ['True'] * len(y_test),
                              'Stride Width': y_test})
    data_pred = pd.DataFrame({'Steps': list(range(len(y_pred))), 'Methods': ['Pred'] * len(y_pred),
                              'Stride Width': y_pred})
    data = pd.concat([data_true, data_pred])

    results = pg.intraclass_corr(data=data, targets='Steps', raters='Methods', ratings='Stride Width')
    results = results.set_index('Description')
    icc = results.loc['Single random raters', 'ICC']

    if precision:
        R2 = np.round(R2, precision)
        RMSE = np.round(RMSE, precision)
        MAPE = np.round(MAPE, precision)
        MAE = np.round(MAE, precision)
        mean_error = np.round(mean_error, precision)

    verbose1 = False
    if verbose1:
        print("plotting...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers+text', text=list(range(len(errors))), name='Labels'))
        # 增加 y=x 的直线
        fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='y=x', line=dict(color='red', dash='dash')))

        fig.show()
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=list(range(len(errors))), y=errors, name='Errors'))
        fig1.show()

        import matplotlib as mpl

        LINE_WIDTH = 2
        mpl.rcParams['hatch.linewidth'] = LINE_WIDTH  # previous svg hatch linewidth
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=LINE_WIDTH)
        ax.yaxis.set_tick_params(width=LINE_WIDTH)
        ax.spines['left'].set_linewidth(LINE_WIDTH)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH)

        ax.scatter(y_test, y_pred, s=40, marker='.', alpha=0.7, edgecolors='none', color='blue')
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))

        ax.set_xlim(min_val, max_val)
        FONT_DICT_LARGE = {'fontsize': 16, 'fontname': 'Arial'}
        FONT_DICT_Mid = {'fontsize': 12, 'fontname': 'Arial'}
        ax.set_xticks(np.arange(int(min_val) - 1, int(max_val) + 1, 10), fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('Ground Truth', fontdict=FONT_DICT_LARGE)

        ax.set_ylim(min_val, max_val)
        ax.set_yticks(np.arange(int(min_val) - 1, int(max_val) + 1, 10), fontdict=FONT_DICT_LARGE)
        ax.set_ylabel('Model Estimation', fontdict=FONT_DICT_LARGE)

        # plt.plot([min_val, max_val], [min_val, max_val], color='black', linewidth=LINE_WIDTH)

        coef = np.polyfit(y_test, y_pred, 1)
        poly1d_fn = np.poly1d(coef)
        black_line, = plt.plot([min_val, max_val], poly1d_fn([min_val, max_val]), color='black',
                               linewidth=LINE_WIDTH)
        scc = spearmanr(y_test, y_pred)[0]
        R2 = r2_score(y_test, y_pred)
        ax.text(0.6, 0.135, 'SCC = {:4.2f}'.format(scc), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        ax.text(0.6, 0.08, '$R^2$ = {:4.2f}'.format(R2), fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        # sign = '+' if coef[1] > 0 else '-'
        ax.text(0.6, 0.03, '$y$ = {:4.2f}$x$ + {:4.2f}'.format(coef[0], coef[1]),
                fontdict=FONT_DICT_Mid, transform=ax.transAxes)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig('exports/new_results/' + str(i_folder) + 'ground_truth_vs_prediction.png', dpi=600)
        plt.show()

    return R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc, icc, NAPE

