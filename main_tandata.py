# -*- coding: utf-8 -*-
# @Time : 2023/5/13 18:47
# @Author : Wanghong
# @FileName: main_tandata.py
# @Software: PyCharm

# from sklearn.model_selection import train_test_split
from copy import deepcopy

import numpy as np
from utils import setup_seed
setup_seed(42)
from torch.utils.data import DataLoader
from tantiandata import sequence_tan
from train_eval import *
from my_log import *
from const import *
from load_data import *
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 30
batch_size = 128
lr = 0.001
use_early_stop = False
test_set_sub_num = 2

log = MyLog(tanlog_path, "Step Width Estimation for Healthy Subjects")
log.add_message("lr: {}, batch_sz: {}, epochs : {}".format(lr, batch_size, epochs))

sub_ids = list(range(17))
folder_num = int(np.floor(len(sub_ids) / test_set_sub_num))
myEvaluation = []
LOSS_test = []
fold_models = []
test_sub_list = []
for i_folder in range(folder_num):
    setup_seed(42)
    test_sub_list.append(i_folder)
    if i_folder < folder_num - 1:
        test_sub_ids = sub_ids[test_set_sub_num * i_folder:test_set_sub_num * (i_folder + 1)]
    else:
        test_sub_ids = sub_ids[test_set_sub_num * i_folder:]  # make use of all the left subjects

    train_sub_ids = list(np.setdiff1d(sub_ids, test_sub_ids))
    hyper_vali_sub_ids = random.sample(train_sub_ids, test_set_sub_num)
    # hyper_vali_sub_ids = train_sub_ids[:2]
    hyper_train_sub_ids = list(np.setdiff1d(train_sub_ids, hyper_vali_sub_ids))
    log.add_message("hyper_train_sub_ids: {}, hyper_vali_sub_ids: {}, test_sub_ids : {}".format(hyper_train_sub_ids,
                                                                                                hyper_vali_sub_ids,
                                                                                                test_sub_ids))

    print("generate train_valid_test data")
    train_data, train_label, train_lens, train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs, max_length = sequence_tan(hyper_train_sub_ids, hyper_vali_sub_ids, test_sub_ids)
    train_set, valid_set, test_set = load_data(train_data, train_label, train_lens, train_subs, valid_data, valid_label,
                                               valid_lens, valid_subs, test_data, test_label, test_lens, test_subs)

    train_iterator = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)  # shuffle=True,
    valid_iterator = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)  # TODO batch size changed
    test_iterator = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    model = RNN(max_length=max_length, dropout=0.2).to(device)
    # model = Sharifi_CNN().to(device)
    # model.load_state_dict(torch.load(os.path.join("initial_weights_test.pth")))
    # torch.save(model.state_dict(), os.path.join("initial_weights_test.pth"))

    loss_function = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001, momentum=0.9)
    log.add_message('RNN')
    log.add_message('MAE')
    log.add_message('All 17 subjects')

    LOSS_valid = []
    LOSS_train = []
    LOSS_valid_power = []
    args = {"data_max_": None,
            "data_min_": None,
            "i_folder": i_folder,
            "test_set": test_sub_ids,
            "plot_path": log.dic_path + '/' + str(i_folder) + 'ground_truth_vs_prediction'}
    log.save()

    for epoch in range(1, epochs+1):
        args["epoch"] = epoch
        # train
        train_loss = run_model(model, train_iterator, optimizer, loss_function, args, type="train")
        # if epoch == epochs-1:
        #     train_loss = run_model(model, train_iterator, optimizer, loss_function, args, type="train", plot=True)

        if epoch % 10 == 0:
            print("Epoch {} complete! Average Training loss cm: {:.4f} ".format(epoch, train_loss))
        log.add_message("Epoch: {} | Training loss cm: {:.4f}".format(epoch, train_loss))
        LOSS_train.append(train_loss)

        # validation
        valid_loss = run_model(model, valid_iterator, optimizer, loss_function, args, type="eval")
        if epoch % 10 == 0:
            print("Epoch {} complete! Average Validation loss cm: {:.4f} ".format(epoch, valid_loss))
        LOSS_valid.append(valid_loss)
        # LOSS_valid_power.append(valid_power_loss / valid_total)
        log.add_message("Epoch: {} | Valid loss cm: {:.4f} ".format(epoch, valid_loss))

        # scheduler.step()  # update learning rate

        # early stop Check for improvement
        """if use_early_stop:
            if valid_loss < best_loss:
                best_loss = valid_loss
                epochs_without_improvement = 0
                save_model(model, save_model_path)  # Save the model's parameters

            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= max_epochs_without_improvement:
                    print("Early stopping! No improvement for {} epochs.".format(epochs_without_improvement))
                    break  # Stop the training loop"""
    # setup_seed(24)
    log.save()
    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(LOSS_train))), y=LOSS_train, mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(len(LOSS_valid))), y=LOSS_valid, mode='lines', name='Valid Loss'))

    fig.write_html(log.dic_path + '/' + str(i_folder) + '_loss_curve.html', auto_open=False)

    # test
    test_loss, R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc, icc, NAPE = run_model(model, test_iterator, optimizer, loss_function, args, type="test", plot=True)

    print("Average Test loss cm: {:.4f}".format(test_loss) + " || 这是第{}折".format(i_folder))
    # print([np.round(x, 3) for x in [R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc]])
    log.add_message("Test loss cm: {:.4f} ".format(test_loss))
    log.save()
    LOSS_test.append(test_loss)
    myEvaluation.append([R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc, icc, NAPE])

    # save model from each fold
    # fold_models.append(deepcopy(model.state_dict()))
    # break


print("average test loss of all folds : {:.4f}±{:.4f}".format(np.mean(np.array(LOSS_test)), np.std(np.array(LOSS_test))))
log.add_message("average test loss of all folds : {:.4f}±{:.4f}".format(sum(LOSS_test) / len(LOSS_test), np.std(np.array(LOSS_test))))
log.save()

myEvaluation_array = np.array(myEvaluation)
# print("average R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc across LOSO :" + str([np.round(np.mean(myEvaluation_array[:, x]), 3) for x in list(range(7))]))
log.add_message(str([np.round(np.mean(myEvaluation_array[:, x]), 3) for x in list(range(len(myEvaluation_array[0])))]))

sub_col_name = np.array(test_sub_list).reshape(-1, 1)
myEvaluation_array = np.hstack((sub_col_name, myEvaluation_array))
log.add_csv([["test_sub", "R2", "RMSE", "MAPE", "MAE", "mean_error", "pearsonr_cc", "spearmanr_cc", "icc", "NAPE"]], "myEvaluation")
log.add_csv(myEvaluation_array, "myEvaluation")
log.save()


# save model from each fold
# for i, model_state in enumerate(fold_models):
#     torch.save(model_state, os.path.join(tansave_model_path, f"model_fold_{i+1}.pth"))
