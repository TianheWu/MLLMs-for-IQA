import numpy as np
import random
import os
import torch
import math

from scipy.optimize import curve_fit
from itertools import combinations
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_srcc(pred, mos):
    srcc, _ = spearmanr(pred, mos)
    return srcc


def calculate_plcc(pred, mos):
    plcc, _ = pearsonr(pred, mos)
    return plcc


def calculate_mse(pred, mos):
    return np.mean((pred - mos) ** 2)


def fit_curve(x, y, curve_type='logistic_4params'):
    r'''Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.
    The function with 4 params is more commonly used.
    The 5 params function takes from DBCNN:
        - https://github.com/zwx8981/DBCNN/blob/master/dbcnn/tools/verify_performance.m
    '''
    assert curve_type in [
        'logistic_4params', 'logistic_5params'], f'curve type should be in [logistic_4params, logistic_5params], but got {curve_type}.'

    betas_init_4params = [np.max(y), np.min(y), np.mean(x), np.std(x) / 4.]

    def logistic_4params(x, beta1, beta2, beta3, beta4):
        yhat = (beta1 - beta2) / (1 + np.exp(- (x - beta3) / beta4)) + beta2
        return yhat

    betas_init_5params = [10, 0, np.mean(y), 0.1, 0.1]

    def logistic_5params(x, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1. / (1 + np.exp(beta2 * (x - beta3)))
        yhat = beta1 * logistic_part + beta4 * x + beta5
        return yhat

    if curve_type == 'logistic_4params':
        logistic = logistic_4params
        betas_init = betas_init_4params
    elif curve_type == 'logistic_5params':
        logistic = logistic_5params
        betas_init = betas_init_5params

    betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=10000)
    yhat = logistic(x, *betas)
    return yhat


def contains_nan(lst):
    return any(math.isnan(x) for x in lst if isinstance(x, float))


def cal_metric_from_same_content(file_path):
    ref_pred_dict, ref_gt_dict = {}, {}
    with open(file_path, 'r') as txt_file:
        for line in txt_file:
            try:
                ref, dis, pred_score, gt_score = line.split()
                if ref not in ref_pred_dict:
                    ref_pred_dict[ref] = [float(pred_score)]
                else:
                    ref_pred_dict[ref].append(float(pred_score))

                if ref not in ref_gt_dict:
                    ref_gt_dict[ref] = [float(gt_score)]
                else:
                    ref_gt_dict[ref].append(float(gt_score))
            except:
                pass
    
    srcc_list, plcc_list = [], []
    for key, value in ref_pred_dict.items():
        x, y = ref_pred_dict[key], ref_gt_dict[key]

        fitted_x = fit_curve(x, y)
        if contains_nan(fitted_x):
            fitted_x = x
        
        cur_srcc = calculate_srcc(fitted_x, y)
        cur_plcc = calculate_plcc(fitted_x, y)

        if math.isnan(cur_srcc):
            cur_srcc = 0
        if math.isnan(cur_plcc):
            cur_plcc = 0
        
        srcc_list.append(cur_srcc)
        plcc_list.append(cur_plcc)

    srcc = sum(srcc_list) / len(srcc_list)
    plcc = sum(plcc_list) / len(plcc_list)
    return srcc, plcc


def cal_global_metric(file_path):
    pred_list, gt_list = [], []
    with open(file_path, 'r') as txt_file:
        for line in txt_file:
            ref, dis, pred_score, gt_score = line.split()
            pred_list.append(float(pred_score))
            gt_list.append(float(gt_score))

    fitted_x = fit_curve(pred_list, gt_list)
    if contains_nan(fitted_x) or all(x == fitted_x[0] for x in fitted_x):
        fitted_x = pred_list

    srcc = calculate_srcc(fitted_x, gt_list)
    plcc = calculate_plcc(fitted_x, gt_list)

    return srcc, plcc


def chunk_list(original_list, num):
    return [original_list[i:i + num] for i in range(0, len(original_list), num)]


def transform_ranking_to_pair(arr):
    pairs = list(combinations(arr, 2))
    pairs = [list(pair) for pair in pairs]
    return pairs


def pair2score(
        iqa_type,
        length_name,
        pred_scores,
        name_idx_dict,
        cur_image_name_list,
        input_seed: int=0
    ):

    print("pair name list length {}".format(len(cur_image_name_list)))
    print("pair pred_scores list length {}".format(len(pred_scores)))

    C = np.array([[0 for _ in range(length_name)] for _ in range(length_name)])
    for i in range(len(pred_scores)):
        if iqa_type == "nr":
            idx_1 = name_idx_dict[cur_image_name_list[i][0]]
            idx_2 = name_idx_dict[cur_image_name_list[i][1]]
        
        elif iqa_type == "fr":
            idx_1 = name_idx_dict[cur_image_name_list[i][1]]
            idx_2 = name_idx_dict[cur_image_name_list[i][2]]
        
        C[idx_1][idx_2] += pred_scores[i]
        C[idx_2][idx_1] += 1 - pred_scores[i]
    
    pred_score_list = optimize_score(C=C, seed=input_seed, original_seed=20)
    return pred_score_list


def optimize_score(C, seed=0, original_seed=20):
    np.random.seed(seed)

    C = np.array(C)
    num_images = C.shape[0]

    def objective(S):
        sum_log_diff = np.sum(C * np.log(np.maximum(norm.cdf(S[:, None] - S), 1e-6)))
        sum_squares = np.sum(S ** 2) / 2
        return -(sum_log_diff - sum_squares)

    initial_scores = np.random.rand(num_images)
    constraints = {'type': 'eq', 'fun': lambda S: np.sum(S)}
    result = minimize(objective, initial_scores, constraints=constraints)
    optimized_scores = result.x
    min_score, max_score = np.min(optimized_scores), np.max(optimized_scores)
    scaled_scores = 100 * (optimized_scores - min_score) / (max_score - min_score)
    np.random.seed(original_seed)

    return scaled_scores


def create_cd_groups(
        arr,
        image_pair_num: int=5,
        group_num: int=2,
        shuffle_seed: int=20,
        original_seed: int=20
    ):
    random.seed(shuffle_seed)
    ret_list = []

    for element in arr:
        attempts = 0
        while attempts < image_pair_num:
            cur_group_list = [element]
            cur_group_num = 0
            while cur_group_num < group_num - 1:
                ele = random.choice(arr)
                if ele not in cur_group_list:
                    cur_group_list.append(ele)
                    cur_group_num += 1

            attempts += 1

            # shuffle list
            # random.shuffle(cur_group_list)
            ret_list.append(cur_group_list)

    random.seed(original_seed)
    return ret_list


def create_nr_groups(
        arr,
        image_pair_num: int=5,
        group_num: int=2,
        shuffle_seed: int=20,
        original_seed: int=20
    ):
    random.seed(shuffle_seed)
    ret_list = []

    for element in arr:
        attempts = 0
        # each image group n times
        while attempts < image_pair_num:
            cur_group_list = [element]
            cur_group_num = 0
            while cur_group_num < group_num - 1:
                ele = random.choice(arr)
                if ele not in cur_group_list:
                    cur_group_list.append(ele)
                    cur_group_num += 1

            attempts += 1

            # shuffle list
            # random.shuffle(cur_group_list)
            ret_list.append(cur_group_list)

    random.seed(original_seed)
    return ret_list


def get_path_list(data_path: str):
    image_path_list = []
    for _, _, files in os.walk(data_path):
        for img in files:
            img_path = os.path.join(data_path, img)
            image_path_list.append(img_path)
    return image_path_list


def scale_values_for_nr(dictionary: dict, scale_num: int=100):
    min_val = min(dictionary.values())
    max_val = max(dictionary.values())
    for key in dictionary:
        dictionary[key] = (dictionary[key] - min_val) / (max_val - min_val) * scale_num

    return dictionary


def scale_values_for_fr(input_list: list, scale_num: int=100):
    score_list = [sublist[2] for sublist in input_list]
    min_val = min(score_list)
    max_val = max(score_list)

    for i in range(len(input_list)):
        input_list[i][2] = (input_list[i][2] - min_val) / (max_val - min_val) * scale_num
    
    return input_list


def scale_values_for_fr_sample(input_list: list, scale_num: int=1):
    score_list = [sublist[2] for sublist in input_list]
    min_val = min(score_list)
    max_val = max(score_list)
    for i in range(len(input_list)):
        input_list[i][2] = (input_list[i][2] - min_val) / (max_val - min_val) * scale_num
    
    try:
        std_list = [sublist[3] for sublist in input_list]
        min_val = min(std_list)
        max_val = max(std_list)

        for i in range(len(input_list)):
            input_list[i][3] = (input_list[i][3] - min_val) / (max_val - min_val) * scale_num
        
        print("scaling the std value done.")
    except:
        pass
    
    return input_list


def scale_values_for_nr_sample(dictionary: dict, scale_num: int=100):
    score_list = [value[0] for value in dictionary.values()]
    min_val = min(score_list)
    max_val = max(score_list)
    for key in dictionary:
        dictionary[key][0] = (dictionary[key][0] - min_val) / (max_val - min_val) * scale_num
    
    try:
        std_list = [value[1] for value in dictionary.values()]
        min_val = min(std_list)
        max_val = max(std_list)
        for key in dictionary:
            dictionary[key][1] = (dictionary[key][1] - min_val) / (max_val - min_val) * scale_num
        
        print("scaling the std value done.")
    except:
        pass

    return dictionary


def creat_fr_pairs(
        image_path_list,
        image_name_list,
        n_pairs,
        shuffle_seed: int=20,
        original_seed: int=20
    ):
    random.seed(shuffle_seed)

    ref_path_name_dict, dis_path_name_dict = {}, {}
    for i in range(len(image_path_list)):
        ref_path_name_dict[image_path_list[i][0]] = image_name_list[i][0]
        dis_path_name_dict[image_path_list[i][1]] = image_name_list[i][1]

    path_ref_dis_dict = {}
    for item in image_path_list:
        key = item[0]
        if key not in path_ref_dis_dict:
            path_ref_dis_dict[key] = []
        path_ref_dis_dict[key].append(item[1])

    name_paired_list, path_paired_list = [], []
    for key, values in path_ref_dis_dict.items():
        for i in range(len(values)):

            cur_group_num = 0
            while cur_group_num < n_pairs:
                random_index = random.randint(0, len(values) - 1)
                
                # selected element is not equal to the current value
                if values[random_index] != values[i]:
                    path_paired_list.append([key, values[i], values[random_index]])
                    name_paired_list.append(
                        [
                            ref_path_name_dict[key],
                            dis_path_name_dict[values[i]],
                            dis_path_name_dict[values[random_index]]
                        ])
                    cur_group_num += 1

    random.seed(original_seed)
    return path_paired_list, name_paired_list


def create_fr_groups(
        image_path_list,
        image_name_list,
        fr_image_group_num,
        each_group_num: int=4,
        shuffle_seed: int=20,
        original_seed: int=20
    ):
    random.seed(shuffle_seed)

    ref_path_name_dict, dis_path_name_dict = {}, {}
    for i in range(len(image_path_list)):
        ref_path_name_dict[image_path_list[i][0]] = image_name_list[i][0]
        dis_path_name_dict[image_path_list[i][1]] = image_name_list[i][1]

    path_ref_dis_dict = {}
    for item in image_path_list:
        key = item[0]
        if key not in path_ref_dis_dict:
            path_ref_dis_dict[key] = []
        path_ref_dis_dict[key].append(item[1])

    name_ranking_list, path_ranking_list = [], []
    for key, values in path_ref_dis_dict.items():
        for i in range(len(values)):
            cur_group_num = 0
            while cur_group_num < fr_image_group_num:

                # first is the reference image, second is current assigned distorted image
                cur_selected_path_list = [key, values[i]]
                cur_selected_name_list = [ref_path_name_dict[key], dis_path_name_dict[values[i]]]

                cur_each_group_num = 0
                while cur_each_group_num < each_group_num - 1:
                    random_index = random.randint(0, len(values) - 1)

                    # selected element is not equal to the current value
                    if values[random_index] != values[i]:
                        cur_selected_path_list.append(values[random_index])
                        cur_selected_name_list.append(dis_path_name_dict[values[random_index]])
                        cur_each_group_num += 1
                
                path_ranking_list.append(cur_selected_path_list)
                name_ranking_list.append(cur_selected_name_list)
                cur_group_num += 1
    
    random.seed(original_seed)
    return path_ranking_list, name_ranking_list



