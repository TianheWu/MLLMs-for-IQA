import base64
import requests
import os
import re
import io
import random
import numpy as np
import threading

from PIL import Image
from utils import transform_ranking_to_pair
from utils import chunk_list
from utils import fit_curve
from utils import cal_metric_from_same_content
from utils import creat_fr_pairs
from utils import create_fr_groups
from utils import optimize_score
from utils import create_nr_groups
from utils import contains_nan
from utils import calculate_srcc, calculate_plcc


def gpt4v_single_nr_batch(
        key: str,
        dataset_name: str,
        image_path_list: list,
        image_name_list: list,
        mos_list: list,
        text_prompt: str,
        save_folder_path: str,
        save_file_name: str,
        prompt_pattern_str: str,
        ic_path: str
    ):

    GPT4v = GPT4VSystem(
        key=key, save_folder_path=save_folder_path, save_file_name=save_file_name
    )
    MesModule = MessageModule()
    print("path length {}, name length {}, mos length {}".format(
            len(image_path_list), len(image_name_list), len(mos_list)
        )
    )

    # process input list
    chunked_image_path_list = chunk_list(original_list=image_path_list, num=1)
    chunked_image_name_list = chunk_list(original_list=image_name_list, num=1)
    chunked_mos_list = chunk_list(original_list=mos_list, num=1)

    # saving the score results
    result_file_name = "{}_NR_num-{}_gpt4v_{}.txt".format(
        dataset_name, len(image_path_list), prompt_pattern_str)    
    result_save_path = os.path.join(save_folder_path, result_file_name)
    f2 = open(result_save_path, 'a')

    pred_score_list = []
    for i in range(len(chunked_image_path_list)):
        # need to send single elemnet
        print("Current {} image names {}".format(i + 1, chunked_image_name_list[i]))

        cur_image_path = ic_path + chunked_image_path_list[i]
        re_role, re_content, pred_score, token_num = dialogue_with_check(
            message=None,
            GPT4v=GPT4v,
            MesModule=MesModule,
            text_prompt=text_prompt,
            image_name_list=chunked_image_name_list[i],
            image_path_list=cur_image_path,
            mos_list=chunked_mos_list[i],
            pattern="single"
        )

        # if error happend 10 times
        log_respone(
            folder_path=save_folder_path,
            file_name=save_file_name,
            text_prompt=text_prompt,
            input_content=re_content,
            input_image_names=chunked_image_name_list[i],
            input_mos=chunked_mos_list[i],
            token_num=token_num
        )

        pred_score_list.append(pred_score[0])
        f2.write("{} {} {}\n".format(image_name_list[i], pred_score[0], mos_list[i]))
    
    f2.close()

    fitted_pred_score_list = fit_curve(pred_score_list, mos_list)
    if contains_nan(fitted_pred_score_list) or all(x == fitted_pred_score_list[0] for x in fitted_pred_score_list):
        fitted_pred_score_list = pred_score_list

    pred = np.array(fitted_pred_score_list).reshape(-1)
    gt =  np.array(mos_list).reshape(-1)

    srcc = calculate_srcc(pred, gt)
    plcc = calculate_plcc(pred, gt)
    print("Global SRCC: {}, PLCC: {}".format(srcc, plcc))

    file_path = os.path.join(save_folder_path, save_file_name)
    f = open(file_path, 'a')
    f.write("Global SRCC: {}, PLCC: {}".format(srcc, plcc))
    f.close()


def gpt4v_double_nr_batch(
        key: str,
        dataset_name: str,
        image_path_list: list,
        image_name_list: list,
        mos_list: list,
        text_prompt: str,
        save_folder_path: str,
        save_file_name: str,
        nr_image_pair_num: int,
        prompt_pattern_str: str,
        ic_path: str
    ):

    GPT4v = GPT4VSystem(
        key=key, save_folder_path=save_folder_path, save_file_name=save_file_name
    )
    MesModule = MessageModule()
    print("path length {}, name length {}, mos length {}".format(
            len(image_path_list), len(image_name_list), len(mos_list)
        )
    )

    # process input list
    dis_score = {}
    for i in range(len(image_name_list)):
        dis_score[image_name_list[i]] = mos_list[i]
    
    path_paired_list = create_nr_groups(
        arr=image_path_list,
        image_pair_num=nr_image_pair_num,
        group_num=2
    )
    name_paired_list = create_nr_groups(
        arr=image_name_list,
        image_pair_num=nr_image_pair_num,
        group_num=2
    )

    paired_mos_list = []
    for i in range(len(name_paired_list)):
        paired_mos_list.append(
            [
                dis_score[name_paired_list[i][0]],
                dis_score[name_paired_list[i][1]]
            ]
        )
    
    # distorted image1 is key, compared distorted image2 and compare result is prediction score
    dis1_dis2_score_dict = {}
    for i in range(len(path_paired_list)):
        # need to send single element
        print("Current {} image names {}".format(i + 1, name_paired_list[i]))
        dis1_name, dis2_name = name_paired_list[i][0], name_paired_list[i][1]

        cur_image_path = ic_path + path_paired_list[i]
        re_role, re_content, pred_score, token_num = dialogue_with_check(
            message=None,
            GPT4v=GPT4v,
            MesModule=MesModule,
            text_prompt=text_prompt,
            image_name_list=name_paired_list[i],
            image_path_list=cur_image_path,
            mos_list=paired_mos_list[i],
            pattern="double"
        )
        log_respone(
            folder_path=save_folder_path,
            file_name=save_file_name,
            text_prompt=text_prompt,
            input_content=re_content,
            input_image_names=name_paired_list[i],
            input_mos=paired_mos_list[i],
            token_num=token_num
        )
        
        if dis1_name not in dis1_dis2_score_dict:
            dis1_dis2_score_dict[dis1_name] = [[dis2_name, pred_score[0]]]
        else:
            dis1_dis2_score_dict[dis1_name].append([dis2_name, pred_score[0]])
    
    # saving the score results
    result_file_name = "{}_NR_num-{}_gpt4v_{}.txt".format(
        dataset_name, len(image_path_list), prompt_pattern_str)  
    
    result_save_path = os.path.join(save_folder_path, result_file_name)
    f2 = open(result_save_path, 'a')

    dis_names = list(dis1_dis2_score_dict.keys())
    name_length = len(dis_names)
    C = np.array([[0 for _ in range(name_length)] for _ in range(name_length)])

    # define the name index for optimizing
    idx = 0
    name_idx = {}
    for name in dis_names:
        name_idx[name] = idx
        idx += 1
    
    # print(dis_names)
    for i in range(name_length):
        # find dis1 compared results and objects
        for j in range(len(dis1_dis2_score_dict[dis_names[i]])):
            dis1 = dis_names[i]
            dis2 = dis1_dis2_score_dict[dis_names[i]][j][0]
            compared_res = dis1_dis2_score_dict[dis_names[i]][j][1]

            if dis1 not in name_idx or dis2 not in name_idx:
                continue

            # avoid other answers
            if compared_res == 2:
                C[name_idx[dis1]][name_idx[dis2]] += 1
                C[name_idx[dis2]][name_idx[dis1]] += 1
            elif compared_res == 1:
                C[name_idx[dis1]][name_idx[dis2]] += 1
            elif compared_res == 0:
                C[name_idx[dis2]][name_idx[dis1]] += 1

    pred_score_list = optimize_score(C=C, seed=0, original_seed=20)
    for i in range(name_length):
        f2.write("{} {} {}\n".format(
            dis_names[i], pred_score_list[i], dis_score[dis_names[i]]))
    f2.close()
        
    file_path = os.path.join(save_folder_path, save_file_name)
    f = open(file_path, 'a')
    _pred_list, _mos_list = [], []
    with open(result_save_path, 'r') as txt_file:
        for line in txt_file:
            try:
                dis, pred_score, gt_score = line.split()
                _pred_list.append(float(pred_score))
                _mos_list.append(float(gt_score))
            except:
                pass
    
    fitted_pred_score_list = fit_curve(_pred_list, _mos_list)
    if contains_nan(fitted_pred_score_list) or all(x == fitted_pred_score_list[0] for x in fitted_pred_score_list):
        fitted_pred_score_list = _pred_list

    pred = np.array(fitted_pred_score_list).reshape(-1)
    gt =  np.array(_mos_list).reshape(-1)

    srcc = calculate_srcc(pred, gt)
    plcc = calculate_plcc(pred, gt)
    print("Global SRCC: {}, PLCC: {}".format(srcc, plcc))
    f.write("Global SRCC: {}, PLCC: {} ".format(srcc, plcc))
    f.close()


def gpt4v_multiple_nr_batch(
        key: str,
        dataset_name: str,
        image_path_list: list,
        image_name_list: list,
        mos_list: list,
        text_prompt: str,
        save_folder_path: str,
        save_file_name: str,
        nr_image_group_num: int,
        prompt_pattern_str: str,
        ic_path: str
    ):
    GPT4v = GPT4VSystem(
        key=key, save_folder_path=save_folder_path, save_file_name=save_file_name
    )
    MesModule = MessageModule()
    print("path length {}, name length {}, mos length {}".format(
            len(image_path_list), len(image_name_list), len(mos_list)
        )
    )

    # process input list
    dis_score = {}
    for i in range(len(image_name_list)):
        dis_score[image_name_list[i]] = mos_list[i]

    path_ranking_list = create_nr_groups(
        arr=image_path_list,
        image_pair_num=nr_image_group_num,
        group_num=4
    )
    name_ranking_list = create_nr_groups(
        arr=image_name_list,
        image_pair_num=nr_image_group_num,
        group_num=4
    )

    ranking_mos_list = []
    for i in range(len(name_ranking_list)):
        cur_mos_list = []
        for j in range(len(name_ranking_list[i])):
            cur_mos_list.append(dis_score[name_ranking_list[i][j]])
        ranking_mos_list.append(cur_mos_list)
    
    # distorted image1 is key, compared distorted image2 and compare result is prediction score
    dis1_dis2_score_dict = {}
    for i in range(len(path_ranking_list)):
        # need to send single element
        print("Current {} image names {}".format(i + 1, name_ranking_list[i]))

        cur_image_path = ic_path + path_ranking_list[i]
        re_role, re_content, pred_score, token_num = dialogue_with_check(
            message=None,
            GPT4v=GPT4v,
            MesModule=MesModule,
            text_prompt=text_prompt,
            image_name_list=name_ranking_list[i],
            image_path_list=cur_image_path,
            mos_list=ranking_mos_list[i],
            pattern="multiple"
        )
        log_respone(
            folder_path=save_folder_path,
            file_name=save_file_name,
            text_prompt=text_prompt,
            input_content=re_content,
            input_image_names=name_ranking_list[i],
            input_mos=ranking_mos_list[i],
            token_num=token_num
        )
        
        dis_name_pair = transform_ranking_to_pair(name_ranking_list[i])
        pred_pair = transform_ranking_to_pair(pred_score)

        for j in range(len(dis_name_pair)):
            dis1_name = dis_name_pair[j][0]
            dis2_name = dis_name_pair[j][1]

            # for three types, gt score
            if pred_pair[j][0] > pred_pair[j][1]:
                pred_score = 1
            elif pred_pair[j][0] < pred_pair[j][1]:
                pred_score = 0
            elif pred_pair[j][0] == pred_pair[j][1]:
                pred_score = 2

            if dis1_name not in dis1_dis2_score_dict:
                dis1_dis2_score_dict[dis1_name] = [[dis2_name, pred_score]]
            else:
                dis1_dis2_score_dict[dis1_name].append([dis2_name, pred_score])
    
    # saving the score results
    result_file_name = "{}_NR_num-{}_gpt4v_{}.txt".format(
        dataset_name, len(image_path_list), prompt_pattern_str)
    
    result_save_path = os.path.join(save_folder_path, result_file_name)
    f2 = open(result_save_path, 'a')

    dis_names = list(dis1_dis2_score_dict.keys())
    name_length = len(dis_names)
    C = np.array([[0 for _ in range(name_length)] for _ in range(name_length)])

    # define the name index for optimizing
    idx = 0
    name_idx = {}
    for name in dis_names:
        name_idx[name] = idx
        idx += 1
    
    # print(dis_names)
    for i in range(name_length):
        # find dis1 compared results and objects
        for j in range(len(dis1_dis2_score_dict[dis_names[i]])):
            dis1 = dis_names[i]
            dis2 = dis1_dis2_score_dict[dis_names[i]][j][0]
            compared_res = dis1_dis2_score_dict[dis_names[i]][j][1]

            if dis1 not in name_idx or dis2 not in name_idx:
                continue

            # avoid other answers
            if compared_res == 2:
                C[name_idx[dis1]][name_idx[dis2]] += 1
                C[name_idx[dis2]][name_idx[dis1]] += 1
            elif compared_res == 1:
                C[name_idx[dis1]][name_idx[dis2]] += 1
            elif compared_res == 0:
                C[name_idx[dis2]][name_idx[dis1]] += 1

    pred_score_list = optimize_score(C=C, seed=0, original_seed=20)
    for i in range(name_length):
        f2.write("{} {} {}\n".format(
            dis_names[i], pred_score_list[i], dis_score[dis_names[i]]))
    f2.close()
        
    file_path = os.path.join(save_folder_path, save_file_name)
    f = open(file_path, 'a')
    _pred_list, _mos_list = [], []
    with open(result_save_path, 'r') as txt_file:
        for line in txt_file:
            try:
                dis, pred_score, gt_score = line.split()
                _pred_list.append(float(pred_score))
                _mos_list.append(float(gt_score))
            except:
                pass
    
    fitted_pred_score_list = fit_curve(_pred_list, _mos_list)
    if contains_nan(fitted_pred_score_list) or all(x == fitted_pred_score_list[0] for x in fitted_pred_score_list):
        fitted_pred_score_list = _pred_list

    pred = np.array(fitted_pred_score_list).reshape(-1)
    gt =  np.array(_mos_list).reshape(-1)

    srcc = calculate_srcc(pred, gt)
    plcc = calculate_plcc(pred, gt)
    print("Global SRCC: {}, PLCC: {}".format(srcc, plcc))
    f.write("Global SRCC: {}, PLCC: {} ".format(srcc, plcc))
    f.close()


def gpt4v_single_fr_batch(
        key: str,
        dataset_name: str,
        image_path_list: list,
        image_name_list: list,
        mos_list: list,
        text_prompt: str,
        save_folder_path: str,
        save_file_name: str,
        mean_metric: bool,
        prompt_pattern_str: str,
        ic_path: str
    ):

    GPT4v = GPT4VSystem(
        key=key, save_folder_path=save_folder_path, save_file_name=save_file_name
    )
    MesModule = MessageModule()
    print("path length {}, name length {}, mos length {}".format(
            len(image_path_list), len(image_name_list), len(mos_list)
        )
    )

    # process input list
    chunked_image_path_list = image_path_list
    chunked_image_name_list = image_name_list
    chunked_mos_list = chunk_list(original_list=mos_list, num=1)

    # saving the score results
    result_file_name = "{}_FR_num-{}_gpt4v_{}.txt".format(
        dataset_name, len(image_path_list), prompt_pattern_str)
    result_save_path = os.path.join(save_folder_path, result_file_name)
    f2 = open(result_save_path, 'a')

    pred_score_list = []
    for i in range(len(chunked_image_path_list)):
        # need to send single elemnet
        print("Current {} image names {}".format(i + 1, chunked_image_name_list[i]))
        cur_image_path = ic_path + chunked_image_path_list[i]
        re_role, re_content, pred_score, token_num = dialogue_with_check(
            message=None,
            GPT4v=GPT4v,
            MesModule=MesModule,
            text_prompt=text_prompt,
            image_name_list=chunked_image_name_list[i],
            image_path_list=cur_image_path,
            mos_list=chunked_mos_list[i],
            pattern="single"
        )

        log_respone(
            folder_path=save_folder_path,
            file_name=save_file_name,
            text_prompt=text_prompt,
            input_content=re_content,
            input_image_names=chunked_image_name_list[i],
            input_mos=chunked_mos_list[i],
            token_num=token_num
        )

        pred_score_list.append(pred_score[0])
        f2.write("{} {} {} {}\n".format(image_name_list[i][0], image_name_list[i][1], pred_score[0], mos_list[i]))
    
    f2.close()

    fitted_pred_score_list = fit_curve(pred_score_list, mos_list)
    if contains_nan(fitted_pred_score_list) or all(x == fitted_pred_score_list[0] for x in fitted_pred_score_list):
        fitted_pred_score_list = pred_score_list

    pred = np.array(fitted_pred_score_list).reshape(-1)
    gt =  np.array(mos_list).reshape(-1)
    srcc = calculate_srcc(pred, gt)
    plcc = calculate_plcc(pred, gt)
    print("Global SRCC: {}, PLCC: {}".format(srcc, plcc))

    file_path = os.path.join(save_folder_path, save_file_name)
    f = open(file_path, 'a')
    f.write("Global SRCC: {}, PLCC: {} ".format(srcc, plcc))
    
    if mean_metric:
        srcc, plcc = cal_metric_from_same_content(file_path=result_save_path)
        print("Local mean SRCC: {}, PLCC: {}".format(srcc, plcc))
        f.write("Local mean SRCC: {}, PLCC: {}".format(srcc, plcc))

    f.close()


def gpt4v_double_fr_batch(
        key: str,
        dataset_name: str,
        image_path_list: list,
        image_name_list: list,
        mos_list: list,
        text_prompt: str,
        save_folder_path: str,
        save_file_name: str,
        fr_image_pair_num: int,
        mean_metric: bool,
        prompt_pattern_str: str,
        ic_path: str
    ):

    GPT4v = GPT4VSystem(
        key=key, save_folder_path=save_folder_path, save_file_name=save_file_name
    )
    MesModule = MessageModule()
    print("path length {}, name length {}, mos length {}".format(
            len(image_path_list), len(image_name_list), len(mos_list)
        )
    )

    # process input list
    dis_score = {}
    for i in range(len(image_name_list)):
        dis_score[image_name_list[i][1]] = mos_list[i]

    path_paired_list, name_paired_list = creat_fr_pairs(
        image_path_list=image_path_list,
        image_name_list=image_name_list,
        n_pairs=fr_image_pair_num
    )

    paired_mos_list = []
    for i in range(len(name_paired_list)):
        paired_mos_list.append(
            [dis_score[name_paired_list[i][1]], dis_score[name_paired_list[i][2]]])

    name_ref_dis_dict = {}
    for item in image_name_list:
        key = item[0]
        if key not in name_ref_dis_dict:
            name_ref_dis_dict[key] = []
        name_ref_dis_dict[key].append(item[1])
        
    # distorted image1 is key, compared distorted image2 and compare result is prediction score
    dis1_dis2_score_dict = {}
    for i in range(len(path_paired_list)):
        # need to send single element
        dis1_name, dis2_name = name_paired_list[i][1], name_paired_list[i][2]

        cur_image_path = ic_path + path_paired_list[i]
        re_role, re_content, pred_score, token_num = dialogue_with_check(
            message=None,
            GPT4v=GPT4v,
            MesModule=MesModule,
            text_prompt=text_prompt,
            image_name_list=name_paired_list[i],
            image_path_list=cur_image_path,
            mos_list=paired_mos_list[i],
            pattern="double"
        )
        log_respone(
            folder_path=save_folder_path,
            file_name=save_file_name,
            text_prompt=text_prompt,
            input_content=re_content,
            input_image_names=name_paired_list[i],
            input_mos=paired_mos_list[i],
            token_num=token_num
        )

        if dis1_name not in dis1_dis2_score_dict:
            dis1_dis2_score_dict[dis1_name] = [[dis2_name, pred_score[0]]]
        else:
            dis1_dis2_score_dict[dis1_name].append([dis2_name, pred_score[0]])
    
    # saving the score results
    result_file_name = "{}_FR_num-{}_gpt4v_{}.txt".format(
        dataset_name, len(image_path_list), prompt_pattern_str)
    
    result_save_path = os.path.join(save_folder_path, result_file_name)
    f2 = open(result_save_path, 'a')

    # print(name_ref_dis_dict)
    for ref_name, dis_names in name_ref_dis_dict.items():
        name_length = len(dis_names)
        
        C = np.array([[0 for _ in range(name_length)] for _ in range(name_length)])

        # define the name index for optimizing

        idx = 0
        name_idx = {}
        for name in dis_names:
            name_idx[name] = idx
            idx += 1
        
        for i in range(name_length):
            # find dis1 compared results and objects
            for j in range(len(dis1_dis2_score_dict[dis_names[i]])):
                dis1 = dis_names[i]
                dis2 = dis1_dis2_score_dict[dis_names[i]][j][0]
                # print(dis1_dis2_score_dict[dis_names[i]])
                compared_res = dis1_dis2_score_dict[dis_names[i]][j][1]

                # avoid other answers
                if dis1 not in name_idx or dis2 not in name_idx:
                    continue
                
                if compared_res == 2:
                    C[name_idx[dis1]][name_idx[dis2]] += 1
                    C[name_idx[dis2]][name_idx[dis1]] += 1
                elif compared_res == 1:
                    C[name_idx[dis1]][name_idx[dis2]] += 1
                elif compared_res == 0:
                    C[name_idx[dis2]][name_idx[dis1]] += 1

        pred_score_list = optimize_score(C=C, seed=0, original_seed=20)
        for i in range(name_length):
            f2.write("{} {} {} {}\n".format(
                ref_name, dis_names[i], pred_score_list[i], dis_score[dis_names[i]]))
    
    f2.close()

    file_path = os.path.join(save_folder_path, save_file_name)
    f = open(file_path, 'a')

    _pred_list, _mos_list = [], []
    with open(result_save_path, 'r') as txt_file:
        for line in txt_file:
            try:
                ref, dis, pred_score, gt_score = line.split()
                _pred_list.append(float(pred_score))
                _mos_list.append(float(gt_score))
            except:
                pass
    
    fitted_pred_score_list = fit_curve(_pred_list, _mos_list)
    if contains_nan(fitted_pred_score_list) or all(x == fitted_pred_score_list[0] for x in fitted_pred_score_list):
        fitted_pred_score_list = _pred_list

    pred = np.array(fitted_pred_score_list).reshape(-1)
    gt =  np.array(_mos_list).reshape(-1)
    srcc = calculate_srcc(pred, gt)
    plcc = calculate_plcc(pred, gt)
    print("Global SRCC: {}, PLCC: {}".format(srcc, plcc))
    f.write("Global SRCC: {}, PLCC: {} ".format(srcc, plcc))

    if mean_metric:
        srcc, plcc = cal_metric_from_same_content(file_path=result_save_path)
        print("Local mean SRCC: {}, PLCC: {}".format(srcc, plcc))
        f.write("Local mean SRCC: {}, PLCC: {}".format(srcc, plcc))

    f.close()


def gpt4v_multiple_fr_batch(
        key: str,
        dataset_name: str,
        image_path_list: list,
        image_name_list: list,
        mos_list: list,
        text_prompt: str,
        save_folder_path: str,
        save_file_name: str,
        fr_image_group_num: int,
        mean_metric: bool,
        prompt_pattern_str: str,
        ic_path: str
    ):

    GPT4v = GPT4VSystem(
        key=key, save_folder_path=save_folder_path, save_file_name=save_file_name
    )
    MesModule = MessageModule()
    print("path length {}, name length {}, mos length {}".format(
            len(image_path_list), len(image_name_list), len(mos_list)
        )
    )

    # process input list
    dis_score = {}
    for i in range(len(image_name_list)):
        dis_score[image_name_list[i][1]] = mos_list[i]

    path_ranking_list, name_ranking_list = create_fr_groups(
        image_path_list=image_path_list,
        image_name_list=image_name_list,
        fr_image_group_num=fr_image_group_num,
        each_group_num=4
    )

    ranking_mos_list = []
    for i in range(len(name_ranking_list)):
        cur_mos_list = []
        for j in range(1, len(name_ranking_list[i])):
            cur_mos_list.append(dis_score[name_ranking_list[i][j]])
        ranking_mos_list.append(cur_mos_list)

    name_ref_dis_dict = {}
    for item in image_name_list:
        key = item[0]
        if key not in name_ref_dis_dict:
            name_ref_dis_dict[key] = []
        name_ref_dis_dict[key].append(item[1])
        
    # distorted image1 is key, compared distorted image2 and compare result is prediction score
    dis1_dis2_score_dict = {}
    for i in range(len(path_ranking_list)):
        # need to send single element
        print("Current {} image names {}".format(i + 1, name_ranking_list[i]))
        cur_image_path = ic_path + path_ranking_list[i]

        re_role, re_content, pred_score, token_num = dialogue_with_check(
            message=None,
            GPT4v=GPT4v,
            MesModule=MesModule,
            text_prompt=text_prompt,
            image_name_list=name_ranking_list[i],
            image_path_list=cur_image_path,
            mos_list=ranking_mos_list[i],
            pattern="multiple"
        )
        log_respone(
            folder_path=save_folder_path,
            file_name=save_file_name,
            text_prompt=text_prompt,
            input_content=re_content,
            input_image_names=name_ranking_list[i],
            input_mos=ranking_mos_list[i],
            token_num=token_num
        )

        dis_name_pair = transform_ranking_to_pair(name_ranking_list[i][1:])
        pred_pair = transform_ranking_to_pair(pred_score)

        for j in range(len(dis_name_pair)):
            dis1_name = dis_name_pair[j][0]
            dis2_name = dis_name_pair[j][1]

            # for three types, gt score
            if pred_pair[j][0] > pred_pair[j][1]:
                cur_pred_score = 1
            elif pred_pair[j][0] < pred_pair[j][1]:
                cur_pred_score = 0
            elif pred_pair[j][0] == pred_pair[j][1]:
                cur_pred_score = 2

            if dis1_name not in dis1_dis2_score_dict:
                dis1_dis2_score_dict[dis1_name] = [[dis2_name, cur_pred_score]]
            else:
                dis1_dis2_score_dict[dis1_name].append([dis2_name, cur_pred_score])
    
    # saving the score results
    result_file_name = "{}_FR_num-{}_gpt4v_{}.txt".format(
        dataset_name, len(image_path_list), prompt_pattern_str)
    
    result_save_path = os.path.join(save_folder_path, result_file_name)
    f2 = open(result_save_path, 'a')

    for ref_name, dis_names in name_ref_dis_dict.items():
        name_length = len(dis_names)
        C = np.array([[0 for _ in range(name_length)] for _ in range(name_length)])

        # define the name index for optimizing
        idx = 0
        name_idx = {}
        for name in dis_names:
            name_idx[name] = idx
            idx += 1
        
        for i in range(name_length):
            # find dis1 compared results and objects
            for j in range(len(dis1_dis2_score_dict[dis_names[i]])):
                dis1 = dis_names[i]
                dis2 = dis1_dis2_score_dict[dis_names[i]][j][0]
                compared_res = dis1_dis2_score_dict[dis_names[i]][j][1]

                if dis1 not in name_idx or dis2 not in name_idx:
                    continue

                # avoid other answers
                if compared_res == 2:
                    C[name_idx[dis1]][name_idx[dis2]] += 1
                    C[name_idx[dis2]][name_idx[dis1]] += 1
                elif compared_res == 1:
                    C[name_idx[dis1]][name_idx[dis2]] += 1
                elif compared_res == 0:
                    C[name_idx[dis2]][name_idx[dis1]] += 1

        pred_score_list = optimize_score(C=C, seed=0, original_seed=20)
        for i in range(name_length):
            f2.write("{} {} {} {}\n".format(
                ref_name, dis_names[i], pred_score_list[i], dis_score[dis_names[i]]))
    
    f2.close()

    file_path = os.path.join(save_folder_path, save_file_name)
    f = open(file_path, 'a')

    _pred_list, _mos_list = [], []
    with open(result_save_path, 'r') as txt_file:
        for line in txt_file:
            try:
                ref, dis, pred_score, gt_score = line.split()
                _pred_list.append(float(pred_score))
                _mos_list.append(float(gt_score))
            except:
                pass
    
    fitted_pred_score_list = fit_curve(_pred_list, _mos_list)
    if contains_nan(fitted_pred_score_list) or all(x == fitted_pred_score_list[0] for x in fitted_pred_score_list):
        fitted_pred_score_list = _pred_list

    pred = np.array(fitted_pred_score_list).reshape(-1)
    gt =  np.array(_mos_list).reshape(-1)
    srcc = calculate_srcc(pred, gt)
    plcc = calculate_plcc(pred, gt)
    print("Global SRCC: {}, PLCC: {}".format(srcc, plcc))
    f.write("Global SRCC: {}, PLCC: {} ".format(srcc, plcc))

    if mean_metric:
        srcc, plcc = cal_metric_from_same_content(file_path=result_save_path)
        print("Local mean SRCC: {}, PLCC: {}".format(srcc, plcc))
        f.write("Local mean SRCC: {}, PLCC: {}".format(srcc, plcc))

    f.close()


def extract_floats_from_string(s, str_following: str="Score:"):
    score_str = s.split(str_following)[1]
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", score_str)
    return [float(num) for num in numbers]


class CommandResult:
    def __init__(self):
        self.re_role = None
        self.re_content = None
        self.token_num = None


def execute_command(result_container, GPT4v, message, image_name_list, mos_list):
    result_container.re_role, result_container.re_content, result_container.token_num = GPT4v.dialogue(
        messages=message, image_names=image_name_list, mos_list=mos_list
    )


def execute_with_timeout(timeout_seconds, GPT4v, message, image_name_list, mos_list):
    result_container = CommandResult()
    thread = threading.Thread(
        target=execute_command, args=(result_container, GPT4v, message, image_name_list, mos_list))
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print("Time Out...........")
        thread.join()
        return execute_with_timeout(timeout_seconds)
    else:
        return result_container.re_role, result_container.re_content, result_container.token_num


def dialogue_with_check(
        message,
        GPT4v,
        MesModule,
        text_prompt,
        image_name_list,
        image_path_list,
        mos_list,
        pattern
    ):
    error_times = 0
    while True:
        # Check whether sucessfully dialogue
        try:
            message = MesModule.add_message(
                original_message=message,
                role="user", text_prompt=text_prompt, image_path_list=image_path_list
            )

            # Add thread to control time
            re_role, re_content, token_num = execute_with_timeout(
                20, GPT4v, message, image_name_list, mos_list)

            # If GPT4V answer other things
            cur_score = extract_floats_from_string(re_content)
            score_flag = len(cur_score) != 1 if pattern == "double" else len(cur_score) != len(mos_list)

            cycle_count = 0
            while "Score:" not in re_content or score_flag:
                cycle_count += 1
                print("=========== Reconversation ===========")
                re_role, re_content, token_num = execute_with_timeout(
                    20, GPT4v, message, image_name_list, mos_list)
                cur_score = extract_floats_from_string(re_content)
                score_flag = len(cur_score) != 1 if pattern == "double" else len(cur_score) != len(mos_list)

                # accelerate.
                if cycle_count >= 2:
                    re_role = ""

                    if pattern == "double":
                        cur_score = [random.randint(0, 2)]
                    elif pattern == "single":
                        cur_score = [random.randint(0, 100)]
                    elif pattern == "multiple":
                        cur_score = [random.randint(0, 3) for _ in range(4)]

                    re_content = "Error Random Score:{}".format(cur_score)
                    token_num = 0
                    break
                    
            break
        except:
            # print("Meet error......{}".format(error_times))
            error_times += 1
            message = None
        
            if error_times >= 3:
                re_role = ""

                if pattern == "double":
                    cur_score = [random.randint(0, 2)]
                elif pattern == "single":
                    cur_score = [random.randint(0, 100)]
                elif pattern == "multiple":
                    cur_score = [random.randint(0, 3) for _ in range(4)]

                re_content = "Error Random Score:{}".format(cur_score)
                token_num = 0
                break
    
    return re_role, re_content, cur_score, token_num


def log_respone(
        folder_path,
        file_name,
        text_prompt,
        input_content,
        input_image_names,
        input_mos,
        token_num
    ):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    
    f = open(file_path, 'a')
    f.write("Your input prompt:\n{}\n".format(text_prompt))
    f.write("Your input image names: {}\n".format(input_image_names))
    try:
        f.write("GPT4V:\n{}\n".format(input_content))
    except:
        f.write("GPT4V:\n{}\n".format("Something wrong"))

    if input_mos is not None:
        f.write("Mos: {}\n".format(input_mos))

    f.write("Used token number: {}\n\n\n".format(token_num))
    f.close()


class GPT4VSystem():
    def __init__(self, key, save_folder_path, save_file_name):
        # OpenAI API Key
        self.api_key = key
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        self.save_folder_path = save_folder_path
        self.save_file_name = save_file_name

    def dialogue(self, messages: list, image_names: list=None, mos_list: list=None):
        text_prompt = messages[-1]["content"][0]["text"]
        payload = {"model": "gpt-4-vision-preview", "messages": messages, "max_tokens": 2000}
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)

        print(response.json())
        # Get GPT4V role
        re_role = response.json()['choices'][0]['message']['role']

        # get GPT4V answer
        re_content = response.json()['choices'][0]['message']['content']
        all_info_respond = response.json()
        token_num = all_info_respond['usage']['total_tokens']

        print("Your input prompt:\n" + text_prompt)
        print("Your input image name: {}\n".format(image_names))
        print("GPT4V:\n" + re_content)

        if mos_list is not None:
            print("Mos: {}\n".format(mos_list))

        return re_role, re_content, token_num


class MessageModule():
    def __init__(self):
        pass

    def define_text_dict(self, input_prompt):
        text_dict = {
            "type": "text",
            "text": input_prompt
        }
        return text_dict

    def define_image_dict(self, encoded_image):
        image_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        }
        return image_dict

    def encode_image(self, image_path):
        with Image.open(image_path) as img:
            # img = img.resize((512, 512))
            img = img.convert('RGB')
            with io.BytesIO() as byte_stream:
                img.save(byte_stream, format='png')
                byte_stream.seek(0)
                encoded_string = base64.b64encode(byte_stream.read()).decode('utf-8')
                return encoded_string
    
    def define_one_content(self, text_prompt: str="", image_path_list: list=[]):
        input_text_dict = self.define_text_dict(text_prompt)
        image_num = 0

        if image_path_list != []:
            input_image_dicts = []
            for i in range(len(image_path_list)):
                img_path = image_path_list[i]
                encoded_image = self.encode_image(img_path)
                cur_image_dict = self.define_image_dict(encoded_image)
                input_image_dicts.append(cur_image_dict)

            image_num = len(input_image_dicts)

        # Add image dicts
        content = [input_text_dict]
        if image_num != 0:
            for i in range(image_num):
                content.append(input_image_dicts[i])
        
        return content

    def define_one_message(self, role: str="user", content: list=None):
        message = [
            {
                "role": role,
                "content": content
            }
        ]
        return message

    def add_message(
            self,
            original_message: list=None,
            role: str="user", # assistant
            text_prompt: str="",
            image_path_list: list=[]
        ):

        new_content = self.define_one_content(text_prompt=text_prompt, image_path_list=image_path_list)
        new_message = self.define_one_message(role=role, content=new_content)
        if original_message is None:
            return new_message
        else:
            original_message.append(new_message[0])

        return original_message

