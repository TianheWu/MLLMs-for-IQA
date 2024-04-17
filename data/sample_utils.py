from utils import *

import random
import os


def parse_datasets(
        dataset_name: str,
        iqa_type: str,
        dis_data_path: str,
        ref_data_path: str=None,
    ):
    
    if dataset_name == "FR_KADID":
        mos_file_path = "./data/FR_KADID/FR_KADID_mos_std.txt"
        sorted_data_asc, all_image_names = process_fr_kadid10k_with_std(mos_file=mos_file_path)

    elif dataset_name == "SPCD":
        mos_file_path = "./data/SPCD/SPCD_mos_std.txt"
        sorted_data_asc, all_image_names = process_spcd_with_std(mos_file=mos_file_path)
        
    elif dataset_name == "NR_KADID":
        mos_file_path = "./data/SPCD/SPCD_mos_std.txt"
        sorted_data_asc, all_image_names = process_nr_kadid10k_with_std(mos_file=mos_file_path)
    
    elif dataset_name == "SPAQ":
        mos_file_path = "./data/SPAQ/SPAQ_mos_std.txt"
        sorted_data_asc, all_image_names = process_spaq_with_std(mos_file=mos_file_path)
    
    elif dataset_name == "AGIQA3K":
        mos_file_path = "./data/AGIQA3K/AGIQA3K_mos_std.txt"
        sorted_data_asc, all_image_names = process_agiqa3k_with_std(mos_file=mos_file_path)

    sampled_data = sorted_data_asc
    random.shuffle(sampled_data)

    if iqa_type == "NR":
        image_path_list = []
        image_name_list = []
        score_list = []
        std_list = []

        _IMAGE_FORMAT = [".BMP", ".bmp", ".png", ".jpg"]
        for i in range(len(sampled_data)):
            for k in range(len(_IMAGE_FORMAT)):
                cur_path = os.path.join(dis_data_path, sampled_data[i][0] + _IMAGE_FORMAT[k])

                if os.path.exists(cur_path):
                    image_path_list.append(cur_path)
                    score_list.append(sampled_data[i][1])
                    std_list.append(sampled_data[i][2])
                    image_name_list.append(sampled_data[i][0] + _IMAGE_FORMAT[k])
                    break
        return image_path_list, image_name_list, score_list, std_list
    
    elif iqa_type == "FR":
        image_path_pair_list = []
        image_name_pair_list = []
        score_list = []
        std_list = []

        _IMAGE_FORMAT = [".BMP", ".bmp", ".png", ".jpg"]
        for i in range(len(sampled_data)):
            for k in range(len(_IMAGE_FORMAT)):
                cur_dis_path = os.path.join(dis_data_path, sampled_data[i][1] + _IMAGE_FORMAT[k])

                if os.path.exists(cur_dis_path):
                    for z in range(len(_IMAGE_FORMAT)):
                        cur_ref_path = os.path.join(ref_data_path, sampled_data[i][0] + _IMAGE_FORMAT[z])

                        if os.path.exists(cur_ref_path):
                            image_path_pair_list.append([cur_ref_path, cur_dis_path])
                            score_list.append(sampled_data[i][2])
                            std_list.append(sampled_data[i][3])
                            image_name_pair_list.append(
                                [sampled_data[i][0] + _IMAGE_FORMAT[z], sampled_data[i][1] + _IMAGE_FORMAT[k]]
                            )
                            break
                    break
    
        return image_path_pair_list, image_name_pair_list, score_list, std_list


def process_fr_kadid10k_with_std(mos_file: str):
    ref_dis_score_std = []
    ref_name = []
    ref_name_dict = {}
    with open(mos_file, 'r') as listFile:
        for line in listFile:
            cur_ref, cur_dis, cur_score, cur_std = line.split()
            cur_dis = cur_dis[:-4]
            cur_ref = cur_ref[:-4]

            ref_dis_score_std.append([cur_ref, cur_dis, float(cur_score), float(cur_std)])
            if ref_name_dict.get(cur_ref, 0) == 0:
                ref_name.append(cur_ref)
                ref_name_dict[cur_ref] = 1
    
    ref_dis_score_std = scale_values_for_fr_sample(input_list=ref_dis_score_std, scale_num=1)
    sorted_data_asc = sorted(ref_dis_score_std, key=lambda x: x[2])
    return sorted_data_asc, ref_name


def process_spcd_with_std(mos_file: str):
    ref_dis_score = []
    ref_name = []
    with open(mos_file, 'r') as listFile:
        for line in listFile:
            cur_dis1, cur_dis2, cur_score, cur_std = line.split()
            cur_dis1 = cur_dis1[:-4]
            cur_dis2 = cur_dis2[:-4]
            ref_dis_score.append([cur_dis1, cur_dis2, float(cur_score), float(cur_std)])
    
    ref_dis_score = scale_values_for_fr_sample(input_list=ref_dis_score, scale_num=1)
    sorted_data_asc = sorted(ref_dis_score, key=lambda x: x[2])
    return sorted_data_asc, ref_name


def process_nr_kadid10k_with_std(mos_file: str):
    dis_score_std = {}
    dis_name = []
    with open(mos_file, 'r') as listFile:
        for line in listFile:
            cur_ref, cur_dis, cur_score, cur_std = line.split()
            cur_dis = cur_dis[:-4]
            dis_score_std[cur_dis] = [float(cur_score), float(cur_std)]
            dis_name.append(cur_dis)
    
    dis_score_std = scale_values_for_nr_sample(dictionary=dis_score_std, scale_num=1)
    sorted_data_asc = [
        [key] + value for key, value in sorted(dis_score_std.items(), key=lambda item: item[1][0])
    ]
    return sorted_data_asc, dis_name


def process_spaq_with_std(mos_file: str):
    dis_score_std = {}
    dis_name = []
    with open(mos_file, 'r') as listFile:
        for line in listFile:
            cur_dis, cur_score, cur_std = line.split()
            cur_dis = cur_dis[:-4]
            dis_score_std[cur_dis] = [float(cur_score), float(cur_std)]
            dis_name.append(cur_dis)
    
    dis_score_std = scale_values_for_nr_sample(dictionary=dis_score_std, scale_num=1)
    sorted_data_asc = [
        [key] + value for key, value in sorted(dis_score_std.items(), key=lambda item: item[1][0])
    ]
    return sorted_data_asc, dis_name


def process_agiqa3k_with_std(mos_file: str):
    dis_score_std = {}
    dis_name = []
    with open(mos_file, 'r') as listFile:
        for line in listFile:
            cur_dis, cur_score, cur_std = line.split()
            cur_dis = cur_dis[:-4]
            dis_score_std[cur_dis] = [float(cur_score), float(cur_std)]
            dis_name.append(cur_dis)
    
    dis_score_std = scale_values_for_nr_sample(dictionary=dis_score_std, scale_num=1)
    sorted_data_asc = [
        [key] + value for key, value in sorted(dis_score_std.items(), key=lambda item: item[1][0])
    ]
    return sorted_data_asc, dis_name

