import pyiqa
import torch
import os
import clip
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
import pickle

from PIL import Image
from utils import *
from data.sample_utils import parse_datasets
from data.helpers_sample import parse_sampled_datasets


DATASET_NAME_IQA_TYPE = {
    "FR_KADID": "FR", "NR_KADID": "NR", "SPAQ": "NR", "AGIQA3K": "NR"
}

DATASET_MEAN_METRIC = {
    "FR_KADID": True, "NR_KADID": False, "SPAQ": False, "AGIQA3K": False
}


class SampleSelection():
    def __init__(
            self,
            model_name_fit,
            model_name_gen,
            model_name_test,
            dataset_name,
            iqa_type,
            sample_num,
            diff_content_num,
            diff_distortion_num,
            gpu_idx
        ):
        self.iqa_type = iqa_type
        self.model_name_fit = model_name_fit
        self.model_name_gen = model_name_gen
        self.model_name_test = model_name_test

        self.dataset_name = dataset_name
        self.sample_num = sample_num
        self.diff_content_num = diff_content_num
        self.diff_distortion_num = diff_distortion_num
        self.device = torch.device("cuda:{}".format(gpu_idx)) if torch.cuda.is_available() else torch.device("cpu")
        
        if dataset_name == "FR_KADID":
            self.dis_data_path = "C:/wutianhe/sigs/research/IQA_dataset/kadid10k/images"
            self.ref_data_path = "C:/wutianhe/sigs/research/IQA_dataset/kadid10k/images"

        elif dataset_name == "SPAQ":
            self.dis_data_path = "/mnt/data/wth22/IQA_dataset/SPAQ/dataset/TestImage/"
            self.ref_data_path = None
        
        elif dataset_name == "NR_KADID":
            self.dis_data_path = "/mnt/data/wth22/IQA_dataset/kadid10k/images/"
            self.ref_data_path = None
        
        elif dataset_name == "AGIQA3K":
            self.dis_data_path = "/mnt/data/wth22/IQA_dataset/AIGC-3K/"
            self.ref_data_path = None
    
    def creat_iqa_model(self, name):
        iqa_model = pyiqa.create_metric(name, device=self.device)
        # iqa_model = pyiqa.create_metric(name).cuda()
        return iqa_model
    
    def creat_sem_model(self,):
        sem_encoder, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        # sem_encoder, preprocess = clip.load("ViT-B/32", jit=False)
        return sem_encoder
    
    def image2tensor(self, image_path, resize: bool=True, size_num: int=256):
        image = Image.open(image_path).convert("RGB")
        if resize:
            image = transforms.functional.resize(image, size_num)
    
        image = transforms.ToTensor()(image).to(self.device)
        # image = transforms.ToTensor()(image).cuda() 
        return image.unsqueeze(0)
    
    def define_dataset(self,):
        self.image_path_list, self.image_name_list, self.score_list, self.std_list = parse_datasets(
            dataset_name=self.dataset_name,
            iqa_type=self.iqa_type,
            dis_data_path=self.dis_data_path,
            ref_data_path=self.ref_data_path
        )

        print("Image num {}".format(len(self.image_path_list)))
        print("score num {}".format(len(self.score_list)))
        print("std length {}".format(len(self.std_list)))
        print("Max std {}, Min std {}, dataset std mean {}".format(
            max(self.std_list), min(self.std_list), sum(self.std_list) / len(self.std_list)))

    def find_top_n_differences(self, list1, list2, std_list, std_scale, n: int=10):
        if len(list1) != len(list2):
            raise ValueError("Lists must have the same length")
        
        differences = []
        for i in range(len(list1)):
            differences.append(
                (list1[i] - list2[i]) ** 2 / (std_list[i] ** 2 + std_scale)
            )

        top_n_indices = sorted(range(len(differences)), key=lambda i: differences[i], reverse=True)[:n]
        return top_n_indices
    
    def predict_and_store(self,):
        file_path = "./Results/Combine_mos_sample_selection/{}/".format(self.dataset_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        fit_score_list, gen_score_list = [], []
        if self.iqa_type == "FR":
            for i in tqdm(range(len(self.image_path_list))):
                image1 = self.image2tensor(self.image_path_list[i][0], resize=True, size_num=256)
                image2 = self.image2tensor(self.image_path_list[i][1], resize=True, size_num=256)

                if image1.shape != image2.shape:
                    image1 = self.image2tensor(self.image_path_list[i][0], resize=True, size_num=(256, 256))
                    image2 = self.image2tensor(self.image_path_list[i][1], resize=True, size_num=(256, 256))

                with torch.inference_mode(mode=True):
                    if self.iqa_model_fit is not None:
                        fit_score = self.iqa_model_fit(image1, image2)
                        fit_score_list.append(np.squeeze(fit_score.data.cpu().numpy()))
                    
                    gen_score = self.iqa_model_gen(image1, image2)
                    gen_score_list.append(np.squeeze(gen_score.data.cpu().numpy()))
        
            _gen_score_list = fit_curve(gen_score_list, self.score_list)
            if self.iqa_model_fit is not None:
                _fit_score_list = fit_curve(fit_score_list, self.score_list)
                pred_score_list = [(x + y) / 2 for x, y in zip(_fit_score_list, _gen_score_list)]
            else:
                pred_score_list = _gen_score_list

            ret_file_path_name = file_path + "{}_fit-{}_gen-{}.txt".format(
                self.dataset_name, self.model_name_fit, self.model_name_gen)
            
            f = open(ret_file_path_name, 'w')
            for i in tqdm(range(len(self.image_path_list))):
                f.write("{} {} {} {} {}\n".format(
                        self.image_name_list[i][0], self.image_name_list[i][1],
                        pred_score_list[i], self.score_list[i], self.std_list[i]
                    )
                )
        
        elif self.iqa_type == "NR":
            for i in tqdm(range(len(self.image_path_list))):
                image = self.image2tensor(self.image_path_list[i], resize=True, size_num=224)

                with torch.inference_mode(mode=True):
                    if self.iqa_model_fit is not None:
                        fit_score = self.iqa_model_fit(image)
                        fit_score_list.append(np.squeeze(fit_score.data.cpu().numpy()))
                    
                    gen_score = self.iqa_model_gen(image)
                    gen_score_list.append(np.squeeze(gen_score.data.cpu().numpy()))
                
            _gen_score_list = fit_curve(gen_score_list, self.score_list)
            if self.iqa_model_fit is not None:
                _fit_score_list = fit_curve(fit_score_list, self.score_list)
                pred_score_list = [(x + y) / 2 for x, y in zip(_fit_score_list, _gen_score_list)]
            else:
                pred_score_list = _gen_score_list
            
            ret_file_path_name = file_path + "{}_fit-{}_gen-{}.txt".format(
                self.dataset_name, self.model_name_fit, self.model_name_gen)
            
            f = open(ret_file_path_name, 'w')
            for i in tqdm(range(len(self.image_path_list))):
                f.write("{} {} {} {}\n".format(
                        self.image_name_list[i],
                        pred_score_list[i], self.score_list[i], self.std_list[i]
                    )
                )

        f.close()
        return ret_file_path_name
    
    def get_sem_feat(self, image_name_list, root_path):
        print("Calculate the image feature by CLIP visual encoder...")
        name_feat_dict = {}
        with torch.inference_mode(mode=True):
            for i in tqdm(range(len(image_name_list))):
                cur_image_path = os.path.join(root_path, image_name_list[i])

                # the image sent in CLIP must be (224, 224)
                cur_image = self.image2tensor(cur_image_path, resize=True, size_num=(224, 224))
                cur_feat = self.sem_encoder.encode_image(cur_image)
                norm_cur_feat = F.normalize(cur_feat, p=2, dim=0)
                name_feat_dict[image_name_list[i]] = norm_cur_feat
        
        return name_feat_dict
    
    def calcaulate_cos_sim(self, feat1, feat2):
        cos_sim = F.cosine_similarity(feat1.squeeze(), feat2.squeeze(), dim=0)
        return cos_sim
    
    def calculate_sem_distance(self, image_name, sampled_feat, name_feat_dict):
        length = sampled_feat.shape[0]
        if length == 0:
            return 0
        
        feat = name_feat_dict[image_name]
        avg_sampled_feat = sampled_feat.mean(dim=0)
        avg_cos_sim = self.calcaulate_cos_sim(feat, avg_sampled_feat)
        return avg_cos_sim
    
    def sample_difficult_fr_dataset(self, sem_scale, std_scale):
        self.define_dataset()

        root_path = "./Results/Combine_mos_sample_selection/{}/".format(self.dataset_name)
        file_path = root_path + "{}_fit-{}_gen-{}.txt".format(
            self.dataset_name, self.model_name_fit, self.model_name_gen)

        if not os.path.exists(file_path):
            if self.model_name_fit is not None:
                print("Load fitted model..")
                self.iqa_model_fit = self.creat_iqa_model(self.model_name_fit)
            else:
                self.iqa_model_fit = None

            self.iqa_model_gen = self.creat_iqa_model(self.model_name_gen)
            self.iqa_model_fit.eval()
            self.iqa_model_gen.eval()
            
            print("Start to predict score with fitted and generalized models...")
            file_path = self.predict_and_store()
            print("Prediction done.")
        else:
            print("The quality score has been calculated, start to sample.")
        
        self.sem_encoder = self.creat_sem_model()
        self.sem_encoder.eval()

        ref_pred_dict, ref_gt_dict, ref_dis_dict, ref_std_dict = {}, {}, {}, {}
        with open(file_path, 'r') as txt_file:
            for line in txt_file:
                try:
                    ref, dis, pred_score, gt_score, std_val = line.split()
                    if ref not in ref_pred_dict:
                        ref_pred_dict[ref] = [float(pred_score)]
                    else:
                        ref_pred_dict[ref].append(float(pred_score))

                    if ref not in ref_gt_dict:
                        ref_gt_dict[ref] = [float(gt_score)]
                    else:
                        ref_gt_dict[ref].append(float(gt_score))
                    
                    if ref not in ref_dis_dict:
                        ref_dis_dict[ref] = [dis]
                    else:
                        ref_dis_dict[ref].append(dis)
                    
                    if ref not in ref_std_dict:
                        ref_std_dict[ref] = [float(std_val)]
                    else:
                        ref_std_dict[ref].append(float(std_val))

                except:
                    pass
        
        ref_name_list = []
        for key, value in ref_pred_dict.items():
            ref_name_list.append(key)
        
        print("There are {} different content".format(len(ref_name_list)))
        
        ref_mse_std_dict = {}
        for key, value in ref_pred_dict.items():
            # x and y are lists
            x, y = ref_pred_dict[key], ref_gt_dict[key]

            squared_errors = []
            for i in range(len(x)):
                squared_errors.append(
                    (x[i] - y[i]) ** 2 / (ref_std_dict[key][i] ** 2 + std_scale)
                )
                # print("mse: {}, std: {}".format((x[i] - y[i]) ** 2, ref_std_dict[key][i] ** 2 + std_scale))
            
            mse_std = sum(squared_errors) / len(squared_errors)
            ref_mse_std_dict[key] = mse_std

        name_feat_dict = self.get_sem_feat(
            image_name_list=ref_name_list, root_path=self.ref_data_path
        )
        
        print("Sample different content reference images...")
        sampled_ref_name = []
        sampled_ref_name_dict = {}
        sampled_feat = torch.Tensor([]).to(self.device)
        for i in tqdm(range(self.diff_content_num)):
            max_val, max_key = 0, 0

            for key, value in ref_mse_std_dict.items():
                sampled_val = 0
                if key not in sampled_ref_name_dict:
                    mse_std_val = value
                    sem_val = self.calculate_sem_distance(
                        image_name=key,
                        sampled_feat=sampled_feat,
                        name_feat_dict=name_feat_dict
                    )
                    sampled_val = mse_std_val + sem_scale * (1 - sem_val)
                    # print("mse_std {}, sem {}".format(mse_std_val, sem_scale * (1 - sem_val)))
                
                if sampled_val >= max_val:
                    max_val = sampled_val
                    max_key = key
            
            sampled_ref_name.append(max_key)
            sampled_ref_name_dict[max_key] = 1
            sampled_feat = torch.cat((sampled_feat, name_feat_dict[max_key]), dim=0)
        
        print(sampled_ref_name)
        
        stored_file_path = "./data/{}/".format(self.dataset_name)
        if not os.path.exists(stored_file_path):
            os.makedirs(stored_file_path)

        f = open(stored_file_path + "{}_FR_sampled_difficult_{}.txt".format(
                self.dataset_name, self.sample_num), 'w')
        
        print("Sample different distorted images...")

        for i in tqdm(range(len(sampled_ref_name))):
            cur_dis_name_list = ref_dis_dict[sampled_ref_name[i]]
            cur_x = ref_pred_dict[sampled_ref_name[i]]
            cur_y = ref_gt_dict[sampled_ref_name[i]]
            cur_std = ref_std_dict[sampled_ref_name[i]]

            top_idx = self.find_top_n_differences(list1=cur_x, list2=cur_y,
                std_list=cur_std, std_scale=std_scale, n=self.diff_distortion_num
            )

            sampled_x = [cur_x[j] for j in top_idx]
            sampled_y = [cur_y[j] for j in top_idx]
            sampled_dis_name_list = [cur_dis_name_list[j] for j in top_idx]

            for j in range(len(sampled_x)):
                f.write("{} {} {}\n".format(
                        sampled_ref_name[i], sampled_dis_name_list[j], sampled_y[j]
                    )
                )

    # NR sample difficult
    def sample_difficult_nr_dataset(self, sem_scale=1e-2, std_scale=1):
        self.define_dataset()

        root_path = "./Results/Combine_mos_sample_selection/{}/".format(self.dataset_name)
        file_path = root_path + "{}_fit-{}_gen-{}.txt".format(
            self.dataset_name, self.model_name_fit, self.model_name_gen)

        if not os.path.exists(file_path):
            if self.model_name_fit is not None:
                print("Load fitted model..")
                self.iqa_model_fit = self.creat_iqa_model(self.model_name_fit)
            else:
                self.iqa_model_fit = None

            self.iqa_model_gen = self.creat_iqa_model(self.model_name_gen)

            self.iqa_model_fit.eval()
            self.iqa_model_gen.eval()
            
            print("Start to predict score with fitted and generalized models...")
            file_path = self.predict_and_store()
            print("Prediction done.")
        else:
            print("The quality score has been calculated, start to sample.")
        
        self.sem_encoder = self.creat_sem_model()
        self.sem_encoder.eval()

        dis_list, x, y, std_list = [], [], [], []
        with open(file_path, 'r') as txt_file:
            for line in txt_file:
                try:
                    dis_name, pred_score, gt_score, std_val = line.split()
                    x.append(float(pred_score))
                    y.append(float(gt_score))
                    dis_list.append(dis_name)
                    std_list.append(float(std_val))
                except:
                    pass
        
        feat_path = './feats/{}_name_feat_dict.pkl'.format(dataset_name)

        if not os.path.exists(feat_path):
            print("Start to calculate the image feature....")
            name_feat_dict = self.get_sem_feat(
                image_name_list=dis_list, root_path=self.dis_data_path
            )
            with open(feat_path, 'wb') as f:
                pickle.dump(name_feat_dict, f)
        else:
            print("Load the feat_dict...")
            with open(feat_path, 'rb') as f:
                name_feat_dict = pickle.load(f)

        sampled_x, sampled_y, sampled_dis_name = [], [], []
        sampled_feat = torch.Tensor([]).to(self.device)
        # sampled_feat = torch.Tensor([]).cuda()
        sampled_dis_name_dict = {}

        stored_file_path = "./data/{}/".format(self.dataset_name)
        if not os.path.exists(stored_file_path):
            os.makedirs(stored_file_path)

        f = open(stored_file_path + "{}_NR_sampled_difficult_{}.txt".format(
                self.dataset_name, self.sample_num), 'w')

        for i in tqdm(range(self.sample_num)):
            max_val, max_idx = 0, 0

            for j in range(len(x)):
                sampled_val = 0
                if dis_list[j] not in sampled_dis_name_dict:
                    mse_val = (x[j] - y[j]) ** 2
                    sem_val = self.calculate_sem_distance(
                        image_name=dis_list[j],
                        sampled_feat=sampled_feat,
                        name_feat_dict=name_feat_dict
                    )
                    sampled_val = mse_val / (std_list[j] ** 2 + std_scale) + sem_scale * (1 - sem_val)
                
                if sampled_val >= max_val:
                    max_val = sampled_val
                    max_idx = j
            
            sampled_x.append(x[max_idx])
            sampled_y.append(y[max_idx])
            sampled_dis_name.append(dis_list[max_idx])
            sampled_dis_name_dict[dis_list[max_idx]] = 1
            sampled_feat = torch.cat((sampled_feat, name_feat_dict[dis_list[max_idx]]), dim=0)
            
            f.write("{} {}\n".format(dis_list[max_idx], y[max_idx]))
    
    def test_benchmark(self, mean_metric: bool):
        self.iqa_model_test = self.creat_iqa_model(name=self.model_name_test)
        image_path_list, image_name_list, score_list = parse_sampled_datasets(
            dataset_name=self.dataset_name,
            iqa_type=self.iqa_type,
            dis_data_path=self.dis_data_path,
            ref_data_path=self.ref_data_path
        )

        print("Testing Image num {}".format(len(image_path_list)))
        print("Testing score num {}".format(len(score_list)))

        if self.iqa_type == "FR":
            ref_pred_dict, ref_gt_dict = {}, {}
            pred_list = []
            for i in tqdm(range(len(image_path_list))):
                image1 = self.image2tensor(image_path_list[i][0], resize=True, size_num=256)
                image2 = self.image2tensor(image_path_list[i][1], resize=True, size_num=256)

                # for the wrong images like ma_sr dataset
                if image1.shape != image2.shape:
                    image1 = self.image2tensor(image_path_list[i][0], resize=True, size_num=(256, 256))
                    image2 = self.image2tensor(image_path_list[i][1], resize=True, size_num=(256, 256))
                
                # predict quality score
                with torch.inference_mode(mode=True):
                    cur_score = self.iqa_model_test(image1, image2)
                
                cur_score = np.squeeze(cur_score.data.cpu().numpy())

                ref = image_name_list[i][0]
                dis = image_name_list[i][1]
                if ref not in ref_pred_dict:
                    ref_pred_dict[ref] = [float(cur_score)]
                else:
                    ref_pred_dict[ref].append(float(cur_score))
                if ref not in ref_gt_dict:
                    ref_gt_dict[ref] = [float(score_list[i])]
                else:
                    ref_gt_dict[ref].append(float(score_list[i]))

                pred_list.append(float(cur_score))
        
        elif self.iqa_type == "NR":
            pred_list = []
            for i in tqdm(range(len(image_path_list))):
                image = self.image2tensor(image_path_list[i], resize=True, size_num=(512, 512))

                # predict quality score
                with torch.inference_mode(mode=True):
                    try:
                        cur_score = self.iqa_model_test(image)
                        cur_score = np.squeeze(cur_score.data.cpu().numpy())
                    except:
                        print("Meet Error............")
                        cur_score = 0

                pred_list.append(float(cur_score))

        if mean_metric:
            print("calculating mean...")
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
            
        else:
            pred_list = fit_curve(pred_list, score_list)
            srcc = calculate_srcc(pred_list, score_list)
            plcc = calculate_plcc(pred_list, score_list)
        
        print("Model: {}, SRCC: {}, PLCC: {}".format(self.model_name_test, srcc, plcc))


if __name__ == "__main__":
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    
    setup_seed(20)
    
    dataset_name = "FR_KADID"
    ClassSample = SampleSelection(
        model_name_fit="dists",
        model_name_gen="lpips-vgg",
        model_name_test="lpips-vgg",
        dataset_name=dataset_name,
        iqa_type=DATASET_NAME_IQA_TYPE[dataset_name],
        sample_num=150,
        diff_content_num=15,
        diff_distortion_num=10,
        gpu_idx=2
    )

    ClassSample.sample_difficult_fr_dataset(sem_scale=0.01, std_scale=1)
    ClassSample.test_benchmark(mean_metric=DATASET_MEAN_METRIC[dataset_name])

