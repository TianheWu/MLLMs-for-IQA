import yaml

from utils import *
from iqa_models.gpt4v import gpt4v_single_nr_batch
from iqa_models.gpt4v import gpt4v_single_fr_batch
from iqa_models.gpt4v import gpt4v_double_fr_batch
from iqa_models.gpt4v import gpt4v_multiple_fr_batch
from iqa_models.gpt4v import gpt4v_double_cd_batch
from iqa_models.gpt4v import gpt4v_double_nr_batch
from iqa_models.gpt4v import gpt4v_multiple_nr_batch
from iqa_models.gpt4v import gpt4v_multiple_cd_batch

from prompts.gpt4v_prompt import get_prompt
from data.helpers_sample import parse_sampled_datasets


DATASET_NAME_IQA_TYPE = {
    "FR_KADID": "FR", "AUG_KADID": "FR", "TQD": "FR", "SPCD": "FR",
    "NR_KADID": "NR", "SPAQ": "NR", "AGIQA3K": "NR"
}

DATASET_MEAN_METRIC = {
    "FR_KADID": True, "AUG_KADID": True, "TQD": True, "SPCD": False,
    "NR_KADID": False, "SPAQ": False, "AGIQA3K": False
}


def read_yaml(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data, Loader=yaml.FullLoader)
        return result


class GPT4VTester():
    def __init__(self,
        nr_image_pair_num, nr_image_group_num, fr_image_pair_num, fr_image_group_num):

        setting_dict = read_yaml('./settings.yaml')
        self.key = setting_dict["KEY"]
        self.dataset_name = setting_dict["DATASET_NAME"]
        self.dis_data_path = setting_dict["DIS_DATA_PATH"]
        self.ref_data_path = setting_dict["REF_DATA_PATH"]
        self.psy_pattern = setting_dict["PSYCHOPHYSICAL_PATTERN"]
        self.nlp_pattern = setting_dict["NLP_PATTERN"]
        self.iqa_type = DATASET_NAME_IQA_TYPE[self.dataset_name]
        self.mean_metric = DATASET_MEAN_METRIC[self.dataset_name]

        if self.nlp_pattern == "ic":
            self.ic_path = setting_dict["IC_PATH"]
        else:
            self.ic_path = []

        self.nr_image_pair_num = nr_image_pair_num
        self.nr_image_group_num = nr_image_group_num
        self.fr_image_pair_num = fr_image_pair_num
        self.fr_image_group_num = fr_image_group_num

        self.text_prompt = get_prompt(
            psy_pattern=self.psy_pattern, nlp_pattern=self.nlp_pattern, iqa_type=self.iqa_type
        )

        # For saving GPT-4V outputs
        self.save_folder_path = "./Results/GPT-4V_{}/".format(self.dataset_name)
        if not os.path.exists(self.save_folder_path):
            os.makedirs(self.save_folder_path)
        self.save_file_name = "GPT-4V_{}_{}_{}.txt".format(
            self.dataset_name, self.iqa_type, self.psy_pattern + "-" + self.nlp_pattern)
        
        self.define_dataset()
    
    def define_dataset(self,):
        self.image_path_list, self.image_name_list, self.mos_list = parse_sampled_datasets(
            dataset_name=self.dataset_name,
            iqa_type=self.iqa_type,
            dis_data_path=self.dis_data_path,
            ref_data_path=self.ref_data_path
        )

        print("Image num {}".format(len(self.image_path_list)))
        print("score num {}".format(len(self.mos_list)))
    
    def test_nr_single_stimulus(self, ):
        gpt4v_single_nr_batch(
            key=self.key,
            dataset_name=self.dataset_name,
            image_path_list=self.image_path_list,
            image_name_list=self.image_name_list,
            mos_list=self.mos_list,
            text_prompt=self.text_prompt,
            save_folder_path=self.save_folder_path,
            save_file_name=self.save_file_name,
            prompt_pattern_str=self.psy_pattern + "-" + self.nlp_pattern,
            ic_path=self.ic_path
        )
    
    def test_nr_double_stimulus(self, ):
        gpt4v_double_nr_batch(
            key=self.key,
            dataset_name=self.dataset_name,
            image_path_list=self.image_path_list,
            image_name_list=self.image_name_list,
            mos_list=self.mos_list,
            text_prompt=self.text_prompt,
            save_folder_path=self.save_folder_path,
            save_file_name=self.save_file_name,
            nr_image_pair_num=self.nr_image_pair_num,
            prompt_pattern_str=self.psy_pattern + "-" + self.nlp_pattern,
            ic_path=self.ic_path
        )
    
    def test_nr_multiple_stimulus(self, ):
        gpt4v_multiple_nr_batch(
            key=self.key,
            dataset_name=self.dataset_name,
            image_path_list=self.image_path_list,
            image_name_list=self.image_name_list,
            mos_list=self.mos_list,
            text_prompt=self.text_prompt,
            save_folder_path=self.save_folder_path,
            save_file_name=self.save_file_name,
            nr_image_group_num=self.nr_image_group_num,
            prompt_pattern_str=self.psy_pattern + "-" + self.nlp_pattern,
            ic_path=self.ic_path
        )
    
    def test_fr_single_stimulus(self, ):
        gpt4v_single_fr_batch(
            key=self.key,
            dataset_name=self.dataset_name,
            image_path_list=self.image_path_list,
            image_name_list=self.image_name_list,
            mos_list=self.mos_list,
            text_prompt=self.text_prompt,
            save_folder_path=self.save_folder_path,
            save_file_name=self.save_file_name,
            mean_metric=self.mean_metric,
            prompt_pattern_str=self.psy_pattern + "-" + self.nlp_pattern,
            ic_path=self.ic_path
        )
    
    def test_fr_double_stimulus(self, ):
        gpt4v_double_fr_batch(
            key=self.key,
            dataset_name=self.dataset_name,
            image_path_list=self.image_path_list,
            image_name_list=self.image_name_list,
            mos_list=self.mos_list,
            text_prompt=self.text_prompt,
            save_folder_path=self.save_folder_path,
            save_file_name=self.save_file_name,
            fr_image_pair_num=self.fr_image_pair_num,
            mean_metric=self.mean_metric,
            prompt_pattern_str=self.psy_pattern + "-" + self.nlp_pattern,
            ic_path=self.ic_path
        )
    
    def test_fr_multiple_stimulus(self, ):
        gpt4v_multiple_fr_batch(
            key=self.key,
            dataset_name=self.dataset_name,
            image_path_list=self.image_path_list,
            image_name_list=self.image_name_list,
            mos_list=self.mos_list,
            text_prompt=self.text_prompt,
            save_folder_path=self.save_folder_path,
            save_file_name=self.save_file_name,
            fr_image_group_num=self.fr_image_group_num,
            mean_metric=self.mean_metric,
            prompt_pattern_str=self.psy_pattern + "-" + self.nlp_pattern,
            ic_path=self.ic_path
        )


if __name__ == "__main__":
    setup_seed(20)
    Tester = GPT4VTester(
        nr_image_pair_num=5, nr_image_group_num=2, fr_image_pair_num=5, fr_image_group_num=2,
    )
    Tester.test_fr_double()





