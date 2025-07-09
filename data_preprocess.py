from torch.utils.data import Dataset, TensorDataset
import logging
from tqdm import tqdm
import json
import random
import numpy as np
import torch
import pickle
from generate_example import generate_description
import argparse
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset


logger = logging.getLogger(__name__)

class Preprocess(Dataset):
    def __init__(self, args, file_path=''):
        self.train_examples = []
        self.test_examples = []
        self.examples = []
        self.split_ratio = 0.8
        dataset_file = file_path

        dataset = []
        # [idx], func, name, label, cwe_id, source, sink, [description]
        idx = 0
        with open(dataset_file) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                js["idx"] = str(idx)
                dataset.append(js)
                idx += 1

        random.shuffle(dataset)

        train_size = int(len(dataset) * self.split_ratio)
        train_data = dataset[:train_size]
        print("train_data: ", len(train_data))
        test_data = dataset[train_size:]
        print("test_data: ", len(test_data))

        self.train_examples = [generate_description(x) for x in tqdm(train_data, total=len(train_data))]
        with open('preprocessed_data/' + args.dataset_name + '_train.pkl', 'wb') as f:
            pickle.dump(self.train_examples, f)
        print("train_examples: ", len(self.train_examples))

        self.test_examples = [generate_description(x) for x in tqdm(test_data, total=len(test_data))]
        with open('preprocessed_data/' + args.dataset_name + '_test.pkl', 'wb') as f:
            pickle.dump(self.test_examples, f)
        print("test_examples: ", len(self.test_examples))

        for idx, example in enumerate(self.examples[:2]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            logger.info("func: {}".format(example.func))
            logger.info("label: {}".format(example.label))
            logger.info("source: {}".format(example.source))
            logger.info("sink: {}".format(example.sink))
            logger.info("description: {}".format(example.description))
            logger.info("cwe_id: {}".format(example.cwe_id))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="the dataset txt file path")
    parser.add_argument("--dataset_name", default=None, type=str, required=True,
                        help="the dataset name")

    parser.add_argument("--pretrain_text_model_name", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--pretrain_code_model_name", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--program_language", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    set_seed(args)
    #text_config = RobertaConfig.from_pretrained(args.pretrain_text_model_name)
    #text_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_text_model_name)
    Preprocess(args, file_path=args.dataset)
