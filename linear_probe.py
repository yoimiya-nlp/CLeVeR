# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing
from data_preprocess import Preprocess
from model import ContrastiveModel, LinearProbe
from dataset import TrainData, DetectionTestData, DetectionProbeData, ClassificationProbeData

cpu_cont = 16
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, code_tokenizer, text_tokenizer, classifier, flag):
    # Build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 10
    # args.save_steps = 1
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)
    classifier.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    criterion = nn.CrossEntropyLoss()

    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.eval()
    classifier.train()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (func_input_ids, func_attention_mask, label) = [x.to(args.device) for x in batch]
            with torch.no_grad():
                func_representation, _ = model(func_input_ids, func_attention_mask, flag="probe")
            output = classifier(func_representation)
            loss = criterion(output, label)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % args.save_steps == 0:
                    if flag == "detection":
                        results = evaluate(args, model, code_tokenizer, text_tokenizer, eval_when_training=True, classifier=classifier)
                    else:
                        results = evaluate_cls(args, model, code_tokenizer, text_tokenizer, eval_when_training=True, classifier=classifier)

                    # Save model checkpoint
                    if results['eval_f1'] >= best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = args.to_checkpoint
                        probe_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(probe_dir):
                            os.makedirs(probe_dir)
                        classifier_to_save = classifier.module if hasattr(classifier, 'module') else classifier

                        #code_pretrain = model.code_encoder.encoder
                        #code_pretrain.save_pretrained("new_code_model")
                        #text_pretrain = model.desc_encoder.encoder
                        #text_pretrain.save_pretrained("new_text_model")

                        probe_dir = os.path.join(probe_dir, '{}'.format('classifier.bin'))
                        torch.save(classifier_to_save.state_dict(), probe_dir)
                        logger.info("Saving classifier checkpoint to %s", probe_dir)
                        torch.cuda.empty_cache()


def compute_similarity(code_embeds, desc_embeds):
    similarity = torch.nn.functional.cosine_similarity(code_embeds, desc_embeds)
    return similarity


def evaluate(args, model, code_tokenizer, text_tokenizer, eval_when_training=False, classifier=None):
    # Build dataloader
    eval_dataset = DetectionProbeData(code_tokenizer, text_tokenizer, args, flag='val')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval step
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    classifier.eval()

    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader):
        (func_input_ids, func_attention_mask, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            func_representation, _ = model(func_input_ids, func_attention_mask, flag="probe")
            output = classifier(func_representation)
            logit = F.softmax(output, dim=-1)

            # eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1

    # Calculate results
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_accuracy": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, code_tokenizer, text_tokenizer, eval_when_training=False, classifier=None):
    # Build dataloader
    test_dataset = DetectionProbeData(code_tokenizer, text_tokenizer, args, flag='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Test step
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    classifier.eval()

    logits = []
    y_trues = []
    for batch in tqdm(test_dataloader):
        (func_input_ids, func_attention_mask, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            func_representation, _ = model(func_input_ids, func_attention_mask, flag="probe")
            output = classifier(func_representation)
            logit = F.softmax(output, dim=-1)

            # eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1

    # Calculate results
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)

    result = {
        "test_accuracy": float(accuracy),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1)
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def evaluate_cls(args, model, code_tokenizer, text_tokenizer, eval_when_training=False, classifier=None):
    # Build dataloader
    eval_dataset = ClassificationProbeData(code_tokenizer, text_tokenizer, args, flag='val')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval step
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    classifier.eval()

    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader):
        (func_input_ids, func_attention_mask, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            func_representation, _ = model(func_input_ids, func_attention_mask, flag="probe")
            output = classifier(func_representation)
            logit = F.softmax(output, dim=-1)

            # eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1

    # Calculate results
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    #best_threshold = 0.5

    y_preds = np.argmax(logits, axis=1)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='weighted')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='weighted')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='weighted')
    result = {
        "eval_accuracy": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test_cls(args, model, code_tokenizer, text_tokenizer, eval_when_training=False, classifier=None):
    # Build dataloader
    test_dataset = ClassificationProbeData(code_tokenizer, text_tokenizer, args, flag='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Test step
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    classifier.eval()

    logits = []
    y_trues = []
    for batch in tqdm(test_dataloader):
        (func_input_ids, func_attention_mask, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            func_representation, _ = model(func_input_ids, func_attention_mask, flag="probe")
            output = classifier(func_representation)
            logit = F.softmax(output, dim=-1)

            # eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1

    # Calculate results
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    #best_threshold = 0.5

    y_preds = np.argmax(logits, axis=1)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_trues, y_preds)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='weighted')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='weighted')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='weighted')

    result = {
        "test_accuracy": float(accuracy),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1)
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def predict(args, model, code_tokenizer, text_tokenizer, best_threshold=0.0):
    # Build dataloader
    pred_dataset = DetectionTestData(code_tokenizer, text_tokenizer, args, flag='test')
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Predict step
    logger.info("***** Running Predict *****")
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in pred_dataloader:
        (inputs_ids, position_idx, csg_edge_mask, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(inputs_ids, position_idx, csg_edge_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # Output results
    logits = np.concatenate(logits, 0)
    y_preds = logits[:, 1] > best_threshold
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(pred_dataset.examples, y_preds):
            if pred:
                f.write(example.url + '\t' + '1' + '\n')
            else:
                f.write(example.url + '\t' + '0' + '\n')

def main():
    # Setup parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="the dataset name")

    parser.add_argument("--dataset_data_file", default=None, type=str,
                        help="An optional input dataset data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--from_checkpoint", default=None, type=str,
                        help="the checkpoint load from")
    parser.add_argument("--to_checkpoint", default=None, type=str,
                        help="the checkpoint save to")
    parser.add_argument("--pretrain_checkpoint", default=None, type=str,
                        help="the checkpoint save to")

    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--pretrain_text_model_name", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--pretrain_code_model_name", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run pred on the dev set.")
    parser.add_argument("--do_train_cls", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test_cls", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--hidden_size", default=768, type=int,
                        help="attention_hidden_size.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()

    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # Setup seed
    set_seed(args)

    #code_config = RobertaConfig.from_pretrained(args.pretrain_code_model_name)
    code_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_code_model_name)
    #text_config = RobertaConfig.from_pretrained(args.pretrain_text_model_name)
    text_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_text_model_name)

    vul_model = ContrastiveModel(args)

    for param in vul_model.parameters():
        param.requires_grad = False

    linear_probe = LinearProbe(input_dim=768, num_classes=2)
    linear_probe_cls = LinearProbe(input_dim=768, num_classes=10)

    # Train Phrase
    if args.do_train:
        train_dataset = DetectionProbeData(code_tokenizer, text_tokenizer, args, flag='train')
        checkpoint_prefix = args.pretrain_checkpoint + '/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        vul_model.load_state_dict(torch.load(output_dir))
        vul_model.to(args.device)

        #linear_prefix = args.from_checkpoint + '/classifier.bin'
        #probe_dir = os.path.join(args.output_dir, '{}'.format(linear_prefix))
        #linear_probe.load_state_dict(torch.load(probe_dir))
        linear_probe.to(args.device)
        train(args, train_dataset, vul_model, code_tokenizer, text_tokenizer, linear_probe, flag="detection")

    # Test Phrase
    results = {}
    if args.do_test:
        checkpoint_prefix = args.pretrain_checkpoint + '/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        vul_model.load_state_dict(torch.load(output_dir))
        vul_model.to(args.device)
        linear_prefix = args.to_checkpoint + '/classifier.bin'
        probe_dir = os.path.join(args.output_dir, '{}'.format(linear_prefix))
        linear_probe.load_state_dict(torch.load(probe_dir))
        linear_probe.to(args.device)
        result = test(args, vul_model, code_tokenizer, text_tokenizer, classifier=linear_probe)
        print("Detection Test result: ", result)

    if args.do_train_cls:
        train_dataset = ClassificationProbeData(code_tokenizer, text_tokenizer, args, flag='train')
        checkpoint_prefix = args.pretrain_checkpoint + '/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        vul_model.load_state_dict(torch.load(output_dir))
        vul_model.to(args.device)

        #linear_prefix = args.from_checkpoint + '/classifier.bin'
        #probe_dir = os.path.join(args.output_dir, '{}'.format(linear_prefix))
        #linear_probe_cls.load_state_dict(torch.load(probe_dir))
        linear_probe_cls.to(args.device)
        train(args, train_dataset, vul_model, code_tokenizer, text_tokenizer, linear_probe_cls, flag="classification")

    if args.do_test_cls:
        checkpoint_prefix = args.pretrain_checkpoint + '/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        vul_model.load_state_dict(torch.load(output_dir))
        vul_model.to(args.device)
        linear_prefix = args.to_checkpoint + '/classifier.bin'
        probe_dir = os.path.join(args.output_dir, '{}'.format(linear_prefix))
        linear_probe_cls.load_state_dict(torch.load(probe_dir))
        linear_probe_cls.to(args.device)
        result = test_cls(args, vul_model, code_tokenizer, text_tokenizer, classifier=linear_probe_cls)
        print("Classification Test result: ", result)


    if args.do_predict:
        checkpoint_prefix = args.from_checkpoint + '/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        vul_model.load_state_dict(torch.load(output_dir))
        vul_model.to(args.device)
        predict(args, vul_model, code_tokenizer, text_tokenizer, best_threshold=0.5)
        print("Predict over")

    return results


if __name__ == "__main__":
    main()
