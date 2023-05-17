from collections import defaultdict
import random, os, csv, argparse, json, logging, sys, torch
import numpy as np
from enum import Enum
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer, XLNetTokenizer, \
    XLNetForSequenceClassification, XLNetConfig, XLMForSequenceClassification, XLMConfig, XLMTokenizer, \
    RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, DistilBertConfig, \
    DistilBertForSequenceClassification, DistilBertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule    
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, \
    AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, \
    T5Config, T5Tokenizer, T5ForConditionalGeneration
import re
import random

from .utils import create_entity_index, are_matching
from .datamodel import Data


def build_optimizer(model, num_train_steps, learning_rate, adam_eps, warmup_steps, weight_decay):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)

    return optimizer, scheduler

def initialize_gpu_seed(seed: int):
    device, n_gpu = setup_gpu()

    init_seed_everywhere(seed, n_gpu)

    return device, n_gpu


def init_seed_everywhere(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_gpu():
    # Setup GPU parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.info("We use the device: '{}' and {} gpu's. Important: distributed and 16-bits training "
                 "is currently not implemented! ".format(device, n_gpu))

    return device, n_gpu

class DataType(Enum):
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    TEST = "Test"

def load_data(examples, label_list, tokenizer, max_seq_length, batch_size, data_type: DataType, model_type):
    logging.info("***** Convert Data to Features (Word-Piece Tokenizing) [{}] *****".format(data_type))

    features = convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_mode="classification",
                                            cls_token_at_end=bool(model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta']),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,)

    logging.info("***** Build PyTorch DataLoader with extracted features [{}] *****".format(data_type))
    logging.info("  Num examples = %d", len(examples))
    logging.info("  Batch size = %d", batch_size)
    logging.info("  Max Sequence Length = %d", max_seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if data_type == DataType.TRAINING:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    return DataLoader(data, sampler=sampler, batch_size=batch_size)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label: int = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DeepMatcherProcessor(object):
    """Processor for preprocessed DeepMatcher data sets"""

    def _tokenize_entity(self, entity: str) -> list:
        return ' '.join((set(filter(None, re.split('[\\W_]', entity.lower())))))

    def split(self,
              candidate_pairs: dict,
              data: Data,
              train_split_size: float = 0.6,
              test_split_size: float = 0.2,
              validation_split_size: float = 0.2,
              shuffle=True) -> Tuple[dict, dict, dict]:

        entities_d1 = data.dataset_1[data.attributes_1] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist()

        if not data.is_dirty_er:
            entities_d2 = data.dataset_2[data.attributes_2] \
                    .apply(" ".join, axis=1) \
                    .apply(self._tokenize_entity) \
                    .values.tolist()

        pairs = []
        matches = {}
        total_pairs = 0
        all_ids = None
        if isinstance(candidate_pairs, dict) and isinstance(list(candidate_pairs.values())[0], set):
            # case of candidate pairs, entity-id -> {entity-id, ..}, i.e result from meta-blocking
            all_ids  = set()
            for _, (gid1, gid2) in data.ground_truth.iterrows():
                id1 = data._ids_mapping_1[gid1]
                id2 = data._ids_mapping_1[gid2] if data.is_dirty_er else data._ids_mapping_2[gid2]
                if (id1 in candidate_pairs and id2 in candidate_pairs[id1]) or   \
                    (id2 in candidate_pairs and id1 in candidate_pairs[id2]):
                    total_pairs+=1
                    # id1 and id2 are a match
                    if id1 not in matches:
                        matches[id1] = set()
                    if id2 not in matches:
                        matches[id2] = set()
                    matches[id1].add(id2)
                    matches[id2].add(id1)
                    all_ids.add(id1)
                    all_ids.add(id2)
                    pairs.append((entities_d1[id1], entities_d1[id2] if data.is_dirty_er \
                                        else entities_d2[id2 - data.dataset_limit], "1"))
        else: # blocks case
            entity_index = create_entity_index(candidate_pairs, data.is_dirty_er)
            for _, (gid1, gid2) in data.ground_truth.iterrows():
                id1 = data._ids_mapping_1[gid1]
                id2 = data._ids_mapping_1[gid2] if data.is_dirty_er \
                                                else data._ids_mapping_2[gid2]
                if id1 in entity_index and id2 in entity_index and \
                        are_matching(entity_index, id1, id2):
                    total_pairs+=1
                    # print('here')
                    # id1 and id2 are a match
                    pairs.append((entities_d1[id1], entities_d1[id2] if data.is_dirty_er \
                                        else entities_d2[id2 - data.dataset_limit], "1"))
                    if id1 not in matches:
                        matches[id1] = set()
                    if id2 not in matches:
                        matches[id2] = set()
                    matches[id1].add(id2)
                    # matches[id2].add(id1)
            all_ids = set(entity_index.keys())

        non_matches_count = 0
        diff_pairs = []
        for entity_id, pairs_ids in matches.items():
            non_matches = all_ids.difference(pairs_ids)
            for no_match_id in non_matches:
                entity_1 = entities_d1[entity_id] if entity_id < data.dataset_limit else entities_d2[entity_id-data.dataset_limit]
                entity_2 = entities_d1[no_match_id] if no_match_id < data.dataset_limit else entities_d2[no_match_id-data.dataset_limit]
                diff_pairs.append((entity_1, entity_2, "0"))
                non_matches_count += 1
            # print(len(all_ids), " - ", len(pairs_ids), " = ", len(non_matches))

        # print(total_pairs)
        # print(len(pairs))
        # print(len(diff_pairs))
        pairs = pairs[:100]
        diff_pairs = diff_pairs[:100]
        print("Ratio of true positives to true negatives: ", len(matches)/non_matches_count)
        print("Ratio of true positives to all pairs: ", len(matches)/(len(pairs)+len(diff_pairs)))

        train, test, validation = [], [], []

        if shuffle:
            random.shuffle(diff_pairs)
            random.shuffle(pairs)

        all_pairs = pairs + diff_pairs
        all_pairs_count = len(all_pairs)
        num_of_training_data = int(train_split_size*all_pairs_count)
        num_of_test_data = int(test_split_size*all_pairs_count)
        num_of_validation_data = int(validation_split_size*all_pairs_count)

        print(
            "#pairs_count: ", len(pairs),
            "\n#diff_pairs: ", len(diff_pairs)
        )

        train.extend(
            [InputExample(guid='train-'+str(i),
                          text_a=pairs[i][0],
                          text_b=pairs[i][1],
                          label=pairs[i][2]) for i in range(0,int(train_split_size*len(pairs)))])
        # print(0," -> ",int(train_split_size*len(pairs)))

        train.extend(
            [InputExample(guid='train-'+str(i),
                          text_a=diff_pairs[i][0],
                          text_b=diff_pairs[i][1],
                          label=diff_pairs[i][2]) for i in range(0,int(train_split_size*len(diff_pairs)))])
        # print(0," -> ", int(train_split_size*len(diff_pairs)))

        test.extend(
            [InputExample(guid='test-'+str(i),
                          text_a=pairs[i][0],
                          text_b=pairs[i][1],
                          label=pairs[i][2]) for i in range(int(train_split_size*len(pairs)),
                                                            int(train_split_size*len(pairs))+int(test_split_size*len(pairs)))])
        # print(int(train_split_size*len(pairs))," -> ",int(train_split_size*len(pairs))+int(test_split_size*len(pairs)))
        
        test.extend(
            [InputExample(guid='test-'+str(i),
                          text_a=diff_pairs[i][0],
                          text_b=diff_pairs[i][1],
                          label=diff_pairs[i][2]) for i in range(int(train_split_size*len(diff_pairs)),
                                                                int(train_split_size*len(diff_pairs))+int(test_split_size*len(diff_pairs)))])
        # print(int(train_split_size*len(diff_pairs))," -> ",int(train_split_size*len(diff_pairs))+int(test_split_size*len(diff_pairs)))

        validation.extend(
            [InputExample(guid='validation-'+str(i),
                          text_a=pairs[i][0],
                          text_b=pairs[i][1],
                          label=pairs[i][2]) for i in range(int((train_split_size+test_split_size)*len(pairs)), len(pairs))])
            
        # print(int((train_split_size+test_split_size)*len(pairs))," -> ",int(len(pairs)))

        validation.extend(
            [InputExample(guid='validation-'+str(i),
                          text_a=diff_pairs[i][0],
                          text_b=diff_pairs[i][1],
                          label=diff_pairs[i][2]) for i in range(int((train_split_size+test_split_size)*len(diff_pairs)), len(diff_pairs))])
        # print(int((train_split_size+test_split_size)*len(diff_pairs))," -> ",len(diff_pairs))
        # print("Total: ", len(train)+len(test)+len(validation))

        return train, validation, test

    # def get_train_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "train.csv")), 
    #         self._read_tsv(os.path.join(data_dir, "tableA.csv")),
    #         self._read_tsv(os.path.join(data_dir, "tableB.csv")),
    #         "train")

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "valid.csv")),
    #         self._read_tsv(os.path.join(data_dir, "tableA.csv")),
    #         self._read_tsv(os.path.join(data_dir, "tableB.csv")),
    #         "dev")

    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "test.csv")),
    #         self._read_tsv(os.path.join(data_dir, "tableA.csv")),
    #         self._read_tsv(os.path.join(data_dir, "tableB.csv")),
    #         "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    
def train(train_dataloader,
          model,
          optimizer,
          scheduler,
          evaluation,
          device,
          num_epocs,
          max_grad_norm,
          save_model_after_epoch,
          experiment_name,
          model_type,
          output_dir=None):

    logging.info("***** Run training *****")
    
    # tb_writer = SummaryWriter(os.path.join("./", experiment_name))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # we are interested in 0 shot learning, therefore we already evaluate before training.
    eval_results = evaluation.evaluate(model, device, -1)
    # for key, value in eval_results.items():
        # tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
        # print('eval_{}'.format(key), value, global_step)
    for epoch in trange(int(num_epocs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            if model_type not in ['distilbert', 'smpnet']:
                inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            global_step += 1

            # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            # print('lr ', scheduler.get_lr()[0], global_step)
            # tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
            # print('loss ', (tr_loss - logging_loss), global_step)
            logging_loss = tr_loss

        eval_results = evaluation.evaluate(model, device, epoch)
        for key, value in eval_results.items():
            print('eval_{}'.format(key))
            # tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        # if save_model_after_epoch:
        #     save_model(model, experiment_name, output_dir, epoch=epoch)

    # tb_writer.close()

def predict(model,
            device,
            test_data_loader,
            include_token_type_ids=False):
    nb_prediction_steps = 0
    predictions = None
    labels = None

    for batch in tqdm(test_data_loader, desc="Test"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            if include_token_type_ids:
                inputs['token_type_ids'] = batch[2]

            outputs = model(**inputs)
            _, logits = outputs[:2]

        nb_prediction_steps += 1

        if predictions is None:
            predictions = logits.detach().cpu().numpy()
            labels = inputs['labels'].detach().cpu().numpy()
        else:
            predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=0)

    # remember, the logits are simply the output from the last layer, without applying an activation function (e.g. sigmoid).
    # for a simple classification this is also not necessary, we just take the index of the neuron with the maximal output.
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class, labels

    # simple_accuracy = (predicted_class == labels).mean()
    # f1 = f1_score(y_true=labels, y_pred=predicted_class)
    # scores = precision_recall_fscore_support(y_true=labels, y_pred=predicted_class)
    # report = classification_report(labels, predicted_class)

    # return simple_accuracy, f1, report, scores, pd.DataFrame({'predictions': predicted_class, 'labels': labels})
    
def _truncate_seq_pair(id, tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    if len(tokens_a) + len(tokens_b) > max_length:
        logging.info("Sample with ID '{}' was too long (tokens_a:{}, tokens_b:{}). "
                     "Max_seq_length is {}, so we reduce it in a smart way".format(id, len(tokens_a), len(tokens_b),
                                                                                   max_length))

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(example.guid, tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

class Evaluation:

    def __init__(self,
                 evaluation_data_loader,
                 experiment_name,
                 n_labels,
                 model_type,
                 model_output_dir=None):
        self.model_type = model_type
        self.evaluation_data_loader = evaluation_data_loader
        self.n_labels = n_labels
        self.verbose_to_file = False
        if model_output_dir:
            self.output_path = os.path.join(model_output_dir, experiment_name, "eval_results.txt")
            self.verbose_to_file = True

    def evaluate(self, model, device, epoch):
        nb_eval_steps = 0
        eval_loss = 0.0
        predictions = None
        labels = None

        for batch in tqdm(self.evaluation_data_loader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

                if self.model_type not in ['distilbert', 'smpnet']:
                    inputs['token_type_ids'] = batch[2] if self.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]     # logits are always part of the output (see BertForSequenceClassification documentation),
                                                        # while loss is only available if labels are provided. Therefore the logits are here to find on first position.

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if predictions is None:
                predictions = logits.detach().cpu().numpy()
                labels = inputs['labels'].detach().cpu().numpy()
            else:
                predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        # remember, the logits are simply the output from the last layer, without applying an activation function (e.g. sigmoid).
        # for a simple classification this is also not necessary, we just take the index of the neuron with the maximal output.
        predicted_class = np.argmax(predictions, axis=1)

        simple_accuracy = (predicted_class == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=predicted_class)
        report = classification_report(labels, predicted_class)

        result = {'eval_loss': eval_loss,
                  'simple_accuracy': simple_accuracy,
                  'f1_score': f1}

        if self.verbose_to_file:
            with open(self.output_path, "a+") as writer:
                tqdm.write("***** Eval results after epoch {} *****".format(epoch))
                writer.write("***** Eval results after epoch {} *****\n".format(epoch))
                for key in sorted(result.keys()):
                    tqdm.write("{}: {}".format(key, str(result[key])))
                    writer.write("{}: {}\n".format(key, str(result[key])))

                tqdm.write(report)
                writer.write(report + "\n")
        else:
            tqdm.write("***** Eval results after epoch {} *****".format(epoch))
            for key in sorted(result.keys()):
                tqdm.write("{}: {}".format(key, str(result[key])))

        return result


class Config():
    DATA_PREFIX = "data"
    EXPERIMENT_PREFIX = "experiments"

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        #'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlnet': (None, AutoModelForSequenceClassification, AutoTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        'sdistilroberta': (None, AutoModelForSequenceClassification, AutoTokenizer),
        #'st5': (None, AutoTokenizer, T5ForConditionalGeneration),
        'sminilm': (None, AutoModelForSequenceClassification, AutoTokenizer),
        'smpnet': (None, AutoModelForSequenceClassification, AutoTokenizer),
        #'glove': (None, AutoModelForSequenceClassification, AutoTokenizer),        
    }


def write_config_to_file(args, model_output_dir: str, experiment_name: str):
    config_path = os.path.join(model_output_dir, experiment_name, "args.json")

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def read_arguments_train():
    parser = argparse.ArgumentParser(description='Run training with following arguments')

    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--data_processor', default=None, type=str, required=True)
    parser.add_argument('--model_name_or_path', default="pre_trained_model/bert-base-uncased", type=str, required=True)
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument('--do_lower_case', action='store_true', default=True)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=3.0, type=float)
    parser.add_argument('--save_model_after_epoch', action='store_true')
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    args.data_path = os.path.join(Config.DATA_PREFIX, args.data_dir)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    logging.info("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        logging.info("argument: {}={}".format(argument, getattr(args, argument)))

    return args


def read_arguments_prediction():
    parser = argparse.ArgumentParser(description='Run testing with the following arguments')

    parser.add_argument('--trained_model_for_prediction', action="store", required=True, type=str)
    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--data_processor', default=None, type=str, required=True)
    parser.add_argument('--do_lower_case', action='store_true', default=True)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--test_batch_size', default=8, type=int)

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    args.data_path = os.path.join(Config.DATA_PREFIX, args.data_dir)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    logging.info("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        logging.info("argument: {}={}".format(argument, getattr(args, argument)))

    return args


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        datefmt="%H:%M:%S",
                        stream=sys.stdout,
                        #filename='log_file_name.log',
                        )

    logging.getLogger('bert-classifier-entity-matching')
    
def save_model(model, experiment_name, model_output_dir, epoch=None, tokenizer=None):
    if epoch:
        output_sub_dir = os.path.join(model_output_dir, experiment_name, "epoch_{}".format(epoch))
    else:
        output_sub_dir = os.path.join(model_output_dir, experiment_name)

    os.makedirs(output_sub_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_save.save_pretrained(output_sub_dir)

    if tokenizer:
        tokenizer.save_pretrained(output_sub_dir)

    return output_sub_dir


def load_model(model_dir, do_lower_case):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)

    return model, tokenizer    