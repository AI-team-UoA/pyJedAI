'''
Supervised methods for end-to-end deduplication.
Contains methods using pre-trained BERT embeddings.

Needs:
- Train data
- Validation data
- Test data
'''
import sys
import warnings
from time import time
from typing import List, Tuple, Dict
import json
from sklearn.metrics import (
    f1_score,
    classification_report,
    precision_recall_fscore_support
)
from .supervised_utils import (
    build_optimizer,
    initialize_gpu_seed,
    load_data,
    DataType,
    DeepMatcherProcessor,
    train,
    predict,
    Config,
    Evaluation,
    save_model
)
import pandas as pd
from .datamodel import Data
from .utils import create_entity_index, are_matching
import warnings
warnings.filterwarnings('always')

class PretrainedSupervisedER():

    _method_name = "PretrainedSupervisedER"
    _method_short_name = "PSER"
    _method_info = ""

    def __init__(self,
                 model_type: str,
                 model_name: str,
                 train_batch_size: int = 16,
                 eval_batch_size: int = 16,
                 max_seq_length: int = 180,
                 num_epochs: int = 1,
                 seed: int =22,
                 do_lower_case: bool = True):
        self.model_type = model_type
        # Available model types: 'roberta' 'bert' 'distilbert' 
        #   'sdistilroberta' 'sminilm' 'albert' 'smpnet' 'xlnet'

        self.model_name = model_name
        # Available models: 'roberta-base' 'bert-base-uncased' 
        #   'distilbert-base-uncased' 'sentence-transformers/all-distilroberta-v1' 
        #   'sentence-transformers/all-MiniLM-L12-v2' 'albert-base-v2' 
        #   'sentence-transformers/all-mpnet-base-v2' 'xlnet-base-cased'

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.seed = seed
        self.do_lower_case = do_lower_case

        # Private attributes
        self._learning_rate: float = 2e-5
        self._adam_eps: float = 1e-8
        self._warmup_steps: int = 0
        self._weight_decay: float = 0.0
        self._max_grad_norm: float = 1.0
        self._save_model_after_epoch: bool = False

    def fit(self):
        pass

    def fit_blocks(self,
                candidate_pairs: dict,
                data: Data,
                train_split_size: float = 0.6,
                test_split_size: float = 0.2,
                validation_split_size: float = 0.2,
                experiment_title='Experiment',
                verbose: int = 0) -> any:

        if data.ground_truth is None:
            raise AttributeError("Can't use this method without ground-truth file")

        self.train_split_size = train_split_size
        self.test_split_size = test_split_size
        self.validation_split_size = validation_split_size

        device, n_gpu = initialize_gpu_seed(self.seed)
        processor = DeepMatcherProcessor()
        label_list = processor.get_labels()
        if verbose:
            print("Training with {} labels: {}".format(len(label_list), label_list))

        train_examples, validation_examples, test_examples = \
            processor.split(candidate_pairs,
                            data,
                            train_split_size,
                            test_split_size,
                            validation_split_size)

        #
        # Selecting pretrained model
        #
        config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[self.model_type]
        if config_class is not None:
            config = config_class.from_pretrained(self.model_name)
            tokenizer = tokenizer_class.from_pretrained(self.model_name,
                                                        do_lower_case=self.do_lower_case)
            model = model_class.from_pretrained(self.model_name, config=config)
            model.to(device)
        else: # SBERT Models
            tokenizer = tokenizer_class.from_pretrained(self.model_name)
            model = model_class.from_pretrained(self.model_name)
            model.to(device)
        if verbose:
            print("Initialized {}-model".format(self.model_type))

        #
        # Training
        #
        training_data_loader = load_data(train_examples,
                                            label_list,
                                            tokenizer,
                                            self.max_seq_length,
                                            self.train_batch_size,
                                            DataType.TRAINING, self.model_type)
        if verbose:
            print("Loaded {} training examples".format(len(train_examples)))

        num_train_steps = len(training_data_loader) * self.num_epochs

        #
        # Optimizer
        #
        optimizer, scheduler = build_optimizer(model,
                                               num_train_steps,
                                               self._learning_rate,
                                               self._adam_eps,
                                               self._warmup_steps,
                                               self._weight_decay)
        if verbose:
            print("Built optimizer: {}".format(optimizer))

        #
        # Validation
        #
        evaluation_data_loader = load_data(validation_examples,
                                           label_list,
                                           tokenizer,
                                           self.max_seq_length,
                                           self.eval_batch_size,
                                           DataType.EVALUATION,
                                           self.model_type)

        evaluation = Evaluation(evaluation_data_loader,
                                experiment_title,
                                len(label_list),
                                self.model_type, None)
        if verbose:
            print("Loaded and initialized evaluation examples {}".format(len(validation_examples)))

        t1 = time()
        train(training_data_loader,
              model,
              optimizer,
              scheduler,
              evaluation,
              device,
              self.num_epochs,
              self._max_grad_norm,
              self._save_model_after_epoch,
              experiment_name=experiment_title,
              model_type=self.model_type, 
              output_dir=None)
        t2 = time()
        training_time = t2-t1

        #
        # Testing
        #
        if verbose:
            print("Loaded {} test examples".format(len(test_examples)))
        test_data_loader = load_data(validation_examples,
                                    label_list,
                                    tokenizer,
                                    self.max_seq_length,
                                    self.eval_batch_size,
                                    DataType.TEST,
                                    self.model_type)

        include_token_type_ids = False
        if self.model_type == 'bert':
            include_token_type_ids = True
        
        t1 = time()
        predictions, labels = predict(model, device, test_data_loader, include_token_type_ids)
        predictions_df = pd.DataFrame({'predictions': predictions, 'labels': labels})
        t2 = time()
        testing_time = t2-t1
        
        return predictions, labels

    def evaluate(self, predictions, true_labels):
        simple_accuracy = (predictions == true_labels).mean()
        f1 = f1_score(y_true=true_labels, y_pred=predictions)
        scores = precision_recall_fscore_support(y_true=true_labels, y_pred=predictions)
        report = classification_report(true_labels, predictions)
        print("Prediction done for test examples.F1: {}, Simple Accuracy: {}".format(
                f1, simple_accuracy))