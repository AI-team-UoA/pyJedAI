import ollama
from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation
from abc import ABC, abstractmethod
import pandas as pd
from typing import Literal, Union
from ollama._types import ResponseError 
import networkx
from time import time
from tqdm import tqdm

from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

DEFAULT_SYSTEM_PROMPT = """You are given two record descriptions and your task is to identify
if the records refer to the same entity or not.

You must answer with just one word:
True. if the records are referring to the same entity,
False. if the records are referring to a different entity."""

class AbstrackLLMMatching(PYJEDAIFeature):
    """Abstract class for Matching with LLMs
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    # @abstractmethod
    def _apply_main_processing(self) -> dict:
        pass

    @abstractmethod
    def _configuration(self) -> dict:
        pass
    
    def stats(self) -> None:
        pass
    
    def report(self) -> None:
        """Prints LLM Matching method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes from D1:\n\t" + ', '.join(c for c in (self.attributes_1 if self.attributes_1 is not None \
                else self.data.dataset_1.columns)) +
            ("\nAttributes from D2:\n\t" + ', '.join(c for c in (self.attributes_2 if self.attributes_2 is not None \
                else self.data.dataset_2.columns)) if not self.data.is_dirty_er else "") +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )


    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        total_matching_pairs = len(prediction)
        
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er \
                                                else self.data._ids_mapping_2[id2]
            
            if (id1, id2) in prediction or   \
                (id2, id1) in prediction:
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives, 
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)


    def export_to_df(self, prediction) -> pd.DataFrame:
        """creates a dataframe with the predicted pairs

        Args:
            prediction (any): Predicted candidate pairs

        Returns:
            pd.DataFrame: Dataframe with the predicted pairs
        """
        
        if self.data.is_dirty_er:
            pairs = [(self.data._gt_to_ids_reversed_1[id1], self.data._gt_to_ids_reversed_1[id2]) for id1, id2 in prediction]
        else:
            pairs = [(self.data._gt_to_ids_reversed_1[id1], self.data._gt_to_ids_reversed_2[id2]) for id1, id2 in prediction]
        
        pairs_df = pd.DataFrame(data=pairs, columns=['id1', 'id2'])
        
        return pairs_df
    
    
class OllamaMatching(AbstrackLLMMatching):
    """Ollama Matching
    """
    
    _method_name = "Ollama Matching"
    _method_short_name: str = "OLLM"
    _method_info = "Utilizes an LLM and predicts which pairs are matching"


    def __init__(self, llm_model: str):
        
        super().__init__()
        self.llm = llm_model
        self.execution_time = 0
        
        
        self.model_name : str
        self.candidate_pairs : list = list()

        try:
            if not llm_model in ollama.list()['models']:
                print(f"Pulling model {llm_model} from ollama")
                ollama.pull(llm_model)
        except ResponseError as e:
            print(e.__str__())
        

    def process(self,
            prediction: Union[networkx.Graph, dict],
            data: Data, 
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            create_examples: bool = False,
            examples: dict = None,
            suffix: Literal ['z', 'ft','tf'] = 'z',
            attributes: any = None,
            tqdm_disable: bool = False) -> dict:
        
        start_time = time()
        self.data = data
        self.tqdm_disable, self.prediction, self.attributes = tqdm_disable, prediction, attributes
        self.system_prompt = f"{system_prompt}\n"
        
        if isinstance(prediction, networkx.Graph):
            self._extract_candidate_pairs_from_graph(prediction)
        else:
            self._extract_candidate_pairs_from_blocks(prediction)

        if suffix != 'z':  
            example_records = self._extract_example_records(examples, create_examples)
            example_cnt = 1
            
            if suffix == 'ft':
                if 'false' in example_records: 
                    for example in example_records['false']:
                            example_str = f"\nExample {example_cnt}\nrecord 1: {example[0]}\nrecord 2: {example[1]}\nAnswer: False."
                            example_cnt += 1
                            self.system_prompt = f"{system_prompt}{example_str}"
                if 'true' in example_records: 
                    for example in example_records['true']:
                            example_str = f"\nExample {example_cnt}\nrecord 1: {example[0]}\nrecord 2: {example[1]}\nAnswer: True."
                            example_cnt += 1
                            self.system_prompt = f"{system_prompt}{example_str}"
            else:
                if 'true' in example_records: 
                    for example in example_records['true']:
                            example_str = f"\nExample {example_cnt}\nrecord 1: {example[0]}\nrecord 2: {example[1]}\nAnswer: True."
                            example_cnt += 1
                            self.system_prompt = f"{system_prompt}{example_str}"                    
                if 'false' in example_records: 
                    for example in example_records['false']:
                            example_str = f"\nExample {example_cnt}\nrecord 1: {example[0]}\nrecord 2: {example[1]}\nAnswer: False."
                            example_cnt += 1
                            self.system_prompt = f"{system_prompt}{example_str}"
        
        self.model_name = f'{self.llm}-{suffix}'
        ollama.create(model=self.model_name, from_=self.llm, system=self.system_prompt)
        print(f"Created ollama model {self.model_name}")


        self._progress_bar_2 = tqdm(total = len(self.candidate_pairs),
                                  desc = f"{self._method_name} [{self.model_name}]",
                                  disable = self.tqdm_disable)


        self.pairs = list()
        for id1, id2 in self.candidate_pairs:
            r1, r2 = self._extract_records(id1, id2)
            

            query = f"record 1: {r1}, record 2: {r2}. Answer with True. or False."

            # return
            resp = ollama.chat(
                model = self.model_name,
                messages = [{'role': 'user', 'content': query}],
                options = {'stop': ['\n','.']},
                stream = False
            )
            
            self._progress_bar_2.update(1)
            
            if resp['message']['content'] == 'True':
                self.pairs.append((id1,id2))
            
            # responses.append(resp['message']['content'])

            # gt_value = 'True' if (dt1_id, dt2_id) in gt_set else 'False'
        
        self.execution_time = time() - start_time
        ollama.delete(self.model_name)


        
        
        
        return self.pairs
    
    def _extract_candidate_pairs_from_graph(self, prediction: networkx.Graph):
        uv_pairs = list(prediction.edges(data=False))
        for pair in uv_pairs: 
            index_1, index_2 = (pair[0], pair[1]) if pair[0] < len(self.data.dataset_1) else (pair[1], pair[0])
            self.candidate_pairs.append((index_1, index_2))
    
    
    def _extract_candidate_pairs_from_blocks(self, prediction: dict) -> None:
        self._progress_bar = tqdm(total = len(prediction),
                                  desc = f"{self._method_name} [Extracting Candidate Pairs]",
                                  disable = self.tqdm_disable)
        
        all_blocks = list(prediction.values())
        
        if 'Block' in str(type(all_blocks[0])):
            #extract from raw blocks
            self._extract_from_raw_blocks(prediction)
        elif isinstance(all_blocks[0], set):
            self._extract_from_candidate_pairs(prediction)
        else:
            raise AttributeError("Wrong type of Blocks")
        

    def _extract_from_raw_blocks(self, blocks: dict):
        if self.data.is_dirty_er:
            for _, block in blocks.items():
                entities_array = list(block.entities_D1)
                for index_1 in range(0, len(entities_array), 1):
                    for index_2 in range(index_1+1, len(entities_array), 1):
                        self.candidate_pairs.append((entities_array[index_1], entities_array[index_2]))
                self._progress_bar.update(1)
        else:
            for _, block in blocks.items():
                for entity_id1 in block.entities_D1:
                    for entity_id2 in block.entities_D2:
                        self.candidate_pairs.append((entity_id1, entity_id2))
                self._progress_bar.update(1)

    def _extract_from_candidate_pairs(self, blocks: dict):
        for entity_id, candidates in blocks.items():
            for candidate_id in candidates:
                self.candidate_pairs.append((entity_id, candidate_id))
            self._progress_bar.update(1)

    def _extract_records(self, id1: int, id2: int) -> tuple:
        id_column_1 = self.data.id_column_name_1
        id_column_2 = self.data.id_column_name_2 if not self.data.is_dirty_er else self.data.id_column_name_1

        
        if isinstance(self.attributes, dict):
            r1 = self.data.entities.iloc[id1][self.attributes.items()].str.cat(sep=' ').lower().strip()
            r2 = self.data.entities.iloc[id2][self.attributes.items()].str.cat(sep=' ').lower().strip()
        elif isinstance(self.attributes, list):
            r1 = self.data.entities.iloc[id1][self.attributes].str.cat(sep=' ').lower().strip()
            r2 = self.data.entities.iloc[id2][self.attributes].str.cat(sep=' ').lower().strip()
        else: 
            r1 = self.data.entities.iloc[id1].drop(id_column_1).str.cat(sep=' ').lower().strip()
            r2 = self.data.entities.iloc[id2].drop(id_column_2).str.cat(sep=' ').lower().strip()

        return (r1, r2)
   
    def _extract_example_records(self, examples: dict, create_examples: bool) -> dict:

        examples_records = {}

        if examples:
            if not ('true' or 'false' in examples):
                raise Exception("Must at least contains key with ids for 'true' or 'false'. (e.g \{'true': [(10, 5)], ...\})")

            if 'true' in examples:
                examples_records['true'] = []
                for true_1, true_2 in examples['true']:
                    id1 = self.data._ids_mapping_1[str(true_1)]
                    id2 = self.data._ids_mapping_1[str(true_2)] if self.data.is_dirty_er \
                                                        else self.data._ids_mapping_2[str(true_2)]
                    examples_records['true'].append(self._extract_records(id1, id2))

            if 'false' in examples:
                examples_records['false'] = []
                for false_1, false_2 in examples['false']:
                    id1 = self.data._ids_mapping_1[str(false_1)]
                    id2 = self.data._ids_mapping_1[str(false_2)] if self.data.is_dirty_er \
                                                        else self.data._ids_mapping_2[str(false_2)]
                    examples_records['false'].append(self._extract_records(id1, id2))
        elif create_examples:
            emb = EmbeddingsNNBlockBuilding(vectorizer='sdistilroberta', similarity_search='faiss')
            _, g = emb.build_blocks(data=self.data, num_of_clusters=1, top_k=1,similarity_distance = 'cosine',with_entity_matching=True, load_embeddings_if_exist=True)
            edges_by_weight = sorted(g.edges(data=True), key=lambda x: x[2]['weight'])
            uv_pairs = [(u, v) for u, v, _ in edges_by_weight]

            examples_records['true'] = []
            examples_records['false'] = []
            
            
            if not isinstance(self.data.ground_truth, pd.DataFrame):
                true_pair = uv_pairs[-1] 
                false_pair = uv_pairs[0]
                
                true_1, true_2 = (true_pair[0], true_pair[1]) if true_pair[0] < len(self.data.dataset_1) else (true_pair[1], true_pair[0])
                false_1, false_2 = (false_pair[0], false_pair[1]) if false_pair[0] < len(self.data.dataset_1) else (false_pair[1], false_pair[0])

                examples_records['true'].append(self._extract_records(true_1, true_2))
                examples_records['false'].append(self._extract_records(false_1, false_2))
            else:
                gt_set = self.data.ground_truth.to_records(index=False).tolist()
                for pair in uv_pairs:
                    index_1, index_2 = (pair[0], pair[1]) if pair[0] < len(self.data.dataset_1) else (pair[1], pair[0])
                    id1 = self.data._gt_to_ids_reversed_1[index_1]
                    id2 = self.data._gt_to_ids_reversed_2[index_2]
                    if (index_1, index_2) in self.candidate_pairs and (id1, id2) in gt_set:
                        examples_records['true'].append(self._extract_records(index_1, index_2))
                        break
                
                for pair in reversed(uv_pairs):
                    index_1, index_2 = (pair[0], pair[1]) if pair[0] < len(self.data.dataset_1) else (pair[1], pair[0])
                    id1 = self.data._gt_to_ids_reversed_1[index_1]
                    id2 = self.data._gt_to_ids_reversed_2[index_2]
                    if (index_1, index_2) in self.candidate_pairs and not (id1, id2) in gt_set:
                        examples_records['false'].append(self._extract_records(index_1, index_2))
                        break
  
        else:
            raise Exception("If suffix in ['ft', 'tf'] must provide a dict with the ids of the examples or set create_examples to True")
        return examples_records
            
    def _configuration(self):
        return {
            "LLM": self.model_name,
            "Prompt" : self.system_prompt,
            
        }
        
        
            
            

