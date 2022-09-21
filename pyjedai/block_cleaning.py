from time import time

import numpy as np
from tqdm.notebook import tqdm

from .datamodel import Block, Data
from .utils import create_entity_index, drop_single_entity_blocks


class BlockFiltering:
    """Retains every entity in a subset of its smallest blocks.
    Filtering consists of 3 steps:
     - Blocks sort in ascending cardinality
     - Creation of Entity Index: inversed block dictionary
     - Retain every entity in ratio % of its smallest blocks
     - Blocks reconstruction
    """

    _method_name = "Block Filtering"
    _method_info = "Retains every entity in a subset of its smallest blocks."

    def __init__(self, ratio: float = 0.8) -> None:
        if ratio > 1.0 or ratio < 0.0:
            raise AttributeError("Ratio is a number between 0.0 and 1.0")
        else:
            self.ratio = ratio
        self.blocks: dict
        self.tqdm_disable: bool
        self.data: Data
        self._progress_bar: tqdm
        self.execution_time: float

    def __str__(self) -> str:
        print(self._method_name + self._method_info)
        print("Ratio: ", self.ratio)
        return super().__str__()

    def process(
            self,
            blocks: dict = None,
            data: Data = None,
            tqdm_disable: bool = False
    ) -> dict:
        """Main method of Block Filtering

        Args:
            blocks (dict, optional): dict of keys to Blocks. Defaults to None.
            data (Data, optional): input dataset module. Defaults to None.
            tqdm_disable (bool, optional): disable progress bars. Defaults to False.

        Returns:
            dict: dict of keys to Blocks
        """
        start_time, self.tqdm_disable, self.data = time(), tqdm_disable, data
        self._progress_bar = tqdm(
            total=3,
            desc=self._method_name,
            dynamic_ncols=True,
            disable=self.tqdm_disable
        )
        sorted_blocks = sort_blocks_cardinality(blocks, self.data.is_dirty_er)
        self._progress_bar.update(1)
        entity_index = create_entity_index(sorted_blocks, self.data.is_dirty_er)
        self._progress_bar.update(1)
        filtered_blocks = {}
        for entity_id, block_keys in entity_index.items():
            # Create new blocks from the entity index
            for key in block_keys[:int(round(self.ratio*len(block_keys)))]:
                filtered_blocks.setdefault(key, Block())
                # Entities ids start to 0 ... n-1 for 1st dataset
                # and n ... m for 2nd dataset
                if entity_id < self.data.dataset_limit:
                    filtered_blocks[key].entities_D1.add(entity_id)
                else:
                    filtered_blocks[key].entities_D2.add(entity_id)
        self._progress_bar.update(1)
        self.blocks = drop_single_entity_blocks(filtered_blocks, self.data.is_dirty_er)
        self._progress_bar.close()
        self.execution_time = time() - start_time

        return self.blocks

    def method_configuration(self) -> dict:
        """Returns configuration details
        """
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def _configuration(self) -> dict:
        return {
            "Ratio" : self.ratio
        }

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )

class BlockPurging:
    """Discards the blocks exceeding a certain number of comparisons.
    """

    _method_name = "Block Purging"
    _method_info = "Discards the blocks exceeding a certain number of comparisons."

    def __init__(self, smoothing_factor: float = 1.025) -> any:
        self.smoothing_factor: float = smoothing_factor
        self.max_comparisons_per_block: float
        self.execution_time: float
        self.tqdm_disable: bool
        self._progress_bar: tqdm
        self.data: Data

    def process(
            self,
            blocks: dict,
            data: Data,
            tqdm_disable: bool = False
    ) -> dict:
        """Main method of Block Purging

        Args:
            blocks (dict): _description_
            data (Data): _description_
            tqdm_disable (bool, optional): _description_. Defaults to False.

        Returns:
            dict: _description_
        """
        self.tqdm_disable, self.data, start_time = tqdm_disable, data, time()
        self._progress_bar = tqdm(total=2*len(blocks), desc=self._method_name, disable=self.tqdm_disable)
        if not blocks:
            raise AttributeError("Empty dict of blocks was given as input!")
        new_blocks = blocks.copy()
        self._set_threshold(new_blocks)
        all_keys = list(new_blocks.keys())
        for key in all_keys:
            if new_blocks[key].get_cardinality(self.data.is_dirty_er) > self.max_comparisons_per_block:
                del new_blocks[key]
            self._progress_bar.update(1)
        self.execution_time = time() - start_time
        self._progress_bar.close()

        return new_blocks

    def _set_threshold(self, blocks: dict) -> None:
        """
        TODO _summary_

        Args:
            blocks (dict): _description_
        """
        sorted_blocks = sort_blocks_cardinality(blocks, self.data.is_dirty_er)
        distinct_comparisons_level = set(b.get_cardinality(self.data.is_dirty_er) \
                                        for _, b in sorted_blocks.items())
        block_assignments = np.empty([len(distinct_comparisons_level)])
        comparisons_level = np.empty([len(distinct_comparisons_level)])
        total_comparisons_per_level = np.empty([len(distinct_comparisons_level)])
        index = -1
        for _, block in sorted_blocks.items():
            if index == -1:
                index += 1
                comparisons_level[index] = block.get_cardinality(self.data.is_dirty_er)
                block_assignments[index] = 0
                total_comparisons_per_level[index] = 0
            elif block.get_cardinality(self.data.is_dirty_er) != comparisons_level[index]:
                index += 1
                comparisons_level[index] = block.get_cardinality(self.data.is_dirty_er)
                block_assignments[index] = block_assignments[index-1]
                total_comparisons_per_level[index] = total_comparisons_per_level[index-1]

            block_assignments[index] += block.get_size()
            total_comparisons_per_level[index] += block.get_cardinality(self.data.is_dirty_er)
            self._progress_bar.update(1)

        current_bc = current_cc = current_size = \
            previous_bc = previous_cc = previous_size = 0
        for i in range(len(block_assignments)-1, 0, -1):
            previous_size = current_size
            previous_bc = current_bc
            previous_cc = current_cc
            current_size = comparisons_level[i]
            current_bc = block_assignments[i]
            current_cc = total_comparisons_per_level[i]
            if current_bc * previous_cc < self.smoothing_factor * current_cc * previous_bc:
                break
        self.max_comparisons_per_block = previous_size

    def _satisfies_threshold(self, block: Block) -> bool:
        return block.get_cardinality(self.data.is_dirty_er) <= self.max_comparisons_per_block

    def method_configuration(self) -> dict:
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def _configuration(self) -> dict:
        return {
            "Smoothing factor" : self.smoothing_factor,
            "Max Comparisons per Block" : self.max_comparisons_per_block
        }

    def report(self) -> None:
        """Prints method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nParameters: \n" + ''.join(
                ['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) +
            "Runtime: {:2.4f} seconds".format(self.execution_time)
        )

def sort_blocks_cardinality(blocks: dict, is_dirty_er: bool) -> dict:
    return dict(sorted(blocks.items(), key=lambda x: x[1].get_cardinality(is_dirty_er)))
