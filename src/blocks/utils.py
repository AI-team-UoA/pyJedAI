def create_entity_index(blocks: dict, is_dirty_er: bool):
    '''
     Creates a dict of entity ids -> block ids
    '''
    entity_index = {}
    for key, block in blocks.items():
        for entity_id in block.entities_D1:
            entity_index.setdefault(entity_id, [])
            entity_index[entity_id].append(key)

        if not is_dirty_er:
            for entity_id in block.entities_D2:
                entity_index.setdefault(entity_id, [])
                entity_index[entity_id].append(key)

    return entity_index

def drop_single_entity_blocks(blocks: dict, is_dirty_er: bool) -> dict:
    '''
     Removes one-size blocks for DER and empty for CCER
    '''
    all_keys = list(blocks.keys())
    # print("All keys before: ", len(all_keys))
    if is_dirty_er:
        for key in all_keys:
            if len(blocks[key].entities_D1) == 1:
                blocks.pop(key)
    else:
        for key in all_keys:
            if len(blocks[key].entities_D1) == 0 or len(blocks[key].entities_D2) == 0:
                blocks.pop(key)
    # print("All keys after: ", len(blocks.keys()))
    return blocks

def print_blocks(blocks, is_dirty_er):
    print("Number of blocks: ", len(blocks))
    for key, block in blocks.items():
        block.verbose(key, is_dirty_er)

def print_candidate_pairs(blocks):
    print("Number of blocks: ", len(blocks))
    for entity_id, candidates in blocks.items():
        print("\nEntity id ", "\033[1;32m"+str(entity_id)+"\033[0m", " is candidate with: ")
        print("- Number of candidates: " + "[\033[1;34m" + \
            str(len(candidates)) + " entities\033[0m]")
        print(candidates)
