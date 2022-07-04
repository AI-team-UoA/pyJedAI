'''
Utility functions
'''


def insert_to_dict(dictionary: dict, key: any, value: any, value_type: any = 'str') -> dict:

    if value_type in ('int' or 'str' or 'float'):
        dictionary.setdefault(key, value)
    elif value_type == 'list':
        dictionary.setdefault(key, [])
        dictionary[key].append(value)
    elif value_type == 'set':
        dictionary.setdefault(key, {})
        dictionary[key].add(value)
    else:
        print("Not supported Value Type")


        
        
def print_clusters(clusters: list) -> None:
    print("Number of clusters: ", len(clusters))
    for (cluster_id, entity_ids) in zip(range(0, len(clusters)), clusters):
        print("\nCluster ", "\033[1;32m"+str(cluster_id)+"\033[0m", " contains: " + "[\033[1;34m" + \
            str(len(entity_ids)) + " entities\033[0m]")
        print(entity_ids)