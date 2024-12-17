import torch



def name2index(taxonomy, reversed_cifar100) :
    # receive input is an dict
    # recursively replace inplace
    for element in taxonomy.keys() :
        if taxonomy[element] == [] :
            # Empty list
            continue
        if type(taxonomy[element]) == type([]) : # type(list)
            current_list = taxonomy[element]
            for idx in range(len(current_list)) :
                name = current_list[idx]
                current_list[idx] = reversed_cifar100[name]
        elif type(taxonomy[element]) == type(dict()) : # a dict, it means taxonomy[element] is NOT a leaf node
            name2index(taxonomy[element], reversed_cifar100)
        else :
            print(taxonomy[element])


def get_name_value(dictionary) :
    keys = []
    values = []
    for key, value in dictionary.items() :
        keys.append(key)
        values.append(value)
    return keys, values

def get_classname_by_layer(taxonomy) :
    layer1_name, layer1_value = get_name_value(taxonomy)

    layer2_name = []
    layer2_value = []
    for l1_value in layer1_value :
        curr_name, curr_value = get_name_value(l1_value)
        layer2_name = layer2_name + curr_name
        layer2_value = layer2_value + curr_value

    layer3_name = []
    layer3_value = []
    for l2_value in layer2_value :
        curr_name, curr_value = get_name_value(l2_value)
        layer3_name = layer3_name + curr_name
        layer3_value = layer3_value + curr_value

    return layer1_name, layer2_name, layer3_name

def flatten_taxonomy(taxonomy, result=None):
    if result is None:
        result = {}
    for key, value in taxonomy.items():
        if isinstance(value, dict):
            # Recursively flatten the dictionary
            flatten_taxonomy(value, result)
        else:
            # Add key-value pairs for leaf nodes
            result[key] = value
    return result






def get_layers_weight(cifar100, taxonomy) :

    reversed_cifar100 = dict()

    for key, value in cifar100.items() :
        if value in reversed_cifar100 :
            print("ERROR")
        else :
            reversed_cifar100[value] = key

    name2index(taxonomy, reversed_cifar100)





    layer1_name, layer2_name, layer3_name = get_classname_by_layer(taxonomy)    
    flattened_taxonomy = flatten_taxonomy(taxonomy)





    layer3_weight = torch.zeros((24, 100))
    layer2_weight = torch.zeros((6, 24))
    layer1_weight = torch.zeros((3, 6))
    print(layer3_weight.shape)


    for i in range(len(layer3_name)) :
        name = layer3_name[i]
        indices = flattened_taxonomy[name]
        print(name, indices)
        for idx in indices :
            layer3_weight[i][idx] = 1


    num_of_classes = [4, 7, 2, 3, 5, 3]


    for i in range(4) :
        layer2_weight[0][i] = 1
    for i in range(4, 11) :
        layer2_weight[1][i] = 1
    for i in range(11, 13) :
        layer2_weight[2][i] = 1
    for i in range(13, 16) :
        layer2_weight[3][i] = 1
    for i in range(16, 21) :
        layer2_weight[4][i] = 1
    for i in range(21, 24) :
        layer2_weight[5][i] = 1



    layer1_weight[0][0] = 1
    layer1_weight[0][1] = 1
    layer1_weight[1][2] = 1
    layer1_weight[1][3] = 1
    layer1_weight[1][4] = 1
    layer1_weight[2][5] = 1


    return [layer1_weight, layer2_weight, layer3_weight]
