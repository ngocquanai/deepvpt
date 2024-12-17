CIFAR100 = {
    0:  'apple',
    1:  'aquarium_fish',
    2:  'baby',
    3:  'bear',
    4:  'beaver',
    5:  'bed',
    6:  'bee',
    7:  'beetle',
    8:  'bicycle',
    9:  'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'crab',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}

CIFAR100_TAXONOMY = {
    "Natural": {
        "Plants": {
            "Fruits": ["apple", "orange", "pear", "sweet_pepper"],
            "Flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "Trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "Fungi": ["mushroom"]
        },
        "Animals": {

            "Large four-legged mammals" : ["bear", "camel", "cattle", "chimpanzee", "elephant", "kangaroo", 
                            "leopard", "lion", "tiger", "wolf"],
            "Small four-legged mammals" : ["beaver", "fox", "hamster", "mouse", "otter", "porcupine", "possum", 
                            "rabbit", "raccoon", "seal", "skunk", "squirrel", "shrew"],
            "Two-legged mammals": ["baby", "boy", "girl", "man", "woman"],

            "Insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach", "spider"],
            "Aquatic": ["aquarium_fish", "crocodile", "dolphin", "flatfish", "lobster", "ray", "shark", "trout", "whale", "crab"],
            "Reptiles": ["dinosaur", "lizard", "snake", "turtle"],
            "Other_Animals": ["snail", "worm"]


            # "Four-legged mammals": ["bear", "beaver", "camel", "cattle", "chimpanzee", "elephant",
            #                 "fox", "hamster", "kangaroo", "leopard", "lion", "mouse", "otter",
            #                 "porcupine", "possum", "rabbit", "raccoon", "seal", "skunk", "squirrel",
            #                 "tiger", "wolf"],

            
        }
    },
    "Man-Made": {
        "Vehicles": {
            "Wheeled": ["bicycle", "bus", "motorcycle", "pickup_truck", "tractor", "train", "streetcar"],
            "Air": ["rocket"],
            # "Water": []
        },
        "Structures": {
            "Buildings": ["castle", "house", "skyscraper"],
            "Bridges": ["bridge"],
            "Other_Structures": ["road"]
        },
        "Objects": {
            "Furniture": ["bed", "chair", "couch", "table", "wardrobe"],
            "Appliances": ["television"],
            "Containers": ["bottle", "bowl", "can", "cup", "plate"],
            "Instruments": ["clock", "keyboard", "lamp", "telephone"],
            "Tools": ["lawn_mower", "tank"]
        }
    },
    "Environment": {
        "Natural Features": {
            "Sky_Env": ["cloud"],
            "Land_Env": ["forest", "mountain", "plain"],
            "Water_Env": ["sea"]
        }
    }
}


