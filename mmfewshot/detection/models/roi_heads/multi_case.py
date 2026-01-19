COCO_SPLIT = dict(
    ALL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    NOVEL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
                   'cow', 'bottle', 'chair', 'couch', 'potted plant',
                   'dining table', 'tv'),
    BASE_CLASSES=('truck', 'traffic light', 'fire hydrant', 'stop sign',
                  'parking meter', 'bench', 'elephant', 'bear', 'zebra',
                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush'))
coco_reason_cls = dict(
    person=[100, 201, 302],
    bicycle=[101, 200, 301],
    car=[101, 200, 301],
    motorcycle=[101, 200, 301],
    airplane=[101, 200, 301],
    bus=[101, 200, 301],
    train=[101, 200, 301],
    truck=[101, 200, 301],
    boat=[101, 200, 301],
    traffic_light=[102, 200, 301],
    fire_hydrant=[102, 200, 301],
    stop_sign=[102, 200, 301],
    parking_meter=[102, 200, 301],
    bench=[102, 200, 301],
    bird=[103, 201, 301],
    cat=[103, 201, 302],
    dog=[103, 201, 302],
    horse=[103, 201, 301],
    sheep=[103, 201, 301],
    cow=[103, 201, 301],
    elephant=[103, 201, 301],
    bear=[103, 201, 301],
    zebra=[103, 201, 301],
    giraffe=[103, 201, 301],
    backpack=[104, 200, 301],
    umbrella=[104, 200, 301],
    handbag=[104, 200, 301],
    tie=[104, 200, 301],
    suitcase=[104, 200, 301],
    frisbee=[105, 200, 301],
    skis=[105, 200, 301],
    snowboard=[105, 200, 301],
    sports_ball=[105, 200, 301],
    kite=[105, 200, 301],
    baseball_bat=[105, 200, 301],
    baseball_glove=[105, 200, 301],
    skateboard=[105, 200, 301],
    surfboard=[105, 200, 301],
    tennis_racket=[105, 200, 301],
    bottle=[106, 200, 302],
    wine_glass=[106, 200, 300],
    cup=[106, 200, 300],
    fork=[106, 200, 300],
    knife=[106, 200, 300],
    spoon=[106, 200, 300],
    bowl=[106, 200, 300],
    banana=[107, 201, 302],
    apple=[107, 201, 302],
    orange=[107, 201, 302],
    broccoli=[107, 201, 300],
    carrot=[107, 201, 300],
    hot_dog=[108, 200, 302],
    pizza=[108, 200, 302],
    donut=[108, 200, 302],
    cake=[108, 200, 302],
    sandwich=[108, 200, 302],
    chair=[109, 200, 300],
    couch=[109, 200, 300],
    potted_plant=[109, 200, 300],
    bed=[109, 200, 300],
    dining_table=[109, 200, 300],
    toilet=[109, 200, 300],
    tv=[110, 200, 300],
    laptop=[110, 200, 300],
    mouse=[110, 200, 300],
    remote=[110, 200, 300],
    keyboard=[110, 200, 300],
    cell_phone=[110, 200, 300],
    microwave=[111, 200, 300],
    oven=[111, 200, 300],
    toaster=[111, 200, 300],
    sink=[111, 200, 300],
    refrigerator=[111, 200, 300],
    book=[112, 200, 300],
    clock=[112, 200, 300],
    vase=[112, 200, 300],
    scissors=[112, 200, 300],
    teddy_bear=[112, 200, 300],
    hair_drier=[112, 200, 300],
    toothbrush=[112, 200, 300])

VOC_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat',
                        'chair', 'diningtable', 'dog', 'horse', 'person',
                        'pottedplant', 'sheep', 'train', 'tvmonitor', 'bird',
                        'bus', 'cow', 'motorbike', 'sofa'),
    ALL_CLASSES_SPLIT2=('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
                        'chair', 'diningtable', 'dog', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'train', 'tvmonitor',
                        'aeroplane', 'bottle', 'cow', 'horse', 'sofa'),
    ALL_CLASSES_SPLIT3=('aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car',
                        'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'person', 'pottedplant', 'train', 'tvmonitor', 'boat',
                        'cat', 'motorbike', 'sheep', 'sofa'),
    NOVEL_CLASSES_SPLIT1=('bird', 'bus', 'cow', 'motorbike', 'sofa'),
    NOVEL_CLASSES_SPLIT2=('aeroplane', 'bottle', 'cow', 'horse', 'sofa'),
    NOVEL_CLASSES_SPLIT3=('boat', 'cat', 'motorbike', 'sheep', 'sofa'),
    BASE_CLASSES_SPLIT1=('aeroplane', 'bicycle', 'boat', 'bottle', 'car',
                         'cat', 'chair', 'diningtable', 'dog', 'horse',
                         'person', 'pottedplant', 'sheep', 'train',
                         'tvmonitor'),
    BASE_CLASSES_SPLIT2=('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
                         'chair', 'diningtable', 'dog', 'motorbike', 'person',
                         'pottedplant', 'sheep', 'train', 'tvmonitor'),
    BASE_CLASSES_SPLIT3=('aeroplane', 'bicycle', 'bird', 'bottle', 'bus',
                         'car', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                         'person', 'pottedplant', 'train', 'tvmonitor'))

# voc_reason_cls = dict(
#     aeroplane=[101, 200, 301],  # 0
#     bicycle=[101, 200, 301],  # 1
#     boat=[101, 200, 301],  # 2
#     bottle=[106, 200, 302],  # 3
#     car=[101, 200, 301],  # 4
#     cat=[103, 201, 302],  # 5
#     chair=[109, 200, 300],  # 6
#     diningtable=[109, 200, 300],  # 7
#     dog=[103, 201, 302],  # 8
#     horse=[103, 201, 301],  # 9
#     person=[100, 201, 302],  # 10
#     pottedplant=[109, 200, 300],  # 11
#     sheep=[103, 201, 301],  # 12
#     train=[101, 200, 301],  # 13
#     tvmonitor=[110, 200, 300],  # 14
#     bird=[103, 201, 301],  # 15
#     bus=[101, 200, 301],  # 16
#     cow=[103, 201, 301],  # 17
#     motorbike=[101, 200, 301],  # 18
#     sofa=[109, 200, 300]  # 19
# )

voc_reason_cls = dict(
    person=[100, 201, 302],  # 10
    aeroplane=[101, 200, 301],  # 0
    bicycle=[101, 200, 301],  # 1
    boat=[101, 200, 301],  # 2
    bus=[101, 200, 301],  # 16
    car=[101, 200, 301],  # 4
    train=[101, 200, 301],  # 13
    motorbike=[101, 200, 301],  # 18
    cat=[102, 201, 302],  # 5
    dog=[102, 201, 302],  # 8
    horse=[102, 201, 301],  # 9
    sheep=[102, 201, 301],  # 12
    bird=[102, 201, 301],  # 15
    cow=[102, 201, 301],  # 17
    bottle=[103, 200, 302],  # 3
    chair=[104, 200, 300],  # 6
    diningtable=[104, 200, 300],  # 7
    pottedplant=[104, 200, 300],  # 11
    sofa=[104, 200, 300],  # 19
    tvmonitor=[105, 200, 300]  # 14
)
