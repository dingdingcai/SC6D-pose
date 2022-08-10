######## GENERAL CONFIG ############
RANDOM_SEED = 2022      # random seed
SO3_INPUT_DIM = 9       # input dimension of rotation matrix
SO3_EMB_DIM = 32        # rotation embedding dimension
RGB_EMB_DIM = 64        # RGB feature map channel dimension
INPUT_IMG_SIZE = 256    # the input image size of network
OUTPUT_MASK_SIZE = 64   # the output mask size of network
Tz_BINS_NUM = 1000      # the number of discretized depth bins

DATASET_ROOT = "Path/to/BOP-Dataset-Root"  # path to the directory root of BOP datasets
VOC_BG_ROOT = "Path/to/VOC2012"            # path to the directory of VOC2022 dataset (used as random realistic background)

####### TESTING CONFIG ######
LOGGING_STEPS = 100
TEST_TIME_AUGMENT = True
SO3_TESTING_Rz_SAMPLINGS = 120
SO3_TESTING_VP_SAMPLINGS = 4000

####### TRAINING CONFIG ######
END_LR = 1e-5
START_LR = 5e-4

LOSS_TZ_W = 1.0
LOSS_ROT_W = 1.0
LOSS_MSK_W = 1.0
LOSS_PXY_W = 10.0
LOSS_NCE_TAUS = 0.1

FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0
DECAY_WEIGHT = 1e-4

USE_CACHE = True
ZOOM_PAD_SCALE = 1.5
ZOOM_SCALE_RATIO = 0.25
ZOOM_SHIFT_RATIO = 0.25   # center shift
COLOR_AUG_PROB = 0.5
CHANGE_BG_PROB = 0.8      # the prob for changing the background
SO3_TRAINING_SAMPLINGS = 5000

RZ_ROTATION_AUG = True
PEMAP_NORMALIZE = True # centered with zeros for PE map

DATASET_CONFIG = {
        'tless': {
                'width': 720,  
                'height': 540, 
                'Tz_near': 0.010, 
                'Tz_far': 1.800,
                'num_class': 30,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(30)}, # to object model_name
                'id2cls': {v+1: v for v in range(30)}, # from object_id to class_index
                'train_set': ['train_pbr'],
                # 'train_set': ['train_pbr', 'train_primesense'],
                'test_set': 'test_primesense',
                },  # √
        'itodd': {
                'width': 1280, 
                'height': 960, 
                'Tz_near': 0.010, 
                'Tz_far': 4.570,
                'num_class': 28,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(28)}, # to object model_name
                'id2cls': {v+1: v for v in range(28)},                        # from object_id to object class_index
                'train_set': ['train_pbr'],
                'test_set': 'test',
                },  # √
        'ycbv': {
                'width': 640,  
                'height': 480, 
                'Tz_near': 0.010, 
                'Tz_far': 2.150, 
                'num_class': 21,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(21)}, # to object model_name
                'id2cls': {v+1: v for v in range(21)}, # from object_id to class_index
                'train_set': ['train_pbr'],
                # 'train_set': ['train_pbr', 'train_real', 'train_synt'],
                'test_set': 'test',
                },  # √
        }
        