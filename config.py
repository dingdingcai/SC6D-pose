

SO3_INPUT_DIM = 9       # input dimension of rotation matrix
INPUT_IMG_SIZE = 256    # the input image size of network
OUTPUT_MASK_SIZE = 64   # the output mask size of network
SO3_EMB_DIM = 32        # rotation embedding dimension
RGB_EMB_DIM = 64        # RGB feature map channel dimension
Tz_BINS_NUM = 1000   # the number of discretized depth bins


SO3_TRAINING_SAMPLINGS = 5000

SO3_TESTING_Rz_SAMPLINGS = 120
SO3_TESTING_VP_SAMPLINGS = 4000


USE_CACHE = True
ZOOM_PAD_SCALE = 1.0
ZOOM_SCALE_RATIO = 0.25
ZOOM_SHIFT_RATIO = 0.25   # center shift
BG_CHANGE_PROB = 0.8      # the prob for changing the background

####### TESTING CONFIG ######
BOP_SUBSET_ONLY = True

BOP_DATASET_ROOT = '/home/hdd/Dingding/Dataspace/BOP_Dataset'

LOGGING_STEPS = 100

BOP_DATASET_CONFIG = {
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
                'train_set': ['train_pbr', 'train_real', 'train_synt'],
                'test_set': 'test',
                },  # √
        'tless': {
                'width': 720,  
                'height': 540, 
                'Tz_near': 0.010, 
                'Tz_far': 1.800,
                'num_class': 30,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(30)}, # to object model_name
                'id2cls': {v+1: v for v in range(30)}, # from object_id to class_index
                'train_set': ['train_pbr'],
                'test_set': 'test_primesense',
                },  # √
        'lmo': {
                'width': 640,  
                'height': 480, 
                'Tz_near': 0.010, 
                'Tz_far': 1.200, 
                'num_class': 8,
                # 'mod2name': {'obj_01':'Ape', 'obj_05':'Can', 'obj_06':'Cat', 'obj_08':'Driller', 'obj_09':'Duck', 'obj_10':'Eggbox', 'obj_11':'Glue', 'obj_12':'Holepunch'},  # 8(LMO) out of 13(LM) objects
                'id2mod': {1:'obj_01', 5:'obj_05', 6:'obj_06', 8:'obj_08', 9:'obj_09', 10:'obj_10', 11:'obj_11', 12:'obj_12'},  # 8(LMO) out of 13(LM) objects
                'id2cls': {1:0, 5:1, 6:2, 8:3, 9:4, 10:5, 11:6, 12:7}, # from object_id to class_index
                'train_set': ['train_pbr'],
                'test_set': 'test',
                }, 
        'icbin': {
                'width': 640,  
                'height': 480, 
                'Tz_near': 0.010, 
                'Tz_far': 1.000,   
                'num_class': 2,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(2)},
                'id2cls': {v+1: v for v in range(2)}, # from object_id to class_index
                'train_set': ['train_pbr'],
                'test_set': ['test'],
                # 'cache_prefix': '/run/nvme/job_12359017/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_icbin_4b62b3184eb2acee0b44c8653c548012.pkl',
                },
        'tudl': {
                # 'num_classes': 3, 
                'width': 640,  
                'height': 480, 
                'Tz_near': 0.010, 
                'Tz_far': 1.800,
                'num_class': 3,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(3)}, # to object model_name
                'id2cls': {v+1: v for v in range(3)}, # from object_id to class_index
                'train_set': ['train_pbr', 'train_real', 'train_render'],
                'test_set': 'test',
                # 'cache_prefix': '/run/nvme/job_12372946/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_tudl_c6b7e728a8b2176f036dfa829f243e68.pkl',
                },
        'hb': {
                'num_classes': 33, 
                'width': 640,  
                'height': 480, 
                'Tz_near': 0.010, 
                'Tz_far': 1.2,   
                'num_class': 33,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(33)},
                'id2cls': {v+1: v for v in range(33)}, # from object_id to class_index
                'train_set': ['train_pbr'],
                'test_set': 'test',
                # 'cache_prefix': '/run/nvme/job_12372946/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_hb_bf5f87d5ffe01e1c8b85a37a8355e62d.pkl',
                },
        }
        