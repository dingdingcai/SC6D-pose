so3_input_dim = 9   # input dimension of rotation matrix
in_rgb_size = 256   # the input image size of network
out_mask_size = 64  # the output mask size of network

so3_emb_dim = 32          # rotation embedding dimension
rgb_emb_dim = 64          # RGB feature map channel dimension

depth_bins_num = 1000      # the number of discretized depth bins


use_cache = True
Rz_rotation_aug = True
uniform_depth_sampling = True

#### hyperparameters for training 
taus = 0.1    # temperature
pxy_w = 10    # weight for projection loss
rot_w = 1.0   # weight for rotation loss
msk_w = 1.0   # weight for mask loss
dep_w = 1.0   # weight for depth classification loss

focal_alpha = 0.5 # hyperparameter for focal loss
focal_gamma = 2.0 # hyperparameter for focal loss

warmup_steps = 0
start_lr = 5e-4      # starting learning rate
ending_lr = 1e-5     # ending learning rate
weight_decay = 1e-4  # weight decay of optimizer


R6d_std = 0.0001     # adding noise to the rotation
knn_sampling_range = 30    # degree
uni_sampling_num = 4000    # 1k:6.1˚, 1.5k, 2k:4.3˚, 4k:3.1˚, sampling number of negative rotations from the entire sphere 
knn_sampling_num = 1000    # sampling number of negative rotations in the range (around the GT rotation) }
knn_Rz_range = 180    # in-plane rotation range

BOP_DATASET_ROOT = '/home/hdd/Dingding/Dataspace/BOP_Dataset'
VOC_DATASET_ROOT = '/home/hdd/Dingding/Dataspace/VOCdevkit/VOC2012'

COLOR_AUG_PROB = 0.8
DZI_PAD_SCALE = 1.5
DZI_SCALE_RATIO = 0.25
DZI_SHIFT_RATIO = 0.25    # center shift
BLACK_PADDING_PROB = 0.25 # the prob for zero-out the background
CHANGE_BG_PROB = 0.0      # the prob for changing the background

SC6D_DATASET_CONFIG = {
        'itodd': {
                'width': 1280, 
                'height': 960, 
                'dep_near': 0.010, 
                'dep_far': 4.570,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(28)}, # to object model_name
                'id2cls': {v+1: v for v in range(28)},                        # from object_id to object class_index
                'type': ['train_pbr'],
                # 'cache_prefix': '/run/nvme/job_12364064/data',
                # 'cache_file': "/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_itodd_81f0f48a92f9b451111c00eafd404bc4.pkl",
                },  # √
        'ycbv': {
                'width': 640,  
                'height': 480, 
                'dep_near': 0.010, 
                'dep_far': 2.150, 
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(21)}, # to object model_name
                'id2cls': {v+1: v for v in range(21)}, # from object_id to class_index
                'type': ['train_pbr', 'train_real', 'train_synt'],
                # 'cache_prefix': "/run/nvme/job_12364064/data",
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_ycbv_3ac9b82f24a8584c688ff971e287d55e.pkl',
                },  # √
        'tless': {
                'width': 720,  
                'height': 540, 
                'dep_near': 0.010, 
                'dep_far': 1.800,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(30)}, # to object model_name
                'id2cls': {v+1: v for v in range(30)}, # from object_id to class_index
                'type': ['train_pbr'],
                # 'type': ['train_pbr', 'train_primesense', 'train_render_reconst'],
                # 'cache_prefix': '/run/nvme/job_12355061/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_tless_d83fbf93bc576becca7c8b5415c35a8b.pkl',
                },  # √
        'lmo': {
                'width': 640,  
                'height': 480, 
                'dep_near': 0.010, 
                'dep_far': 1.200, 
                # 'mod2name': {'obj_01':'Ape', 'obj_05':'Can', 'obj_06':'Cat', 'obj_08':'Driller', 'obj_09':'Duck', 'obj_10':'Eggbox', 'obj_11':'Glue', 'obj_12':'Holepunch'},  # 8(LMO) out of 13(LM) objects
                'id2mod': {1:'obj_01', 5:'obj_05', 6:'obj_06', 8:'obj_08', 9:'obj_09', 10:'obj_10', 11:'obj_11', 12:'obj_12'},  # 8(LMO) out of 13(LM) objects
                'id2cls': {1:0, 5:1, 6:2, 8:3, 9:4, 10:5, 11:6, 12:7}, # from object_id to class_index
                'type': ['train_pbr'],
                # 'cache_prefix': '/run/nvme/job_12372946/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_lmo_114bda8f4ea02ea46850929249e46a74.pkl',
                }, 
        'icbin': {
                'width': 640,  
                'height': 480, 
                'dep_near': 0.010, 
                'dep_far': 1.000,   
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(2)},
                'id2cls': {v+1: v for v in range(2)}, # from object_id to class_index
                'type': ['train_pbr'],
                # 'cache_prefix': '/run/nvme/job_12359017/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_icbin_4b62b3184eb2acee0b44c8653c548012.pkl',
                },
        'tudl': {
                # 'num_classes': 3, 
                'width': 640,  
                'height': 480, 
                'dep_near': 0.010, 
                'dep_far': 1.800,
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(3)}, # to object model_name
                'id2cls': {v+1: v for v in range(3)}, # from object_id to class_index
                'type': ['train_pbr', 'train_real', 'train_render'],
                # 'cache_prefix': '/run/nvme/job_12372946/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_tudl_c6b7e728a8b2176f036dfa829f243e68.pkl',
                },
        'hb': {
                'num_classes': 33, 
                'width': 640,  
                'height': 480, 
                'dep_near': 0.010, 
                'dep_far': 1.2,   
                'id2mod': {v+1: "obj_{:02d}".format(v+1) for v in range(33)},
                'id2cls': {v+1: v for v in range(33)}, # from object_id to class_index
                'type': ['train_pbr'],
                # 'cache_prefix': '/run/nvme/job_12372946/data',
                # 'cache_file': '/scratch/project_2003593/Workspace/SO3_SA6D/lib/.cache/dataset_dicts_hb_bf5f87d5ffe01e1c8b85a37a8355e62d.pkl',
                },
        }