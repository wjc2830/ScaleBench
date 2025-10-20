from easydict import EasyDict as edict


# init
__C_data2 = edict()

cfg_data = __C_data2

__C_data2.TRAIN_SIZE = (512,512)
__C_data2.DATA_PATH = 'domain_2'
__C_data2.TRAIN_LST = 'new_train.txt'
__C_data2.VAL_LST =  'Gaussian_val.txt'
__C_data2.VAL4EVAL = 'Gaussian_val_gt_loc.txt'


__C_data2.TEST_LST =  'Gaussian_whole.txt'
__C_data2.TEST4EVAL = 'Gaussian_whole_gt_loc.txt'

__C_data2.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

__C_data2.LABEL_FACTOR = 1
__C_data2.LOG_PARA = 1.
__C_data2.RESUME_MODEL = ''#model path
__C_data2.TRAIN_BATCH_SIZE = 3 #imgs

__C_data2.VAL_BATCH_SIZE = 1 # must be 1


