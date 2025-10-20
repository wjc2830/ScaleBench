from easydict import EasyDict as edict


# init
__C_data4 = edict()

cfg_data = __C_data4

__C_data4.TRAIN_SIZE = (512,512)
__C_data4.DATA_PATH = 'domain_4'
__C_data4.TRAIN_LST = 'new_train.txt'
__C_data4.VAL_LST =  'Gaussian_val.txt'
__C_data4.VAL4EVAL = 'Gaussian_val_gt_loc.txt'


__C_data4.TEST_LST =  'Gaussian_whole.txt'
__C_data4.TEST4EVAL = 'Gaussian_whole_gt_loc.txt'



__C_data4.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

__C_data4.LABEL_FACTOR = 1
__C_data4.LOG_PARA = 1.
__C_data4.RESUME_MODEL = ''#model path
__C_data4.TRAIN_BATCH_SIZE = 3 #imgs

__C_data4.VAL_BATCH_SIZE = 1 # must be 1


