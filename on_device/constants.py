OUTPUT_MODEL_SET_DIR = 'v1-0-0-initial-commit'

APP_NAME = 'facemesh'

BACKEND_COREML_FLOAT16 = 'coreml16'
BACKEND_COREML_FLOAT32 = 'coreml32'
BACKEND_ONNX_FLOAT16 = 'onnx16'
BACKEND_ONNX_FLOAT32 = 'onnx32'

CHECKPOINTS_DIR = 'model_ckpts'

FORCE_CPU_FALSE = False
FORCE_CPU_TRUE = True

FRAMEWORK_COREML_SPEC6 = 'spec6'
FRAMEWORK_ONNX_OPSET_12 = 'opset12'
FRAMEWORK_ONNX_OPSET_13 = 'opset13'
FRAMEWORK_TORCHSCRIPT_V16 = 'torchscript16'

IMAGES_DIR = 'images_fullres'
INPUT_DIR = 'input'

MODEL_DIFF_ALLOWABLE_TOLERANCE_RTOL = 1e-05  # rtol=1e-05
MODEL_DIFF_ALLOWABLE_TOLERANCE_ATOL = 1e-05 # atol=1e-05
MODEL_NAME_CASCADE_Y_NET = 'cascade_y_net'
MODEL_NAME_ENCODER_DECODER = 'encoder_decoder'
MODEL_NAME_MATTING = 'matting'
MODEL_NAME_SVD = 'svd'
MODEL_NAME_SVD_ATSS_HEAD = 'svd_atss_head'
MODEL_NAME_SVD_BACKBONE = 'svd_backbone'
MODEL_NAME_PVT_B4 = 'pvd_b4'
MODEL_NAME_SVD_RPN = 'svd_rpn'
MODEL_NAME_SVD_SYNC_BATCH_NORM4 = 'sync_batch_norm4'
MODEL_NAME_VOLO_FEATS = 'volo_feats'
MODEL_PATH_GENERIC_BASE =  'generic_segformer/iter_80000.pth'
MODEL_PATH_MATT =  'portrait/pre_trained_pytorch/matting_cloud_ft.pth'
MODEL_PATH_PORTRAIT_BASE = 'portrait_segformer_update/PVT_0726.pth'
MODEL_PATH_SVD = 'SVD/Sensei_SVD_v2v3_FTv7_0360000.pth'
MODEL_PATH_TRIMAP = 'portrait/pre_trained_pytorch/trimap_0131.pth'
MODEL_PATH_URM =  'refinement/URM_v2/cascade_mobile_ynet_d1.pth'
MODELS_DIR = 'models'
MOSAIC_SOURCE_IMAGE_DPI = 72
MOSAIC_SUBTITLE_FONT_SIZE = 48
MOSAIC_TITLE_FONT_SIZE = 64
MOSAIC_SUBPLOT_FONT_SIZE = 36
MOSAIC_AXIS_LABEL_FONT_SIZE = 24
MOSAIC_BAR_WIDTH = 0.25

OUTPUT_DIR = 'output'
OUTPUT_MODEL_OPTIONS_DIR = 'model-options'
OUTPUT_IMAGES_DIR = 'output/images/'
OUTPUT_INTERMEDIATE = 'output/intermediate/'
OUTPUT_LOGS_DIR = 'output/logs/'
OUTPUT_MODELS_DIR = 'output/models/'
OUTPUT_PLATFORM_MAC_64 = 'mac64' # mac64 macarm, win64, winarm
OUTPUT_PLATFORM_MAC_ARM = 'macarm'
OUTPUT_PLATFORM_WIN_64 = 'win64'
OUTPUT_PLATFORM_WIN_ARM = 'winarm'
OUTPUT_PROTOBUFS_DIR = 'output/protobufs/'
OUTPUT_TENSORS_DIR = 'output/tensors/'
OUTPUT_MOSAICS_DIR = 'output/mosaics/'

PRECISION_FLOAT32 = 'float32'
PRECISION_FLOAT16 = 'float16'
PRECISION_QUANT8 = 'quant8'

TEST_IMAGE_RANDOM_PIXEL = 'random_pixel_image.png'
RANDOM_SEED = 1234567890

VERBOSE_FALSE = False
VERBOSE_TRUE = True

