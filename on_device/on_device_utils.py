import getpass
import logging
import os
import platform
from datetime import datetime
import colorlog
import numpy as np
from constants import (APP_NAME, IMAGES_DIR, INPUT_DIR, MODEL_DIFF_ALLOWABLE_TOLERANCE_ATOL, MODEL_DIFF_ALLOWABLE_TOLERANCE_RTOL, MODELS_DIR, ONEDRIVE_ROOT, OUTPUT_DIR, OUTPUT_IMAGES_DIR,
                       OUTPUT_INTERMEDIATE, OUTPUT_LOGS_DIR,
                       OUTPUT_MODEL_SET_DIR, OUTPUT_MODELS_DIR,
                       OUTPUT_MOSAICS_DIR, OUTPUT_PROTOBUFS_DIR,
                       OUTPUT_TENSORS_DIR, TEST_IMAGE_RANDOM_PIXEL)


def get_logger():
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        console = logging.StreamHandler()
        console.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'))
        logger.addHandler(console)

        make_output_dirs(ONEDRIVE_ROOT)
        
        filename = f"{APP_NAME}_{datetime.today().strftime('%Y-%m-%d')}.log"
        file_path = os.path.join(ONEDRIVE_ROOT, OUTPUT_MODEL_SET_DIR, OUTPUT_LOGS_DIR, filename)
        logfile = logging.FileHandler(file_path)
        logfile.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        logger.addHandler(logfile)
    return logger

def make_output_dirs(root_dir):
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR), exist_ok=True)
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR,
                OUTPUT_IMAGES_DIR), exist_ok=True)
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR,
                OUTPUT_LOGS_DIR), exist_ok=True)
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR,
                OUTPUT_MODELS_DIR), exist_ok=True)
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR,
                OUTPUT_PROTOBUFS_DIR), exist_ok=True)
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR,
                OUTPUT_TENSORS_DIR), exist_ok=True)
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR,
                OUTPUT_MOSAICS_DIR), exist_ok=True)
    os.makedirs(os.path.join(root_dir, OUTPUT_MODEL_SET_DIR,
                OUTPUT_INTERMEDIATE), exist_ok=True)

logger = get_logger()


def get_onedrive_root():

    username = getpass.getuser()   
    
    if platform.system() == 'Darwin':
        return f'/Users/{username}/git-projects/storage-root/facemesh'
    elif platform.system() == 'Windows':
        return f'C:\\Users\\{username}\\git-projects\\storage-root\\facemesh'
    else:
        logger.error(f'System OS could not be determined')
        raise Exception('System OS could not be determined')

def get_model_export_name(model_basename, framework_version, precision, output_platform, primary_input_h=None, primary_input_w=None):
    inference_size_tag = f'{primary_input_h}h{primary_input_w}w'
        
    if framework_version[0:5] == 'torch':
        return f'{model_basename}_{precision}_{framework_version}'
    else:
        return f'{model_basename}_{inference_size_tag}_{precision}_{framework_version}_{output_platform}'

def get_model_export_full_path(model_export_name, framework_version):
    if framework_version[0:5] == 'opset':
        filename = f'{model_export_name}.onnx'
    elif framework_version[0:5] == 'torch':
        filename = f'{model_export_name}.pt'
    elif framework_version[0:4] == 'spec':
        filename = f'{model_export_name}.mlmodel'
    else:
        raise Exception(f'Invalid framework version: {framework_version}')

    storage_root = get_onedrive_root()
    return os.path.join(
        storage_root, OUTPUT_MODEL_SET_DIR, OUTPUT_DIR, MODELS_DIR, filename)

def get_protobuf_full_path(model_export_name, framework_version):
    if framework_version[0:5] == 'opset':
        filename = f'{model_export_name}.proto'
    elif framework_version[0:5] == 'torch':
        filename = f'{model_export_name}.prototxt'
    elif framework_version[0:4] == 'spec':
        filename = f'{model_export_name}.prototxt'
    else:
        raise Exception(f'Invalid framework version: {framework_version}')

    storage_root = get_onedrive_root()
    return os.path.join(
        storage_root, OUTPUT_MODEL_SET_DIR, OUTPUT_PROTOBUFS_DIR, filename)

def get_image_full_path(image_filename):
    storage_root = get_onedrive_root()
    return os.path.join(storage_root, OUTPUT_MODEL_SET_DIR, INPUT_DIR, IMAGES_DIR, image_filename)

def compare_pytorch_ondevice_results(compare_description, torch_result_t, on_device_result_t):
    try:
        torch_result_t = torch_result_t.detach().numpy()
        on_device_result_t = on_device_result_t.detach().numpy()

        pytorch_ondevice_isclose = np.isclose(
            torch_result_t, on_device_result_t, MODEL_DIFF_ALLOWABLE_TOLERANCE_RTOL, MODEL_DIFF_ALLOWABLE_TOLERANCE_ATOL)
        logger.info(
            f'Comparing inference results {compare_description}')
        false_values = np.size(pytorch_ondevice_isclose) - \
            np.count_nonzero(pytorch_ondevice_isclose)
        total_values = np.size(pytorch_ondevice_isclose)
        error_vs_pytorch = false_values / total_values
        logger.info(
            f'{compare_description} error rate (0.0 -> 1.0) : {error_vs_pytorch}')
        return error_vs_pytorch

    except TypeError as e:
        logger.error(
            'compare_pytorch_ondevice_results error: {e}', exc_info=True)
    except ValueError as e:
        logger.error(
            'compare_pytorch_ondevice_results error: {e}', exc_info=True)
    return None

def get_results_mosaic_name(op_name, image_filename):

    if image_filename is None:
        image_filename = TEST_IMAGE_RANDOM_PIXEL

    image_basename_tag = image_filename.replace(
        '.png', '').replace('.jpg', '').replace(' ', '_')

    mosaic_image_name = f'{op_name}_str_{image_basename_tag}_results'

    return mosaic_image_name



def get_absolute_average_error(torch_result, onnx_result):
    # quantitative comparison
    torch_result_np = torch_result.detach().numpy()
    onnx_result_np = onnx_result.detach().numpy()

    rtol = MODEL_DIFF_ALLOWABLE_TOLERANCE_RTOL  # rtol=1e-05
    atol = MODEL_DIFF_ALLOWABLE_TOLERANCE_ATOL  # atol=1e-08

    
    output_isclose = np.isclose(torch_result_np, onnx_result_np, rtol, atol)
    total_values = np.size(output_isclose)
    false_values = total_values - np.count_nonzero(output_isclose)
    error_vs_pytorch = false_values / total_values
    abs_avg_delta = np.average(np.sqrt((torch_result_np - onnx_result_np)**2))
    root_square_avg_3d = np.average(np.sqrt(torch_result_np ** 2), axis=1)
    root_square_avg_2d = np.average(root_square_avg_3d, axis=1)
    root_square_avg_1d = np.average(root_square_avg_2d)
    percent_error = abs_avg_delta / root_square_avg_1d

    logger.info(f'output: output_isclose false values: {false_values}')
    logger.info(f'output: output_isclose total values: {total_values}')
    logger.info(
        f'output: output_isclose (rtol={rtol}, atol={atol}, error rate (0.0 -> 1.0): {error_vs_pytorch:1.5f}')
    logger.info(
        f'absolute delta / average base values : {abs_avg_delta} / {root_square_avg_1d}')
    logger.info(
        f'absolute average error rate from torch values (0.0 -> 1.0): {percent_error:1.5f}')
    return percent_error
