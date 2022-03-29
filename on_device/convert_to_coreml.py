# convert_to_coreml.py
import coremltools as ct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from on_device.on_device_utils import get_onedrive_root, get_logger, make_output_dirs
import coremltools as ct
import tensorflow as tf

logger = get_logger()
storage_root = get_onedrive_root()

logger.info(f'Ensuring output directories exist under {storage_root}')
make_output_dirs(storage_root)

def convert_to_coreml32():

    model_name = 'facemesh'

    h5_fullpath = 'on_device/facemesh_tf.h5'
    coreml32_path = 'on_device/facemesh_float32.mlmodel'
    input_image = "sample.jpg"

    logger.info(f'Converting model to CoreML as {coreml32_path}')

    coreml_model = ct.convert(h5_fullpath)
    coreml_model.save(coreml32_path)

    tf.keras.backend.clear_session()
    coreml_tf = tf.keras.models.load_model(h5_fullpath)
    inp_node = coreml_tf.inputs[0].name[:-2].split('/')[-1]
    out_node = coreml_tf.outputs[0].name[:-2].split('/')[-1]
    logger.info(inp_node, out_node)

    inp_image = Image.open(input_image)
    inp_image = inp_image.resize((192, 192))
    inp_image_np = np.array(inp_image).astype(np.float32)
    inp_image_np = np.expand_dims((inp_image_np/127.5) - 1, 0)

    coreml_model = ct.models.MLModel(coreml32_path)

    logger.info("Checking model sanity across tensorflow, tflite and coreml")
    tf_output = coreml_tf.predict(inp_image_np)

    coreml_out_dict = coreml_model.predict({"input_image": inp_image})
    output_names =  list(coreml_out_dict.keys())

    logger.info(f'Renaming output label to {output_names}')

    # coreml_spec = coreml_model.get_spec()
    # ct.utils.rename_feature(coreml_spec, list(coreml_out_dict.keys())[0], output_names)
    # coreml_model_after = ct.models.MLModel(coreml_spec)
    # coreml_model_after.save(coreml32_path)

    logger.info(f'Running test CoreML prediction using image = {inp_image_np.shape}')

    coreml_output = coreml_model.predict(
        {"input_image": inp_image})["points_confidence"]

    interpreter.set_tensor(0, inp_image_np)
    interpreter.invoke()

    logger.info("Tensorflow output mean: {}, {}".format(
        tf_output[:, :-1].mean(), tf_output[:, -1]))
    logger.info("Tflite output mean: {}, {}".format(interpreter.get_tensor(
        213).mean(), interpreter.get_tensor(210).mean()))
    logger.info("CoreMl output mean: {}, {}".format(
        coreml_output[:, :-1].mean(), coreml_output[:, -1]))

    detections = coreml_output[:, :-1].reshape(468, 3)[:, :2]
    plt.imshow(inp_image)
    plt.scatter(detections[:, 0], detections[:, 1], s=1.0, marker="+")
    plt.savefig("facemesh_out.jpg")
    plt.show()

if __name__ == '__main__':
    convert_to_coreml32()
