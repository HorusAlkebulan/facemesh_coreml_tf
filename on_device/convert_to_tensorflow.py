# This file takes the tflite model of facemesh and converts it to coreml, tf format
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, PReLU, DepthwiseConv2D, Add, Reshape, Concatenate
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from layers import down_sampling, res_block
from utils import restore_variables
print(tf.__version__)

tflite_path = "./tflite_models/face_mesh.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
tf_lite_mapping = {}
for i in interpreter.get_tensor_details():
    if ("ker" in i["name"].lower()) or ("bia" in i["name"].lower()) or ("alph" in i["name"].lower()):
        tf_lite_mapping[i['name']] = interpreter.get_tensor(i["index"])


def create_facenet(input_shape, batch_size=1, output_dim=1404, data_format="channels_last"):
    if data_format == "channels_first":
        shared_axes = [2, 3]
    else:
        shared_axes = [1, 2]
    input_tensor = Input(
        shape=input_shape, batch_size=batch_size, name="input_image")
    pre_conv_out = Conv2D(16, kernel_size=(3, 3), strides=(
        2, 2), padding="SAME", data_format=data_format)(input_tensor)
    act_out = PReLU(shared_axes=shared_axes)(pre_conv_out)
    block1 = res_block(act_out, 16, data_format=data_format)
    down1 = down_sampling(block1, 32, data_format=data_format)
    block2 = res_block(down1, 32, data_format=data_format)
    down2 = down_sampling(block2, 64, data_format=data_format)
    block3 = res_block(down2, 64, data_format=data_format)
    down3 = down_sampling(block3, 128, data_format=data_format)
    block4 = res_block(down3, 128, data_format=data_format)
    down4 = down_sampling(block4, 128, data_format=data_format)
    block5 = res_block(down4, 128, data_format=data_format)

    land1 = down_sampling(block5, 128, data_format=data_format)
    land2 = res_block(land1, 128, data_format=data_format)
    land3 = Conv2D(32, kernel_size=(1, 1), strides=(
        1, 1), data_format=data_format)(land2)
    land3 = PReLU(shared_axes=shared_axes)(land3)
    land4 = DepthwiseConv2D(kernel_size=(3, 3), strides=(
        1, 1), padding="SAME", data_format=data_format)(land3)
    land4 = Conv2D(32, kernel_size=(1, 1), strides=(
        1, 1), data_format=data_format)(land4)
    land_add_out = Add()([land3, land4])
    land_act_out = PReLU(shared_axes=shared_axes)(land_add_out)
    land_final = Conv2D(output_dim, kernel_size=(3, 3), strides=(
        1, 1), data_format=data_format)(land_act_out)
    land_final = Reshape((output_dim,), name="landmarks")(land_final)

    conf1 = down_sampling(block5, 128, data_format=data_format)
    conf2 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1),
                   name="conv2d_28", data_format=data_format)(conf1)
    conf2 = PReLU(shared_axes=shared_axes, name="p_re_lu_26")(conf2)
    conf3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(
        1, 1), padding="SAME", name="depthwise_conv2d_23", data_format=data_format)(conf2)
    conf3 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1),
                   name="conv2d_29", data_format=data_format)(conf3)
    conf_add_out = Add()([conf2, conf3])
    conf_act_out = PReLU(shared_axes=shared_axes,
                         name="p_re_lu_27")(conf_add_out)
    conf_final = Conv2D(1, kernel_size=(3, 3), strides=(
        1, 1), name="conv2d_30", data_format=data_format)(conf_act_out)
    conf_final = Reshape((1,), name="confidence")(conf_final)
    # concatenation 1404D landmarks and 1D confidence into 1405D output
    final_out = Concatenate()([land_final, conf_final])
    return tf.keras.Model(inputs=[input_tensor], outputs=[final_out])


data_format = "channels_last"
facemesh_tf = create_facenet(
    (192, 192, 3), batch_size=None, output_dim=1404, data_format=data_format)
restore_variables(facemesh_tf, tf_lite_mapping, data_format)
facemesh_tf.save("on_device/facemesh_tf.h5")

tf.keras.backend.clear_session()
coreml_tf = tf.keras.models.load_model("on_device/facemesh_tf.h5")
inp_node = coreml_tf.inputs[0].name[:-2].split('/')[-1]
out_node = coreml_tf.outputs[0].name[:-2].split('/')[-1]
print(inp_node, out_node)

inp_image = Image.open("sample.jpg")
inp_image = inp_image.resize((192, 192))
inp_image_np = np.array(inp_image).astype(np.float32)
inp_image_np = np.expand_dims((inp_image_np/127.5) - 1, 0)

print("Checking model sanity across tensorflow, tflite and coreml")
tf_output = coreml_tf.predict(inp_image_np)
interpreter.set_tensor(0, inp_image_np)
interpreter.invoke()

print("Tensorflow output mean: {}, {}".format(
    tf_output[:, :-1].mean(), tf_output[:, -1]))
print("Tflite output mean: {}, {}".format(interpreter.get_tensor(
    213).mean(), interpreter.get_tensor(210).mean()))

detections = tf_output[:, :-1].reshape(468, 3)[:, :2]
plt.imshow(inp_image)
plt.scatter(detections[:, 0], detections[:, 1], s=1.0, marker="+")
plt.savefig("facemesh_out.jpg")
plt.show()
