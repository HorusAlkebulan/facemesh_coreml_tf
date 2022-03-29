# On Device Framework for facemesh_coreml_tf 

## Virtual Environment Setup (Conda)

### Create Environment

```bash
conda env create --file on_device/environment.yml
conda activate mediapipe-sensei-ondevice
```

### Update Environment

```bash
conda env update --file on_device/environment.yml --prune
```

### Remove Environment

```bash
conda activate base
conda env remove -n mediapipe-sensei-ondevice
```

## Running Conversion Scripts

### From TFLite to TensorFlow h5 Saved Model

```bash
PYTHONPATH=%PYTHONPATH%:. python on_device/convert_to_tensorflow.py
```

### From TensorFlow h5 Saved Model to CoreML

```bash
PYTHONPATH=%PYTHONPATH%:. python on_device/convert_to_coreml.py
```