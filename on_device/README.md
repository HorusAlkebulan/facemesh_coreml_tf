# On Device Framework for facemesh_coreml_tf 

## Virtual Environment Setup (Conda)

### Create Environment

```bash
cd on_device
conda env create --file environment.yml
conda activate mediapipe-sensei-ondevice
```

### Update Environment

```bash
cd on_device
conda env update --file environment.yml --prune
```

### Remove Environment

```bash
conda activate base
conda env remove -n mediapipe-sensei-ondevice
```

## Running Conversion Scripts

### From TFLite to TensorFlow h5 Saved Model

```bash
cd on_device
python convert_to_tensorflow.py
```
