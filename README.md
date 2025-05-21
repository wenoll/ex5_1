# 创建 Python 3.9 虚拟环境


```python
# 安装 python3.9 及必要工具
!sudo apt-get update -y
!sudo apt-get install python3.9 python3.9-venv python3.9-distutils curl -y
# 创建虚拟环境（不会自带 pip）
!python3.9 -m venv /content/tflite_env
# 下载官方 get-pip 脚本
!curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# 使用虚拟环境中的 python 执行脚本安装 pip
!/content/tflite_env/bin/python get-pip.py
#验证 pip 是否生效
!/content/tflite_env/bin/pip --version


```


# 安装核心依赖


```python
! /content/tflite_env/bin/pip install -q \
  tensorflow==2.10.0 \
  keras==2.10.0 \
  numpy==1.23.5 \
  protobuf==3.19.6 \
  tensorflow-hub==0.12.0 \
  tflite-support==0.4.2 \
  tensorflow-datasets==4.8.3 \
  sentencepiece==0.1.99 \
  sounddevice==0.4.5 \
  librosa==0.8.1 \
  flatbuffers==23.5.26 \
  matplotlib==3.5.3 \
  opencv-python==4.8.0.76


```

# 安装 tflite-model-maker 本体


```python
! /content/tflite_env/bin/pip install tflite-model-maker==0.4.2

```


# 补充缺失依赖


```python
! /content/tflite_env/bin/pip install matplotlib_inline IPython


```


# 验证是否成功安装


```python
! /content/tflite_env/bin/python -c "from tflite_model_maker import image_classifier; print('TFLite Model Maker 已成功导入')"

```

    2025-05-21 01:01:48.212516: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64
    2025-05-21 01:01:48.212570: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    /content/tflite_env/lib/python3.9/site-packages/tensorflow_addons/utils/tfa_eol_msg.py:23: UserWarning: 
    
    TensorFlow Addons (TFA) has ended development and introduction of new features.
    TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
    Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 
    
    For more information see: https://github.com/tensorflow/addons/issues/2807 
    
      warnings.warn(
    /content/tflite_env/lib/python3.9/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.13.0 and strictly below 2.16.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.8.4 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      warnings.warn(
    TFLite Model Maker 已成功导入


# 训练模型（花卉分类）


```python
# step_train.py
with open('/content/step_train.py', 'w') as f:
    f.write("""
import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

image_path = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

model = image_classifier.create(train_data)
loss, acc = model.evaluate(test_data)
print(f'✅ 测试准确率: {acc:.4f}')
model.export(export_dir='.')
""")
! /content/tflite_env/bin/python /content/step_train.py


```




# tflite模型下载


```python
from google.colab import files
files.download('model.tflite')

```

# 使用实验三的应用验证生成的模型

