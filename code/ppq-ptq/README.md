## PPQ

一个可扩展的、高性能的、面向工业应用的神经网络量化工具! 非常优秀，易用。

repo:

[https://github.com/openppl-public/ppq](https://github.com/openppl-public/ppq)

```
git clone https://github.com/openppl-public/ppq.git
cd ppq
pip install -r requirements.txt
python setup.py install
```

1. ptq

```
python export_onnx.py
python export_engine.py
```

2. qat
   关注 [pytorch-quantization](https://github.com/cvdong/Quantization/tree/main/code/pytorch-quantization)!
