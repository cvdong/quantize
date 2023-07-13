## 量化

### tensorrt-ptq

* fp16

build:

```
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
```

test:

```
trtexec --loadEngine=best.engine 
```

这里也可以使用py/cpp脚本实现：

```
python trt-ptq-engine.py --has-half=True
```
