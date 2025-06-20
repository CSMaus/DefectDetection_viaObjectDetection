import onnx, numpy as np
from onnx import numpy_helper

model = onnx.load("models/000-ImprovedMSC.onnx")

for init in model.graph.initializer:
    if (init.data_type == onnx.TensorProto.INT64 and
        list(init.dims) == [3]):               # shape tensors only
        arr = numpy_helper.to_array(init)
        # pick any pattern that matches the hard-coded reshape
        if (arr[1], arr[2]) == (8, 8):
            arr[0] = -1                         # dynamic dim
            init.CopyFrom(numpy_helper.from_array(arr.astype(np.int64),
                                                  name=init.name))

onnx.save(model, "models/ImprovedMSC_dynamic.onnx")
