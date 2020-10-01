# tf2trt_wtih_onnx
This repo documnet how to convert Tensorflow / Keras model to TRT engine using ONNX.  
Note: If [tf2onnx](https://github.com/onnx/tensorflow-onnx) didn't work on Keras model try to use [keras2onnx](https://github.com/onnx/keras-onnx)

## TODOs
- Try [keras2onnx](https://github.com/onnx/keras-onnx) on Facenet model and compare results
- Try [tf2onnx](https://github.com/onnx/tensorflow-onnx) with `--fold_const` option. which may benefit the optimization as stated in [repo's README.md](https://github.com/onnx/tensorflow-onnx#--fold_const) :
> When set, TensorFlow fold_constants transformation is applied before conversion. This benefits features including Transpose optimization (e.g. Transpose operations introduced during tf-graph-to-onnx-graph conversion will be removed), and RNN unit conversion (for example LSTM). Older TensorFlow version might run into issues with this option depending on the model.
- Try to freeze the graph as using [freeze_graph.py](https://github.com/davidsandberg/facenet/blob/master/src/freeze_graph.py) in Facenet repo.
