import engine as eng
from onnx import ModelProto
import tensorrt as trt 

TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
def create_engine(onnx_path, engine_output_path):
    
    batch_size = 1 

    model = ModelProto()
    with open(onnx_path, "rb") as f:
      model.ParseFromString(f.read())
    print('ONNX model laoded...')
    
    print('Creating engine from this onnx file, ', onnx_path)
    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size , d0, d1 ,d2]
    engine = eng.build_engine(onnx_path, shape= shape)
    eng.save_engine(engine, engine_output_path)
    print('TRT engine created and saved at, ', engine_output_path)
