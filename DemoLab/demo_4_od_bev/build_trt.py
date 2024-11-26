import os
import tensorrt as trt
from utils.calibrator import Calibrator, CalibDataLoader

LOGGER = trt.Logger(trt.Logger.VERBOSE)

info = {
    "inputs_name": ["img_front_fisheye",    "img_right_fisheye", "img_rear_fisheye", "img_left_fisheye", "img_front_short", 
                    "indice_front_fisheye", "indice_right_fisheye", "indice_rear_fisheye", "indice_left_fisheye", "indice_front_short"],
    "outputs_name" : [
        "heatmap", 
        "reg", 
        "dim",
        "rot"
    ],
    "input_width": 960,
    "input_height": 720,
    "confidence_thres": 0.001,
    "iou_thres": 0.7,
    "max_det": 300,
    "providers": ["CUDAExecutionProvider"]
}


def buildEngine(
    onnx_file, engine_file, FP16_mode, INT8_mode, data_loader, calibration_table_path
):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))

    if FP16_mode == True:
        config.set_flag(trt.BuilderFlag.FP16)

    elif INT8_mode == True:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(data_loader, calibration_table_path)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("EXPORT ENGINE FAILED!")

    with open(engine_file, "wb") as f:
        f.write(engine)


def main(mode):
    onnx_file = "//home/bruce_ultra/workspace/onnx_models/modelv5_0915.onnx"
    engine_file = f"/home/bruce_ultra/workspace/quant_workspace/perception_quanti/avp_obstacle/POD/20240827_trt/outputs_trt/od_bev_0915_{mode}.engine"
    calibration_cache = "/home/bruce_ultra/workspace/quant_workspace/perception_quanti/avp_obstacle/POD/20240827_trt/outputs_trt/od_bev_0915_calib.cache"

    if mode=='fp16':
        FP16_mode = True
        INT8_mode = False
    else:
        INT8_mode = True
        FP16_mode = False

    dataloader = CalibDataLoader(batch_size=1, calib_count=256, info=info)

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)

    print("Load ONNX file from:%s \nStart export, Please wait a moment..." % (onnx_file))
    
    buildEngine(onnx_file, engine_file, FP16_mode, INT8_mode, dataloader, calibration_cache)
    
    print("Export ENGINE success, Save as: ", engine_file)


if __name__ == "__main__":
    # main("fp16")
    main("int8")
