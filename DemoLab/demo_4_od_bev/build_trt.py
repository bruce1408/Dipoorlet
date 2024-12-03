import os, sys
import tensorrt as trt
from utils.calibrator import Calibrator, CalibDataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import demo_utils.quant_config as config

LOGGER = trt.Logger(trt.Logger.VERBOSE)

info = {
        "inputs_name": [
            "front_short_camera", 
            "front_fisheye_camera",
            "right_fisheye_camera",
            "rear_fisheye_camera", 
            "left_fisheye_camera", 
            "indices"
        ],
        "outputs_name" : [
            "dim", 
            "height", 
            "reg",
            "rot",
            "hm"
        ],
        "input_width": [1920, 960, 960, 960, 960],
        "input_height": [720, 720, 720, 720, 720],
        "indices": [5, 256, 192, 4, 2],
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
    onnx_file = f"{config.od_bev_onnx_models}/od_bev_1125_v2.onnx"
    engine_file = f"{config.od_bev_outputs}/trt_od_bev_trt_intrinsic_kl/od_bev_1125_v2_{mode}.engine"
    calibration_cache = f"{config.od_bev_outputs}/trt_od_bev_trt_intrinsic_kl/od_bev_1125_v2_calib.cache"

    if mode=='fp16':
        FP16_mode = True
        INT8_mode = False
    else:
        INT8_mode = True
        FP16_mode = False

    dataloader = CalibDataLoader(batch_size=6, calib_count=10, info=info)

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)

    print("Load ONNX file from:%s \nStart export, Please wait a moment..." % (onnx_file))
    
    buildEngine(onnx_file, engine_file, FP16_mode, INT8_mode, dataloader, calibration_cache)
    
    print("Export ENGINE success, Save as: ", engine_file)


if __name__ == "__main__":
    # main("fp16")
    main("int8")
