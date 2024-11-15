import os, config
import tensorrt as trt
from printk import print_colored_box, print_colored_box_line
from calibrator import Calibrator, CalibDataLoader

LOGGER = trt.Logger(trt.Logger.VERBOSE)

current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = f"{current_dir}/trt_mobilev2_trt_intrinsic_kl"

os.makedirs(log_dir, exist_ok=True)

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


def main():
    onnx_file = f'{config.train_mobile_v2_dir}/mobilev2_model_new.onnx'
    calibration_cache = f"{config.calibration_dir}/mobilev2_model_calib.cache"        

    FP16_mode = False
    INT8_mode = True

    dataloader = CalibDataLoader(batch_size=1, calib_count=1000)

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)
    print(
        "Load ONNX file from:%s \nStart export, Please wait a moment..." % (onnx_file)
    )
    
    # 导出 tensorrt engine
    engine_file = f"{config.tensorrt_dir}/mobilev2_model_trt_{'int8' if INT8_mode else 'fp16'}.engine"

    buildEngine(
        onnx_file, engine_file, FP16_mode, INT8_mode, dataloader, calibration_cache
    )
    print_colored_box(f"Export ENGINE success, Save as: {engine_file}")


if __name__ == "__main__":
    main()
