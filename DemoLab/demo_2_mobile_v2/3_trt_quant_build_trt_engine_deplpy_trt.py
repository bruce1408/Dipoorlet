import os
import sys
# sys.path.append('/mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/DemoLab')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import demo_utils.quant_config as config_param
import tensorrt as trt
from loguru import logger
from contextlib import redirect_stdout
from demo_utils.calibrator import Calibrator, CalibDataLoader
from printk import print_colored_box


LOGGER = trt.Logger(trt.Logger.VERBOSE)

current_dir = os.path.dirname(os.path.realpath(__file__))
export_dir = f"{config_param.export_work_dir}/trt_mobilev2_trt_intrinsic_kl"

# 配置 loguru 日志
log_file_path = f"{export_dir}/engine_export.log"
logger.add(
    log_file_path,
    rotation="10 MB",  # 文件超过 10MB 自动创建新文件
    retention="10 days",  # 保留最近 10 天的日志文件
    level="INFO",  # 设置日志级别
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)

os.makedirs(export_dir, exist_ok=True)

def buildEngine(
    onnx_file, engine_file, FP16_mode, INT8_mode, data_loader, calibration_table_path
):
    logger.info("Initializing TensorRT builder and network...")
    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))

    if FP16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 mode.")

    if INT8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(data_loader, calibration_table_path)
        logger.info("Enabled INT8 mode with calibration.")

    logger.info(f"Parsing ONNX file: {onnx_file}")
    if not parser.parse_from_file(onnx_file):
        for i in range(parser.num_errors):
            logger.error(f"Parser Error {i}: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse the ONNX file: {onnx_file}")

    logger.info("Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        logger.error("EXPORT ENGINE FAILED!")
        raise RuntimeError("Failed to export TensorRT engine.")

    with open(engine_file, "wb") as f:
        f.write(engine)
    logger.info(f"TensorRT engine exported successfully: {engine_file}")


def main():
    onnx_file = f"{config_param.export_work_dir}/mobilev2_model_new.onnx"
    calibration_cache = f"{config_param.trt_calib_cache_dir}/mobilev2_model_calib.cache"

    FP16_mode = True
    INT8_mode = False

    dataloader = CalibDataLoader(batch_size=1, calib_count=1000)

    if not os.path.exists(onnx_file):
        logger.error(f"LOAD ONNX FILE FAILED: {onnx_file}")
        return

    logger.info(
        f"Load ONNX file from: {onnx_file} \nStart export, Please wait a moment..."
    )

    engine_file = f"{export_dir}/mobilev2_model_trt_{'int8' if INT8_mode else 'fp16'}.engine"

    try:
        buildEngine(
            onnx_file, engine_file, FP16_mode, INT8_mode, dataloader, calibration_cache
        )
        print_colored_box(f"Export ENGINE success, Save as: {engine_file}")
    except Exception as e:
        logger.exception(f"Failed to export engine: {str(e)}")


if __name__ == "__main__":
    # 使用 loguru 捕获全局未捕获的异常
    logger.add(sys.stderr, level="ERROR")  # 终端输出高于 ERROR 的日志
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled exception occurred!")
