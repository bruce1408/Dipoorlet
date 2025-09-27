import os
import sys
import tensorrt as trt
from loguru import logger
from contextlib import redirect_stdout
from spectrautils import print_utils
from common.configs import get_cfg_defaults
from dipoorlet_utils.calibrator import Calibrator, CalibDataLoader

cfg = get_cfg_defaults()
LOGGER = trt.Logger(trt.Logger.VERBOSE)

export_dir = f"{cfg.DIPOORLET.tensorrt_export_dir}/trt_resnet18"
os.makedirs(export_dir, exist_ok=True)

log_file_path = f"{export_dir}/engine_export.log"

logger.add(
    log_file_path,
    rotation="10 MB",  # 文件超过 10MB 自动创建新文件
    retention="10 days",  # 保留最近 10 天的日志文件
    level="INFO",  # 设置日志级别
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)


def buildEngine(
    onnx_file, engine_file, mode, data_loader, calibration_table_path
):
    logger.info("Initializing TensorRT builder and network...")
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))

    if mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 mode.")

    if mode == "int8":
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


def main(mode="int8"):
    onnx_file = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"
    calibration_cache = f"{export_dir}/resnet18_calib.cache"

    dataloader = CalibDataLoader(batch_size=1, calib_count=1000)
    
    if not os.path.exists(onnx_file):
        logger.error(f"LOAD ONNX FILE FAILED: {onnx_file}")
        return

    logger.info(f"Load ONNX file from: {onnx_file} \nStart export, Please wait a moment...")

    engine_file = f"{export_dir}/resnet18_trt_{mode}.engine"

    try:
        buildEngine(onnx_file, engine_file, mode, dataloader, calibration_cache)
        print_utils.print_colored_text(f"Export ENGINE success, Save as: {engine_file}", "green")
    except Exception as e:
        logger.exception(f"Failed to export engine: {str(e)}")


if __name__ == "__main__":
    main("int8")
   
