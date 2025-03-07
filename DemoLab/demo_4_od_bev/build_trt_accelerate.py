import os, sys
import tensorrt as trt
import concurrent.futures
from loguru import logger
from utils.calibrator import Calibrator, CalibDataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dipoorlet_utils.quant_config as config
from printk import print_colored_box

os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_ids

LOGGER = trt.Logger(trt.Logger.VERBOSE)

# 配置 loguru 日志
log_file_path = f"{config.od_bev_outputs}/engine_export_0306.log"
logger.add(
    log_file_path,
    rotation="10 MB",  # 文件超过 10MB 自动创建新文件
    retention="10 days",  # 保留最近 10 天的日志文件
    level="INFO",  # 设置日志级别
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)

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
    logger.info("Initializing TensorRT builder and network...")
    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    parser = trt.OnnxParser(network, LOGGER)
    builder_config = builder.create_builder_config()
    
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    
    logger.info(f"Parsing ONNX file: {onnx_file}")
    
    # 读取ONNX文件到内存
    with open(onnx_file, 'rb') as f:
        onnx_data = f.read()
    success = parser.parse(onnx_data)
    if not success:
        err_count = parser.num_errors
        for i in range(err_count):
            logger.error(f"ONNX解析错误 {i}: {parser.get_error(i)}")
        return False
    
    # 增加工作空间大小
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 4GB
    
    # 设置优化档案
    builder_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    
    if FP16_mode:
        builder_config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 mode.")
        
    elif INT8_mode:
        builder_config.set_flag(trt.BuilderFlag.INT8)
        builder_config.int8_calibrator = Calibrator(data_loader, calibration_table_path)
        logger.info("Enable INT8 mode with calibration.")
    
    # 添加优化配置
    builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    
    # 使用预设优化
    builder_config.builder_optimization_level = 4
    
    logger.info("Building TensorRT engine...")
    engine = builder.build_serialized_network(network, builder_config)
    if engine is None:
        logger.error("EXPORT ENGINE FAILED!")
        print("EXPORT ENGINE FAILED!")
        return False

    with open(engine_file, "wb") as f:
        f.write(engine)
    logger.info(f"TensorRT engine exported successfully: {engine_file}")
    return True
        
def main(modes=None):
    if modes is None:
        modes = ["fp16"]
    
    onnx_file = f"{config.od_bev_onnx_models}/od_bev_0306.onnx"
    
    # 验证ONNX文件存在
    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)
        return
    
    # 准备dataloader - 只初始化一次
    dataloader = CalibDataLoader(batch_size=6, calib_count=2, info=info)
    logger.info(f"Load ONNX file from: {onnx_file}")
    
    # 确保输出目录存在
    output_dir = f"{config.od_bev_outputs}/trt_od_bev_trt_intrinsic_kl"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    # 使用多线程同时构建不同模式的引擎
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(modes)) as executor:
        futures = []
        for mode in modes:
            engine_file = f"{output_dir}/od_bev_25_0306_v2_{mode}.trt"
            calibration_cache = f"{output_dir}/od_bev_25_0306_calib.cache"
            
            FP16_mode = (mode == 'fp16')
            INT8_mode = (mode == 'int8')
            
            futures.append(
                executor.submit(
                    buildEngine, 
                    onnx_file, 
                    engine_file, 
                    FP16_mode, 
                    INT8_mode, 
                    dataloader, 
                    calibration_cache
                )
            )
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            mode = modes[i] 
            results[mode] = future.result()
            
    # 打印结果
    for mode, success in results.items():
        if success:
            engine_file = f"{output_dir}/od_bev_25_0306_v2_{mode}.trt"
            print_colored_box(f"Export ENGINE success, Save as: {engine_file}")
        else:
            print_colored_box(f"Failed to export {mode} engine", "red")

if __name__ == "__main__":
    # 同时构建FP16和INT8模式
    # main(["fp16", "int8"])
    main(["fp16"])
