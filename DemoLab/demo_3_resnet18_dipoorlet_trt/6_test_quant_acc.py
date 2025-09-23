'''
version: 1.0.0
Author: BruceCui
Date: 2024-11-13 16:57:30
LastEditors: BruceCui
LastEditTime: 2024-12-03 19:40:59
'''
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time, os, sys
import torch
from PIL import Image
from dipoorlet_utils.dataset import get_dataloaders
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
current_file_path = os.path.dirname(os.path.abspath(__file__))

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory"""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_tensor_shape(binding)
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        
        size = trt.volume(shape)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        
        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):

    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    
    # context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


def deserializing_engine(engine_file):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file, "rb") as f:
        serialized_engine = f.read()
    return runtime.deserialize_cuda_engine(serialized_engine)


def main(quant_mode, imagenet_mode="normal"):
    
    if imagenet_mode == "normal":
        datasets_dir = cfg.SYSTEM.imagenet_dir
    else:
        datasets_dir = cfg.SYSTEM.imagenet_200_dir
        
    val_batch_size = 1

    _, val_dataset, _ = get_dataloaders(
        datasets_dir=datasets_dir,
        imagenet_mode=imagenet_mode,
        batch_size=val_batch_size
    )
        
    # engine_file = f"{current_file_path}/trt/mobilev2_model_dipoorlet_brecq_{mode}.engine"
    # engine_file = f"{config.export_work_dir}/mobilev2_model_trt_{mode}.engine"
    # engine_file = f"{config.export_work_dir}/trt_mobilev2_trt_intrinsic_kl/mobilev2_model_trt_{mode}.engine"
    
    # engine_file = f"{current_file_path}/trt_mobile_v2_dipoorlet_mse/mobilev2_model_dipoorlet_{mode}.engine"
    # engine_file = f"{current_file_path}/trt_mobile_v2_dipoorlet_brecq/mobilev2_model_dipoorlet_mse_brecq_{mode}.engine"
    # engine_file = f"{current_file_path}/trt_mobile_v2_dipoorlet_mse_brecq/mobilev2_model_dipoorlet_mse_brecq_{mode}.engine"
    # engine_file = f"{current_file_path}/trt_mobile_v2_dipoorlet_hist/mobilev2_model_dipoorlet_hist_{mode}.engine"
    engine_file = f"{cfg.DIPOORLET.tensorrt_export_dir}/trt_resnet18/resnet18_trt_{quant_mode}.engine"
    engine = deserializing_engine(engine_file)
    input_name = engine.get_binding_name(0)

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Do inference
    shape_of_output = (val_batch_size, 1000)  # 这里是1000是因为imagenet数据集有1000个类别
    
    # Load data to the buffer
    running_corrects = 0.0
    for i, (inps, labels) in enumerate(val_dataset):

        current_batch_size = inps.shape[0]
        if current_batch_size != val_batch_size:
            # 对于不满的 batch，可以跳过或者单独处理，这里简单跳过
            print(f"Skipping last batch of size {current_batch_size}.")
            continue
        
        # --- 修改点 2: 在推理前设置当前 batch 的实际形状 ---
        # 这对于动态形状的 engine 至关重要
        # context.set_binding_shape(0, inps.shape)
        context.set_input_shape(input_name, tuple(inps.shape))


        # 准备输入数据
        inputs[0].host = np.ascontiguousarray(inps.cpu().numpy())
        
        t1 = time.time()
        # --- 修改点 3: 调用新的 do_inference，不再需要 batch_size 参数 ---
        trt_outputs = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        t2 = time.time()
        
        # --- 修改点 4: 后处理时也使用当前的 batch size ---
        # 从输出buffer中只取出有效部分
        h_outputs = trt_outputs[0]
        feat = postprocess_the_outputs(h_outputs, (current_batch_size, 1000))

        feat = torch.tensor(feat)
        _, preds = torch.max(feat, 1)

        running_corrects += torch.sum(preds == labels.data)
        
        total_samples = len(val_dataset.dataset) 
    

        # print(inps.shape)
        # print(labels.shape)
        
        # inputs[0].host = np.ascontiguousarray(inps.cpu().numpy())
        # # inps = inps.numpy()        
        # # inputs[0].host = inps.reshape(-1)
        
        # t1 = time.time()
        # trt_outputs = do_inference(
        #     context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        # )  # numpy data
        # t2 = time.time()
        # feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

        # feat = torch.tensor(feat)
        # _, preds = torch.max(feat, 1)

        # running_corrects += torch.sum(preds == labels.data)

    print(f"Accuracy with TRT {quant_mode} infer : {running_corrects / len(val_dataset) * 100}%")


if __name__ == "__main__":
    # main("fp16")
    main("int8")

# pytorch
# Accuracy : 67.7699966430664%

# trt KL INT8
# Accuracy with TRT int8 infer : 65.06999969482422%

# dipoorlet MSE INT8
# Accuracy with TRT int8 infer : 66.54000091552734%

# dipoorlet MSE+Brecq INT8
# Accuracy with TRT int8 infer : 67.27999877929688%




# pytorch
# Accuracy : 73.24%

# trt fp16
# Accuracy with TRT int8 infer : 73.28%

# dipoorlet MSE INT8
# Accuracy with TRT int8 infer : 66.54000091552734%

# dipoorlet MSE+Brecq INT8
# Accuracy with TRT int8 infer : 67.27999877929688%

