#!/bin/bash

export CDSP_LIBRARY_PATH="/opt/app/tmp/qnn_226/lib/hexagon-v75;/dsplib/image/dsp;/dsplib/image/dsp/cdsp0;/mnt/etc/images/dsp/"
export CDSP0_LIBRARY_PATH="/opt/app/tmp/qnn_226/lib/hexagon-v75;/dsplib/image/dsp;/dsplib/image/dsp/cdsp0;/mnt/etc/images/dsp/"
export CDSP1_LIBRARY_PATH="/opt/app/tmp/qnn_226/lib/hexagon-v75;/dsplib/image/dsp;/dsplib/image/dsp/cdsp1;/mnt/etc/images/dsp/"
export HWINFO_LIB="/mnt/lib64/dll:/mnt/usr/lib64"
export PATH="/opt/app/tmp/qnn_226/bin:$PATH"
export VENDOR_LIB="/opt/app/tmp/qnn_226/lib"
export ADSP_LIBRARY_PATH="/mnt/etc/images/dsp:/opt/app/tmp/qnn_232_patch_for_laneline/lib/hexagon-v75"
export LD_LIBRARY_PATH="${HWINFO_LIB}:${VENDOR_LIB}:$LD_LIBRARY_PATH"
export LD_PRELOAD="/ifs/lib64/libsocket.so.4"

echo "处理模型 : qnn_resnet18_raw_data_panel.context.bin"

qnn-net-run --backend libQnnHtp.so \
    --retrieve_context /var/ssd_data0/bruce/qnn_resnet18_quant_fp16.context.bin \
    --input_list /var/ssd_data0/bruce/qnn_resnet18_raw_data_panel.txt \
    --output_dir //var/ssd_data0/bruce/qnn_resnet18_quant_fp16_output 
    # --synchronous
    # --profiling_level detailed \
    # --keep_num_outputs 0 &


# qnn-profile-viewer --input_log /data/cdd/qnn-profiling-data.log --output /data/cdd/profile_od_226.csv