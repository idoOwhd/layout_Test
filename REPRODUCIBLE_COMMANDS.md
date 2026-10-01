# Reproducible GPU-1 commands

Every runner below creates a new timestamped directory unless `--output` is
provided.  All runners refuse to overwrite an existing output directory.

## Previous representative v13 experiment

```bash
cd /home/liangyilei/ladder_home
GPU_PHYSICAL_INDEX=1 \
PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python \
TORCH_PYTHON=/home/liangyilei/conda/envs/cuda-opt/bin/python \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
bash staged/baseline_framework/layout_research/run_v13_all_gpu1.sh --full \
  --model /home/liangyilei/ladder_home/staged/baseline_framework/layout_research/models/Qwen2.5-0.5B-Instruct-7ae5576
```

## Previous observation-validation experiment

```bash
cd /home/liangyilei/ladder_home
GPU_PHYSICAL_INDEX=1 \
PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python \
TORCH_PYTHON=/home/liangyilei/conda/envs/cuda-opt/bin/python \
TVM_PYTHON=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs/layout-tvm/bin/python \
CUTLASS_DIR=/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass \
CHECK_SOURCE_URLS=0 RUN_ALL_640=1 CUDA_WARMUP=8 CUDA_ITERATIONS=30 \
TRITON_WARMUP=8 TRITON_ITERATIONS=30 \
bash staged/baseline_framework/layout_research/run_observation_validation_gpu1.sh --full
```

## v13: 640 LLM cases × six frameworks × ten RQs

```bash
cd /home/liangyilei/ladder_home
GPU_PHYSICAL_INDEX=1 \
PYTHON_BIN=/home/liangyilei/conda/envs/cuda-opt/bin/python \
TORCH_PYTHON=/home/liangyilei/conda/envs/cuda-opt/bin/python \
FRAMEWORK_ENV_FILE=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/framework_envs.generated.sh \
SERVING_MODEL_PATH=/home/liangyilei/ladder_home/staged/baseline_framework/layout_research/models/Qwen2.5-0.5B-Instruct-7ae5576 \
CUTLASS_DIR=/home/liangyilei/cursor_home/Ladder/3rdparty/cutlass \
CUDA_WARMUP=8 CUDA_ITERATIONS=30 TRITON_WARMUP=8 TRITON_ITERATIONS=30 \
ALL_640_WARMUP=3 ALL_640_ITERATIONS=10 \
bash staged/baseline_framework/layout_research/run_v13_640_all_frameworks_gpu1.sh
```

The Cartesian-product report always contains 38,400 cells.  A cell is marked
native-supported only for an exact case-level paired counterfactual.  Hexcute's
public artifact targets A100/H100, so its A10 cells remain explicitly missing;
RQ5 remains blocked on one GPU and H4.4 remains blocked without another GPU
architecture.
