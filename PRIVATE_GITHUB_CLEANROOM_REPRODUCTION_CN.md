# 私有 GitHub 仓库的 clean-room 复现

统一入口是：

`private_github_cleanroom_reproduce.sh`

## 仓库中包含什么

运行 `export` 后，`SOURCE_FILE_LIST.txt` 是唯一权威清单。主要包括：

- `layout_research/` 中的运行器、分析器、CUDA/Python源码、RQ registry、测试和 `ref_talks`；
- 640-case LLM 输入清单、shape发现代码和共享workload；
- 2025–2026视觉清单与参考runner源码；
- CUTLASS softmax boundary 的Python driver与CUDA源；
- 固定版本文件 `reproduction_versions.env`。

不包含旧 `results`、模型权重、虚拟环境、下载的框架源码、CUTLASS checkout、编译缓存、动态库或旧安装日志。

## 创建并推送私有仓库

只创建干净目录：

```bash
cd /home/liangyilei/ladder_home
bash staged/baseline_framework/layout_research/private_github_cleanroom_reproduce.sh \
  export --export-dir /tmp/layout-validation-private
```

自动创建并推送私有仓库（要求 `gh auth login` 已完成）：

```bash
bash staged/baseline_framework/layout_research/private_github_cleanroom_reproduce.sh \
  publish \
  --export-dir /tmp/layout-validation-private \
  --github-repo OWNER/layout-validation-private
```

## 在新GPU主机复现

```bash
git clone git@github.com:OWNER/layout-validation-private.git
cd layout-validation-private

bash staged/baseline_framework/layout_research/private_github_cleanroom_reproduce.sh \
  all --gpu 1 \
  --output-root "$PWD/reproduction_outputs/run_$(date +%Y%m%d_%H%M%S)"
```

`all`依次完成主机检查、Python 3.12 bootstrap、vLLM/SGLang安装、CUDA TVM源码编译、pinned CUTLASS checkout、pinned 0.5B模型下载、16-case smoke、256-case高判别力实验以及v10/v13 640-case实验。

推荐第一次使用时分阶段执行：

```bash
SCRIPT=staged/baseline_framework/layout_research/private_github_cleanroom_reproduce.sh
bash "$SCRIPT" source-check
bash "$SCRIPT" doctor --gpu 1
bash "$SCRIPT" install --gpu 1
bash "$SCRIPT" prepare --gpu 1
bash "$SCRIPT" smoke --gpu 1
bash "$SCRIPT" full-diversity --gpu 1
bash "$SCRIPT" full-640 --gpu 1
```

每次实验默认使用时间戳目录；显式指定的输出目录如果已经存在，脚本会拒绝覆盖。

复现的是相同源码、输入契约、候选layout、correctness gate、运行次数和报告schema。不同GPU、驱动、CUDA版本、温度和时钟下，绝对延迟不应被要求逐位相同；跨硬件应比较winner、crossover、regret和相对加速，并保留脚本写出的环境清单。
