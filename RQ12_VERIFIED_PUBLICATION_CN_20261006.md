# 已验证 RQ1/RQ2 源码发布范围（2026-10-06）

本次发布对应“上一个问题之前 RQ1/2 已经验证的代码”，不包含最新的 memory-layer sharing 补充实验，也不宣称所有 upstream PR/issue 已完成原生复现。

包含 RQ1 多样性测试、失败单元补跑、正确性/覆盖检查，RQ2 real-forward graph growth、conditional domain-anchor、失败组补跑、paired holdout 统计审计，以及此前已完成的 RQ1/RQ2 汇报生成/核对代码。所需设计清单、固定版本的模型配置和 PR/issue 文本证据也保留。新增文件范围见 RQ12_VERIFIED_SOURCE_FILES.txt（目录项表示该目录内的源证据输入）。

排除 results/ 中的所有原始结果、报告、图和编译产物；排除环境、模型权重、下载的完整框架仓库。排除新增加的 rq2_share_*、analyze_rq2_share_levels.py、run_rq2_share_levels.sh、相关新测试/文档与 audit_rq2_existing_shared_layout.py。其它尚在修订/排队的 RQ 补充实现不在本次提交范围内。

## 核对依据

RQ1 核心执行/分析代码已与修复后完整实验的 preflight 源码指纹核对。RQ2 核心 GPU 执行代码已与已完成 graph-growth 的 source_snapshot 核对，domain-anchor 与已完成 conditional 实验快照核对。rq2_growth_search.py 保留此前已通过回归检查的置信区间修正：paired bootstrap 使用与 speedup 点估计一致的 ratio-of-medians，不能与 median-of-paired-ratios 混淆。

这是代码来源/回归检查，不是新的性能测量。此前代码中的 native replay / analogue / blocked 分类保持不变。没有使用全目录 git add -A，没有修改旧结果。

## 克隆后的输入路径

原有生成器仍使用 layout_research 的同级 real_world_shapes 路径。本次不修改已验证的生成器或 benchmark，而在 reproduction_inputs/rq12/real_world_shapes 中额外保存精确的输入快照（配置文件，不是权重）。

在克隆仓库根目录执行以下命令可在缺失时恢复同级输入；不覆盖已有文件：

```bash
mkdir -p ../real_world_shapes
cp -a -n reproduction_inputs/rq12/real_world_shapes/. ../real_world_shapes/
```

已有安装/编译入口 setup_and_run_native_frameworks_gpu1.sh 保持不变。框架环境需在运行主机上单独安装。具体实验参数、适用范围和 smoke/full 顺序参见 RQ1_DIVERSITY_SUPPLEMENT_CN.md、RQ2_GRAPH_GROWTH_IMPLEMENTATION_AND_RUN_CN_20261002.md、RQ2_CONTROLLED_FOLLOWUP_METHODS_AND_SCOPE_CN_20261002.md。

## 发布前回归检查

禁用 GPU 可见性，运行以下历史回归测试：76 项全部通过。没有重新计时 GPU，也没有修改之前的实验结果。

```bash
CUDA_VISIBLE_DEVICES='' /home/liangyilei/conda/envs/cuda-opt/bin/python -m unittest \
  test_rq1_diversity test_rq1_failure_retry \
  test_rq2_conditional_smoke test_rq2_domain_anchor test_rq2_failed_groups \
  test_rq2_growth_search test_rq2_mixed_requests test_rq2_progress_and_statistics \
  test_rq2_result_audit test_rq2_smoke_approval test_rq2_source_audit
```

未来性能结果不保证与旧机器上的 timing 数值逐位相同，应保留新实验目录和环境/源码指纹。
