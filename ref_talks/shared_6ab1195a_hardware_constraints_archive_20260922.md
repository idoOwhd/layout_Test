# 共享讨论归档：GPU layout 约束与有效搜索空间

- 原始链接：https://chatgpt.com/share/6ab1195a-ab98-83ea-b078-eb047722e5c5
- 页面标题：`ChatGPT - 计算Layout组合数量`
- 访问核验：2026-09-22 通过公开 share URL 成功返回完整会话载荷。
- 证据边界：本文是对讨论的可读归档；硬件事实仍应以 CUDA Programming Guide、PTX ISA 和 pinned CUTLASS/CuTe 源码为准。

## 不能把各层 layout 当作独立笛卡尔积

错误计数是：

\[
N_{layout}=N_{GMEM}N_{SMEM}N_{REG}N_{TMEM}.
\]

更准确的计数是：

\[
N_{valid}=
\sum_{x\in\Omega}
\mathbf 1[
C_{semantic}(x)\land C_{ISA}(x)\land C_{copy}(x)\land
C_{resource}(x)\land C_{graph}(x)].
\]

上游少数选择确定后，许多内部 layout 是 dependent 或 derived，不能继续作为独立乘数。

## 三类决策

### Independent choice

主要是跨 kernel 边界仍可见的状态：

- external GMEM A/B/C/D major/order/packing；
- graph-visible QKV schema、NHD/HND、paged/block KV；
- KV domain/pool/page organization；
- CTA tile、warp/TiledMMA mapping、MMA instruction family（严格说属于 schedule 生成变量）；
- output D layout 和 persistent metadata organization。

### Dependent choice

候选集只有在上游 context 已知时才有意义：

- A/B SMEM base layout/swizzle；
- GMEM→SMEM tiled copy；
- `ldmatrix`/SMEM→REG decomposition；
- epilogue tile、epilogue SMEM layout 和 vector store；
- Blackwell `tcgen05.ld` 与 TMEM→REG/SMEM epilogue 策略。

其合法集是：

\[
L_S\in LegalSMEM(MMA,major,dtype,tile,copy,stages,capacity).
\]

### ISA/rule derived

- Ampere `mma.sync` A/B/accumulator register fragment；
- 选定 `ldmatrix` 后的 lane/register ownership；
- Blackwell TMEM accumulator data-path layout A–G；
- TMA/tcgen05 descriptor 对已选映射的编码；
- staged layout 在 base layout 和 stage 数确定后的扩展。

这些状态不应再作为独立 layout 搜索轴。

## REG、SMEM 与 GMEM 的真实自由度

### Register/TMEM

选定 instruction shape、dtype、major/operand role 后，fragment mapping 通常是 ISA-defined，独立自由度接近 0。真正可选的是上游 MMA shape、warp repetition、CTA-group/warp-specialization，不是任意 fragment permutation。

### Shared memory

SMEM 是重要的 kernel-private 优化层，但候选受 bank mapping、alignment、copy width、`ldmatrix`/TMA/tcgen consumer、capacity 和 stage depth 共同约束。“bank conflict 最少”只是必要特征之一，不是唯一目标：两个都 conflict-free 的 layout 仍可能因 vectorization、occupancy、pipeline 或 epilogue 代价而性能不同。

### Global memory / persistent state

“连续”不能唯一确定 GMEM layout，因为需要回答“哪个轴在哪个 consumer 的访问顺序中连续”。NHD/HND、row/column major、page/head/token order、block-major/layer-major、scale/index/page-table 的共同组织都可以保持某一维连续，却服务不同 consumer。因此 graph-visible GMEM/persistent-state layout 仍是主要科研空间。

## 对 Ampere 和 Blackwell 的量级判断

- 对固定 tile 的 Ampere FP16/BF16 dense GEMM，纯 data-path layout 常只有 \(O(1)\) 到 \(O(10)\) 个有意义的上游候选；扩展 tile/warp/stage/schedule 后才到 \(O(10^2)\) 或更多。
- Blackwell 多了 TMA/TMEM/tcgen05 state，但 descriptor 和 TMEM A–G 不是独立乘数。真正上游候选是 MMA shape、CTA-group/WS、major/source、SMEM swizzle 与 epilogue strategy；固定 tile 下的合法配置空间可到 \(O(10^2)\) 到 \(O(10^3)\)，但其中很多已是 schedule 而不是纯 tensor layout。

## 对 LLM layout RQ 的启示

当 REG/TMEM 接近固定、SMEM 被约束 grammar 缩小后，重点转向：

1. 跨 producer/consumer 的 graph-visible layout rank inversion；
2. conversion/materialization 与 reuse/lifetime 的摊销；
3. 不同 phase/batch/shape/hardware 下最优策略是否稳定；
4. data 与 page table/slot/scale/routing metadata 的联合组织；
5. page/block/domain/allocator/placement 等 persistent-state 策略；
6. 受约束候选 grammar 是否遗漏了可实现但更快的布局。

