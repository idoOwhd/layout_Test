# 共享讨论归档：layout trade-off 与 cost model

- 原始链接：https://chatgpt.com/share/6aa11b07-6128-83ec-9a74-dac4b8297f19
- 页面标题：`ChatGPT - 内存Layout性能评估`
- 访问核验：2026-09-22 通过公开 share URL 成功返回完整会话载荷。
- 归档方式：本文保留来源、核心公式与可复核论点；不把 ChatGPT 讨论当作一手论文证据。实验结论以本地 raw data 和 pinned GitHub/官方文档为准。

## 核心命题

讨论将 layout 表示与 layout 选择严格区分：现有框架已能表示多数 LLM kernel 需要的 GMEM/SMEM/REG/TMEM、thread/warp/lane/register ownership、MMA fragment、swizzle 与 persistent KV 布局；更普遍的缺口是，如何对可表示候选进行跨算子、跨存储层级、跨时间的组合代价建模。

讨论给出的总代价分解为：

\[
C(G,L,\Theta,\xi)=
C_{memory}+C_{communication}+C_{instruction}+C_{resource}+
C_{pipeline}+C_{materialization}+C_{lifetime}+C_{dynamic}.
\]

其中：

- \(G\) 是子图拓扑和算子语义；
- \(L\) 是跨 edge/tensor 的 layout assignment；
- \(\Theta\) 是 tile/warp/stage/fusion/instruction 等 schedule 状态；
- \(\xi\) 是 batch、query/KV length、page table、MoE expert skew、reuse/lifetime 等运行时状态。

对 edge \(e\) 的状态可写为：

\[
S_e=(L_e,M_e,O_e,I_e),
\]

分别表示 layout、memory locale、ownership 和 instruction role。

## 对旧 unary + pairwise 公式的限定

基本形式是：

\[
J(\mathbf L)=\sum_{v\in V}U_v(L_v)+
\sum_{(u,v)\in E}P_{uv}(L_u,L_v).
\]

`pairwise` 不等于贪心或局部搜索；它仍可以作为全图优化目标。但 LLM layout 中以下因素不能被简单 unary/pairwise 完整表达：

- 多 tensor 共同造成的 register/SMEM/TMEM 容量和 occupancy 非线性；
- fusion 是否消除 materialization；
- 多 consumer fan-out 和持久 state 的重复读取；
- MoE routing/gather 中由 `topk_ids`/expert counts 决定的 runtime permutation；
- paged KV 的 page table、allocator、prefix reuse 与 block transfer；
- 通信与计算是否重叠。

因此应将目标扩展为 constrained contextual objective：

\[
\min_{\mathbf L,\Theta,\mathcal P,\pi}
\mathbb E_{\xi\sim D}
\left[C(G,\mathbf L,\Theta,\mathcal P,\pi;\xi)\right]
\quad\text{s.t.}\quad
Legal_{semantic}\land Legal_{ISA}\land Legal_{resource}\land Legal_{ABI}.
\]

\(\mathcal P\) 是 layout domain 划分，\(\pi\) 是运行时策略。

## 对常见子图的关键补充

- Attention 应联合建模 \(QK^T\rightarrow softmax\rightarrow PV\)，而不是只为某个 GEMM accumulator 选 layout。MMA1 accumulator、row reduction 和 MMA2 operand 的偏好可能冲突。
- MoE/Gather 的代价必须包含 token-major 到 expert-major 的 runtime permutation、padding/ragged representation、grouped GEMM 以及 unpermute。
- KV cache 是长生命期 state；layout 同时服务 attention kernel、connector/offload、allocator 和 page metadata。vLLM 当前更接近“合法交集 + preference”而非测量驱动的 analytical argmin。
- 同一 layout 必须分清 logical meaning、physical storage order、runtime indexing/metadata 三者；`block_table`/`topk_ids` 不是普通 stride layout，但会改变 layout 的有效代价。

## 本地实验如何使用该讨论

本归档中的公式是建模候选，不是实验结论。A10 的 256-case 实验用以下反事实给它们提供或否定数据：

- consumer/edge winner inversion；
- direct/view/materialize 的成组对照；
- shape/phase/batch/page boundary；
- native 与 extended layout candidate；
- vLLM/SGLang 的 page/layout/backend 对照与 request concurrency；
- KV data + slot metadata 的原生 writer 对照。

