# RQ2 实验局部编号与 v13 原始假设对齐审计

日期：2026-10-03。依据 [v13 §6.5](ref_talks/layout_summary_v13_complete_problem_trial_audit_RQ1_RQ10.md)。旧 raw、旧冻结源码与原报告不修改。

## 必须纠正的解释

新扩图代码中使用的局部标签“H2.1 growth / H2.2 QKV second representation / H2.3 remote anchor”不能与 v13 的同编号假设混为一谈。字段 `all_declared_H2_1_full_matrix_passed` 是现有1020图执行覆盖的兼容字段，不是“v13 H2.1 已被充分证明”的科学判据。

今后描述采用三个不会撞编号的 protocol ID：

- `GROWTH-NEARSET`：同一forward小图近优布局集合在扩图后是否仍足够。
- `QKV-DUAL`：L0 packed-QKV的显式第二份表示在给定有限选择下的成本/收益。
- `REMOTE-KV-ANCHOR`：固定其它方案，上游QKV/KV与远端KV选择的2×2条件响应。

它们是用户“图更大、算子种类更多，内部选择怎么变”的相关实验，但不是下面五个原假设的完整替代。

| v13原假设 | 原问题/判据 | 当前协议支持到哪里 | 必须另补的直接实验 |
|---|---|---|---|
| H2.1 / RQ2-E2 | consumer heterogeneity超过阈值后split domain获益 | QKV-DUAL能测一个局部split；扩图有间接拓扑变化 | 可控consumer偏好/次数/bytes异质性、统一域与多域的合法选择、实际转换与共享参数/状态、finite-space oracle |
| H2.2 / RQ2-E3 | executed-byte/time加权regret比等权投票更好 | 现有脚本不比较这两种policy | 固定候选支持集合；真实consumer频率/bytes/time trace；等权/加权训练与独立heldout对照 |
| H2.3 / RQ2-E0 | 低heterogeneity/reuse时一个shared domain更好 | QKV-DUAL可成为局部对象，尚无充分异质性扫描 | 预注册低/高heterogeneity正反对照，含真实存储/repair/lifetime开销 |
| H2.4 / RQ2-E4 | speculative acceptance改变target/draft/verify最优partition | 当前dense Qwen3 nested forward没有实际speculative接受/验证流程 | 真实target/draft/verify消费者集合、合法backend/layout交集、控制acceptance distribution与draft depth |
| H2.5 / RQ2-E5 | hybrid KV/recurrent state的split只在overhead阈值后获益 | 1020图并无实际Mamba/GDN状态与KV共享pool实验 | 真实混合模型forward/state geometry，shared page-domain vs separate pool，allocator/lifetime/consumer成本 |

独立进程、算法来源归因与source/base-head replay仍是另外的证据要求。旧v13中`COMPLETE_STRONG`说明source/reasoning链完整，不是GPU实验全部完成。

## RQ3也必须按原始假设区分

| v13原假设 | 当前能测/已有代码 | 未满足的判据 |
|---|---|---|
| H3.1 reuse break-even | core/shared/epochs/mixed可直接测完整horizon与多个R；GPU full尚待运行 | 原生policy/base-head、更多机制；应报告无crossover与不确定而不是强求正例 |
| H3.2 temporary-memory pressure使R*上移 | shared记录两表示活跃bytes公式 | buffer预分配、没有实际allocator/capacity干预；公式不算实验 |
| H3.3 transfer-native/overlap降低R* | GPU-resident ingestion只构成局部copy对照 | 尚无真实wire/host-transfer overlap；单卡扩展不能证明多卡网络 |
| H3.4合法view/stride存在竞争区域 | keep与row-materialization可比较真实GEMM penalty | 只是parameter地址映射的局部区域，不是所有KV/scale/prefix/on-chip布局 |
| H3.5低reuse不应materialize | R=1/2/3等小R已在设计里 | 原假设要求allocation/copy完整收费；持久buffer预分配，cold allocation不在现有计时，不能完整确认H3.5 |

另一个需补的RQ3原始场景：上游PR常从**已经加载的参数**出发。core把每个horizon定义为GPU-resident source→新表示的ingestion，所有route都有P；若研究“模型参数本已加载，只决定是否首次pack”，应另测无ingestion的合法keep/pack-once/pack-each，不能默默把core的P成本归到原引擎。

## 处理原则与运行顺序

当前RQ2正常执行、不推翻原始测量；解释收窄为上述局部协议。保留旧报告与字段，在新的审计/汇总中明确区分。用户要求的新RQ2源码与逐PR补充安排在RQ3后；本文件列出的直接实验不能冒称已经实现或跑过。

当前代码/卡0smoke改善了持久权重子问题的干预有效性和计算正确性，仍没有完成全部v13假设。不能通过改编号、增加shape数量或写`complete=true`解决这些科学设计缺口。
