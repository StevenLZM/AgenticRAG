# 正式语料 Agentic RAG 自动评测 v2 实施方案

> 执行方式：逐项实施、逐项验证；用户于2026-09-30明确授权实施整套评分器改造。每次继续先阅读本方案与 `docs/formal-rag-evaluation-v2-progress.md` 的恢复点。

**目标：** 用正式环境原文建立可追溯金标，通过真实 QueryGraph 的最终答案与金标比较评价系统；阶段检索分数用于定位原因，不能替代答案正确性。

**架构：** 原文事实层与当前分块 ID 映射层分离。冻结语料快照，真实执行查询，保存阶段排序与最终上下文；规则评分和独立 LLM Judge 分别评分，报告合并质量、运行可靠性与成本。

**技术：** Python、现有 MySQL/ES/API、现有持久化查询轨迹、固定版本 Ragas 0.4.2。

## 最新执行索引（2026-10-01）

下文保留原实施清单与历史统计；逐项当前状态以`docs/formal-rag-evaluation-v2-progress.md`为准，可执行命令见`docs/formal-rag-evaluation-v2.md`。v2契约、正式CLI、真实轨迹、端到端裁判、关键字段、阶段指标、成本账本、恢复核验、调参CLI及人工审核入口均已有实现。最新回归951 passed/47 skipped，正式20题真实测评正在`formal-v2-smoke-20261001`运行；不是4453题全量成绩。人工语义审核、graded qrels、Judge校准、自然问法/困难负例扩充、完整validation/test与参数验收仍未完成。

## 2026-09-30 完整正式处理与定位修复进展

- [x] 原1000份合成文档普通上传全部completed，经API8000/MySQL/正式ES9200核验；两份旧index-v2原文通过正常reprocess接口发布新版本，旧原文和版本保留。
- [x] 完整快照`var/artifacts/evals/formal-corpus-gold-v2-20260930-production/snapshot.json`覆盖1003 active文档，全部searchable；统一docling-v1/ingestion-v2/text-embedding-v3/index-v3，共2231 Parents/2237 Children。
- [x] 全部原文/AST/manifest及当前生产分块器重算结果一致，SQL/ES内容和定位元数据一致；见`production-pipeline-verification.json`。没有为了金标使用特殊分块器。
- [x] `gold.v2.2.jsonl`共4453题，dev=2670/validation=890/test=893。12105事实组中12104 mapped、parent_only=0、unmapped=1；初版592组逐条全部修复，见`gold-binding-verification.json`。
- [ ] 新发现1处真实Docling阅读顺序异常：`S047-2026-product:section-4:F3`，正文中的“介”被单独错排。新进程复现，保持原文答案和未映射状态，不删题或假标qrel；须另行修复正常解析器并发布新版本、新快照。详见`docs/formal-document-pipeline-repair-progress.md`。
- [x] 本轮代码全仓859 passed、47 skipped，相关ruff通过；独立审查无剩余Critical/Important。这些是代码验证，不是RAG质量分数。
- [ ] strict v2契约、人工语义审核、测评入口/轨迹/最终答案评分仍未完成，没有新的真实端到端测评成绩。

以下839文档/3724题数据是首版历史快照，不是当前完整语料统计；保留历史证据，不覆盖冻结产物。

## 首版已完成与边界（历史记录）

- [x] 正式环境只读盘点并冻结 `var/artifacts/evals/formal-corpus-gold-v2-20260930/snapshot.json`。
- [x] 从保存的实际上传原文生成 `gold.v2.1.jsonl`，共 3724 条，覆盖 default_user 全部 839 份 active 文档。
- [x] 验证原文 SHA256、836 份合成文档每份四个章节、表格缓存计算；3 份历史 PDF 使用原文事实，不将电话、身份证、邮箱写进金标。
- [x] Parent 事实定位：10123 个事实组均有 Parent；9531 组有单 Child 完整锚点，592 组暂为 parent_only，不能宣称 Child qrels 已完整。
- [x] 数据按公司分组切分，两年和不同文档类别在同一 split。当前 dev=2670、validation=890、test=164；因上传不完整，test 覆盖不均衡，暂不做发布验收。
- [x] 后续已完成原1000份上传并冻结完整生产快照；本历史快照仅含当时active的836份合成文档，不包含当时剩余164份。
- [ ] 人工语义抽检、自然问法、多跳与困难负例扩充；当前 review_status 为原文程序核验通过、人工审核待完成。
- [ ] 接入 v2 CLI、真实评分器和新报告；本轮没有执行正式查询测评，没有新的质量成绩。

当前问题分布：章节理解3344、单跳87、表格计算168、明确资料不足83、多跳42。章节模板占比过高，不能仅凭数量评价基准质量。

索引边界：839 份有效原文中，837 份在 `agenticrag-children-active -> agenticrag-children-index-v3` 有完整 Child 数；两份历史 index-v2 文档不在当前别名。不要静默排除它们，也不要将上传失败/排队标为查询无答案。API/Worker 本次只读盘点未发现运行进程，下一次真实执行须先检查服务健康。

初稿`gold.v2.jsonl`保留为历史草案；本历史快照使用`gold.v2.1.jsonl`，完整语料使用production目录的`gold.v2.2.jsonl`。不覆盖已有冻结文件。

## 全局约束

- 使用正式 API :8000 / ES :9200 的真实查询流程；评测只读原索引，不自动上传、重建或删除业务文档。
- 用户隔离：当前只存在 default_user；未来用户各自生成快照，禁止跨用户合并证据。
- 金标从原文生成，不能从系统答案、TopK 命中反推答案或排除难题。
- `index_generation` 只是索引代际，不足以标识数据快照；绑定 cluster UUID、具体索引 UUID、文档版本+原文哈希、Child 内容/成员哈希、配置和提示词哈希。
- schema/单位/抽取失败是评分故障，不是假答案正确或模拟分数；禁止用契约测试结果充当真实评测。
- 分别报告执行成功率、评分覆盖率及质量分母；评分缺失输出 null，验收视为不通过，并报告成功案例分数与保守全量通过率，不能靠删失败样本提高成绩。

## 任务 1：金标契约与数据质检

文件：新增 `evals/gold_v2_models.py`、`evals/gold_v2_validation.py`；扩充 `evals/formal_gold.py`、`scripts/build_formal_eval_gold.py`；测试 `tests/unit/evals/test_formal_gold.py`。

接口：`load_gold_v2(path: Path, snapshot: CorpusSnapshot) -> list[GoldCaseV2]`；`validate_gold(cases, originals, snapshot) -> GoldValidationReport`。

- [ ] 先写失败测试：重复 case_id、跨用户证据、原文哈希变化、错误计算、NaN/Inf、值无单位、事实无来源均拒绝。
- [ ] 将 schema_version=2 的松散 JSON 转为 strict Pydantic 模型；事实记录 subject/conditions/predicate/value/unit/polarity。计算记录原单元格、输入、公式、舍入容差。
- [ ] 原文定位补 page/文本区间/工作表单元格；用 Canonical AST 仅映射分块，不能将解析错误当原文真值。
- [x] 补`child_evidence_sets_any_of: list[list[child_id]]`：组内AND，组间OR，跨Child事实要全部片段；`child_ids_any_of`仅表示单Child独立充分证据。真实AST区间联合覆盖，拒绝缺口，标题需要确实在Child context内。包含真实HybridChunker分隔空白回归测试。
- [ ] 对同义/重复文档登记 equivalent_fact_id；同一事实的两份证据是替代来源，不要求全部召回，不重复加分。两份简历的相同教育事实需合并等价组。
- [ ] 初始二元 qrels 保持可复现；新增人工核验的 graded qrels 0/1/2/3（无关/相关背景/部分事实/充分证据），禁止自动把未命中的来源标无关。
- [ ] 每类至少抽检20题、所有多跳和拒答题、全部历史文档题；错误按模板回溯到全体同模板题。保留 reviewer/reason/修改历史。
- [ ] 增加自然问法、同城不同公司、同公司不同年、目标与实际、批准试运行与正式商用、表格单位等困难负例；同一问法变体及关联公司全部同 split。
- [x] 完成上传后按全部50公司重新切分held-out：dev2670/validation890/test893，年份及文档类别同公司不跨split。类别/模板宏平均报告尚待任务6实现，不能用4000条模板题淹没多跳错误。

验收：原文文档覆盖100%；错误来源0；已审核集的事实、Parent 与 Child 充分证据组全部可解释；冻结后不能用 test 调参。

## 任务 2：通用正式语料入口与快照校验

文件：改 `evals/command.py`、`scripts/eval_rag.sh`；新增 `evals/corpus_snapshot.py`；测试 `tests/unit/evals/test_formal_eval_command.py`。

接口：`preflight_snapshot(snapshot, live_clients, mode: Literal['strict','robustness']) -> PreflightReport`。

- [ ] 先写失败测试：任意一份无关文档增删、alias切换、相同 generation 内数据变化、配置变化都影响实验指纹。
- [ ] 去掉只认旧8文档/24题 source-run 的前提；读显式 `--dataset` 与 `--corpus-snapshot`，保留旧格式适配，不混写旧结果。
- [ ] ES 使用 PIT/search_after 或明确清理的 scroll，不能依赖 size=10000 假全量。冻结实际检索使用的别名及全文档，含无关干扰文档。
- [ ] strict 模式检测漂移就停止；robustness 模式另建实验记录缺失文档，保留对应题并按缺失检索处理，不将原本 answerable 改无答案。
- [ ] 查询新建隔离 session，默认不读写长期记忆，避免一题污染下一题；多轮任务显式定义 turn 和会话范围。生产实现需加受控 evaluation metadata 与 memory policy，不直接清空用户记忆。
- [ ] 预检服务/模型/裁判可用性和费用上限；已有索引路径不得触发上传。续跑绑定 dataset+snapshot+config+judge 哈希。

目标命令（尚未实现，不能当可执行说明）：

```sh
./scripts/eval_rag.sh --dataset <gold.v2.2.jsonl> --corpus-snapshot <snapshot.json> \
  --api-url http://localhost:8000 --split test --output-dir <new-run-dir>
```

验收：一条命令在现有正式索引完成预检→真实查询→评分→报告；异常可按原 run_id 续查，不自动重复 POST。

## 任务 3：真实端到端轨迹

文件：改 `evals/collector.py`、`evals/clients.py`、`evals/run.py`、`src/agentic_rag/retrieval/graph.py`；必要时新增 `evals/trace_v2.py`。

接口：`collect_case_trace(run_id: str) -> CaseTraceV2`，包括实际最终输出，不是拟造答案。

- [ ] 测试 fast_rag、research 多轮/多个 Subagent、cannot_answer、round_limit、timeout、chat；缺上下文不得拿 TopK 补成“实际上下文”。
- [ ] 每次 retrieval 分别记录 query、Dense/BM25/RRF/rerank 的原始有序 Child ID、分数和候选预算；Parent聚合排序与截断单独记录，不能提前去重到Parent后称为Child排名。
- [ ] 记录送给回答模型的实际有序上下文及裁剪，和最终回答、引用、事实审核、状态、router选择。
- [ ] 记录 Todo/DAG 依赖满足、工具和 Subagent 调用、检索调用数、行动数、重试数；“研究轮数”不能混同检索次数。
- [ ] 每阶段延时、TTFT、总耗时、实际provider usage和模型/embedding/rerank请求数持久化；缺 usage 明确 unknown，不把字符预算写成付费 token 数。

验收：一次真实多跳运行能重建各次候选、最终证据与最终答案，并与 API/run_id/checkpoint 对账。

## 任务 4：Recall、MRR、NDCG 的正确口径

文件：新增 `evals/retrieval_metrics_v2.py`，改 `evals/metrics.py`/`evals/report.py`；测试 `tests/unit/evals/test_retrieval_metrics_v2.py`。

接口：`score_retrieval_stage(case: GoldCaseV2, ranked_ids: list[str], ks: list[int], level: str) -> StageMetrics`。

- [ ] 测试相关答案排第31、重复证据、两跳只找到一跳、同Parent多个Child、空检索和截断不足K。
- [ ] 分阶段、分 Child/Parent 算 Recall@K（K=1/3/5/10/20/30/50，超过阶段实际输出仍按实际列表处理）。另报 eligible、evaluated、failed、qrel_unmapped；映射不足不能假计0。
- [ ] 文档/块 Recall 之外增加 FactRecall 与 EvidenceGroupCoverage，满足一个充分证据集才覆盖事实；跨年比较需两年两组都覆盖，报 CompleteEvidenceRate。
- [ ] MRR 使用首次充分相关证据的倒数排名；它只说明第一条答案有多靠前，不能代替多跳完整性。
- [ ] 二元 NDCG 与 graded NDCG 分开命名；graded 使用 `(2^rel-1)/log2(rank+1)`，IDCG 来自冻结 qrels，不从实际命中集合计算。
- [ ] 多轮分别报每次排序与全局事实覆盖/新事实增益；未实施全局融合不得给多轮列表伪造一个 RRF 排名。
- [ ] 索引不覆盖的真实可回答题保留在端到端分母；同时归因为索引/入库问题，不冒充排序问题。

验收：反例测试中“只找到2026年标准”MRR可以好，但跨年完整证据率不能满分。

## 任务 5：最终答案正确性为主指标

文件：新增 `evals/answer_fields.py`、`evals/answer_judge.py`；改 `evals/judge.py`、`evals/ragas_adapter.py`；测试 `tests/unit/evals/test_answer_correctness_v2.py`。

接口：`extract_answer_fields(question, response, field_specs_without_values) -> ExtractedFields`；`compare_fields(extracted, expected) -> FieldScore`；`judge_answer(case, response, actual_contexts) -> AnswerScores`。

- [ ] 测试：答案流畅但金额错、年份错、缺单位、值正确但对象错、遗漏、自己矛盾、拒答、答案与错误上下文一致、无效JSON。
- [ ] 抽取器只看问题/最终答案/字段定义，不给金标数值；要求 quote/evidence_span，未提及返回 null。Schema保证结构，Python规则决定数值/单位/条件是否正确。
- [ ] 对所有 critical_fields 精确单位换算和题目容差；错误、缺失、矛盾分别报告。不能从背景/金标补抽取结果；有一项关键事实错就不算整题成功。
- [ ] 接入 Ragas AnswerCorrectness（question+response+reference，显式指定judge和embedding）及独立 FactualCorrectness precision/recall/F1（response+reference）；后者防止语义相似掩盖事实错误。
- [ ] 保留 Faithfulness（答案是否受实际上下文支持）、AnswerRelevancy、ContextPrecision，并补 ContextRecall。Faithfulness高不表示金标答案正确。
- [ ] 核心输出 CriticalFieldAccuracy、AllCriticalFieldsPass、RequiredFactRecall、ContradictionRate、TaskSuccessRate；LLM阈值必须经校准，不把几个分数平均当发布通过。
- [ ] 无答案题单独 judge 正确拒答与错误断言；可回答题拒答是任务失败；round_limit/tool_failure/timeout 分别归因，不能算正确拒答或删除样本。
- [ ] 让 Judge 与回答模型独立（供应商/模型可配置）；记录版本、rubric、温度、输出、费用。至少100个分层样本人工校准，其中包含已知错误和同义正确答案；分歧保留待裁定。

验收：把470元答为410元即使检索Recall=1、Faithfulness高也不能通过；评分故障给unknown和覆盖率，不捏造0或1。

## 任务 6：Agent 行为评测与报告

文件：新增 `evals/agent_metrics_v2.py`；改 `evals/report.py`、报告渲染脚本。

- [ ] 测试有依赖Todo提前执行、独立Todo并行、Subagent重复事实、全局研究上限及确定缺材料。
- [ ] Todo按依赖关系验证，不要求唯一线性节点序列；按最终事实完成度判断计划是否有效，不只看todo数量。
- [ ] 多跳报各事实获得轮次、重复检索率、新证据增益、Subagent重叠率、预算耗尽率、错误拒答率、过度检索率。
- [ ] 报告首屏展示端到端答案正确性和整题成功率，再展示上下文和分阶段排序诊断；每个失败题提供原文事实→候选→最终上下文→最终回答的证据链。
- [ ] 每项分数有有效n/总n、缺失原因；按类型/公司/格式/路由分组，公司分组bootstrap置信区间，不把模板题当独立样本高估置信度。
- [ ] 脱敏导出：报告不包含原文敏感个人字段；本地完整轨迹受访问范围控制。

## 任务 7：候选数量与成本联合优化

文件：新增 `evals/tuning.py`、`evals/tuning_models.py`；改 `src/agentic_rag/retrieval/graph.py`、`reranker.py`，新增预算配置。

当前源码：Dense40/BM2540→RRF30→rerank10→Parent6；reranker内部 max_candidates=30。只扩大RRF而不改reranker硬截断无效。

- [ ] 先测试配置能贯穿全部阶段、rerank实际收到超过30的输入、ANN候选预算与召回K一致。
- [ ] 在dev做逐级筛选，不做全量笛卡尔积：Dense/BM25各20/40/80，RRF30/50/80，rerank输出10/20/30，Parent6/10/15；先查候选召回上限，再查排序，再查最终上下文截断。
- [ ] 固定索引/模型/提示词/金标/证据预算做配对比较；仅查询阶段改预算，不反复重入库。验证配置改变才产生新实验缓存键。
- [ ] 优先满足答案正确性/完整证据/错误拒答约束，再选P95延时、TTFT、LLM token、rerank候选开销的Pareto前沿；不假定候选越多越好。
- [ ] 用validation选方案，test一次独立验收；校验公司/问法家族不泄漏；生产A/B或灰度另行批准，不自动发布参数。

## 任务 8：回归与可恢复执行

- [ ] 20题分层smoke（使用真实查询/真实judge，不是模拟）；涵盖数值、表格、多跳、无答案、历史索引缺口。
- [ ] smoke确认轨迹/数值正确后跑validation全套；满足费用与并发约束，保存逐题ledger、query_run_id、judge状态。
- [ ] 测试中断/POST响应未知/评分429：续跑只恢复未完成阶段；已有查询不重复扣费执行；不能用不同配置复用裁判结果。
- [ ] 保存基线实验和全量报告；发布门槛待人工校准定值。硬门槛优先：跨用户泄漏0、虚假金标0、未知评分不得通过、关键事实错误单题不得通过。

下一步继续任务1：strict v2契约、原文定位与等价组、人工语义审核/困难负例，以及已记录的PDF解析缺陷。正式完整上传与原592组定位已完成；随后再做入口/轨迹/评分，最后调参，不能将本轮入库核验当真实RAG质量成绩。

## 指标依据

- Ragas 0.4.2 AnswerCorrectness：事实与语义相似结合，需显式embedding：https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/answer_correctness/
- Ragas 0.4.2 FactualCorrectness：独立事实precision/recall/F1：https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/factual_correctness/

这些是评分器能力说明，不是本项目已经得到的评分结果。
