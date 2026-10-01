# RAG 裁判保真与生产测评验收修改方案

> **For agentic workers:** 经用户审阅批准后，使用 superpowers:executing-plans 按复选框逐项实施；若用户明确选择子代理执行，再使用 superpowers:subagent-driven-development。本文件是方案草案，不是执行授权或已完成记录。

**Goal:** 防止事实裁判改写实体造成误判，建立可追溯的金标审核与裁判校准，最后完成真实 validation 和 test 测评。

**Architecture:** 回答事实必须绑定原文，金标评分依据单独冻结；程序校验明确字段，LLM 判断语义关系。裁判故障、系统错误、金标问题分别报告，校准合格后才用分数选择检索参数。

**Tech Stack:** Python、Pydantic、现有正式评测 CLI、真实 API 和查询轨迹、固定 Ragas 0.4.2；本方案不要求升级业务依赖。

**Spec:** 继承 `docs/superpowers/plans/2026-09-30-formal-corpus-evaluation-v2.md` 的目标与边界；本次补充设计见下文。当前事实依据为 `docs/formal-rag-evaluation-v2-progress.md` 的最后恢复点，以及本轮核对的评分器源码、S004 真实评分产物。

**实施范围更新（2026-10-01）：** 用户批准第一批 Tasks 1–3，并指定在既有 `real-rag-evaluation` 分支修改；实际本地分支为 `codex/real-rag-evaluation`。不自动提交/推送。任务4及之后的人工审核、裁判校准和全量验收尚未启动。

## 一 现状和修改范围

- 冻结金标 4453 题：dev 2670、validation 890、test 893。基于原有 1003 份正式文档，不重新生成或上传语料。
- 当前真实 smoke 有 20 个 Run：9 个回答、8 个研究轮次上限、3 个 cannot_answer 无正文。这不是全量测评。
- 当前 v4 裁判已经解决 claim 编号对齐，以及 P/R/F1 重复采样不一致；尚未校验拆解事实是否忠于回答。
- S004 回答中的公司为“虚构星河04公司”，拆解后却出现“虚构星尘04公司”；短 reference 又未包含问题中的公司、年份、城市，导致错误扣分。
- 已有 213 题人工金标审核队列、9 条有正文的整题校准队列，均未获人工批准。213 题是风险分层样本，不代表全部 4453 题人工审核。
- 当前校准工具只检查整题 verdict，不能证明 Ragas 每个指标或事实抽取已经校准。

本次先修改评测器、审核工具、报告和实验约束，不修改正式检索默认参数、业务解析器、分块器或 ES 文档。已知 PDF 阅读顺序问题保留为系统真实缺陷，不通过改金标或删题掩盖。

## 二 方案选择

1. 只强化提示词或更换 Judge：改动小，但无法检验实体是否被改写，只作为校准时的候选对照。
2. **推荐：原文绑定事实抽取、独立金标事实、规则与语义联合判定。** 可以定位错误发生在抽取、判定还是系统回答，复用现有工程较多。
3. 只做全文字符串或正则比较：便宜，但会错杀同义表达、跨句指代和正确单位换算，不作为通用主评估。

推荐方案并不保证消灭所有语义误判。原文片段相符只证明没有凭空造词，不能证明主谓关系、否定范围或事实覆盖完整；这些也必须测试、标注和校准。

## 三 全局约束

- 旧 gold.v2.2、旧评分、旧 Run 账本全部保留；新裁判用新版本和新实验目录。
- 只改裁判时优先只读回放现有 Run，不触发新的 Query POST；真实 Judge 调用仍收费，必须设置请求数、时间和费用边界。
- 不能从系统答案或本次检索结果生成、补全金标；Faithfulness 对照实际上下文，Correctness 对照原文核验的金标，两者不混用。
- 抽取器看不到金标值；允许从问题解析所指对象，但不得据问题补出回答未给出的金额、结论或推理步骤。
- 无法校验的抽取或裁判响应记 unknown/error，不记答案错误、不记通过；总样本数保留，发布门槛阻断。
- 系统超时、轮限、空正文等运行失败仍计任务失败，不能用“裁判未知”掩盖。
- 不自动重试到满意分数，不混用不同 Judge 版本的评分。任何允许的重试必须预先规定次数、原因并保留全部尝试。
- 不自动填写 human approval，不自动提交 Git，不创建工作树，不自动发布参数。当前工作区既有修改全部保留。
- 人工审阅和阈值批准是外部依赖；缺失时可以继续离线契约测试，但不得宣称生产验收通过。

## 四 评分设计

### 事实抽取只选择原文和关系

新增 `SourceSpan`，保存 source（response/question）、start/end、quote；Python 验证 Unicode 左闭右开区间与原文严格相等。只允许对唯一、完全相同的 quote 做确定性位置修正；重复引用有歧义则失败。

新增 `GroundedFact`：fact_id、evidence_spans、subject_spans、predicate_spans、value_spans、unit_spans、condition_spans、polarity_spans、coreference_links。语义字段可以为空但不能假装已知；至少有回答证据。LLM 不输出替代原文的自由 claim 文本，给判定器的内容由程序按这些位置取出，并附实际完整回答供消歧。

关键约束：

- 公司名、年份、金额、单位、否定和例外条件均有来源；明确说错的回答条件不能被问题中的正确条件覆盖。
- “该公司”“次年”等只记录指向和依据；有歧义时 unknown，不偷偷改成金标对象。
- 原始数值和规范化值分开存储。允许的单位、数字形式转换由版本化规则完成，不因不是原文连续数字就把正确中文数字或换算判错。
- 不同主体、不同条件不能拼成一个事实；相同事实重复表述不重复加分。记录未覆盖的候选事实片段，抽取遗漏必须进入校准。
- 真正无事实的寒暄、明确拒答不走普通事实分数，使用对应路由或拒答评价。抽取器错误返回空列表则是评分故障。

### 完整金标使用独立评分说明

新增 `GoldAnswerSpec`，按 case_id、原 gold case hash 和 corpus snapshot 绑定：question_conditions、required_facts、optional_facts、source_fact_ids、source_refs、critical_rules、review_binding。

从现有金标、critical_fields、原文事实构建草案，不在评测时让模型临时扩写 reference。每项 required/optional 的选择必须可审计，禁止看了系统答案后降低要求。

S004 的评分依据应包含公司、2026 年、深圳、500 元/人/晚，并独立记录含税条件。是否要求答案明确说“含税”须在校准之前确定；若要求且未答，应扣完整性分，而不是把整个公司事实判错。

评分说明是 sidecar 文件，其内容 hash 加入实验绑定；不覆盖原 gold.v2.2。若发现原金标事实真正错误，则另发新金标版本和差异记录，现有跨 gold hash 恢复限制继续生效。本方案不新增绕过该限制的快捷开关。

### 主评估与 Ragas 诊断分开

- 主事实评估：已验证回答事实与审核后的参考事实双向核对，输出 supported、contradicted、not_in_reference；裁判结构错误另记 judge_error。
- not_in_reference 表示金标无法支持，不等同证明现实中错误；单独报告。可用于支持率计算，但不得无条件写成“事实造假”。
- 主事实 Precision = 被金标支持的非重复回答事实数 / 非重复回答事实数。
- 主必需事实 Recall = 回答覆盖的必需金标事实数 / 必需金标事实数。
- F1 从上述 P/R 计算，三项共用一次抽取与判定记录。两个方向粒度可能不同，不将回答端 TP 强行当作金标端已覆盖数；空分母记 N/A 并说明，不自动记 1。
- 上述新口径命名为 `grounded_factual_v1`，明确是项目自定义指标；保留现有 Ragas 指标及原始版本作为辅助对照，不覆盖旧名称冒充原生指标改善。
- 关键字段验证覆盖数值、单位、主体、年份、条件及矛盾。硬错误不能被平均语义分抵消。
- 独立整题 verdict 使用同一冻结评分说明，但独立阅读实际回答；主裁判异常不能被另一个“通过”静默冲抵，冲突进入复核。
- Ragas AnswerCorrectness、Faithfulness 等存在自己的内部模型步骤；没有逐项校准证据时继续标记诊断性质，不把整题 verdict 校准推广为全部指标已校准。

## 五 文件与接口

以下为拟新增或修改的接口，不是当前已可调用的功能。

| 文件 | 职责 |
| --- | --- |
| 新增 `evals/grounded_facts.py` | SourceSpan/GroundedFact 契约、来源校验、规范化和重复事实处理 |
| 新增 `evals/gold_answer_spec.py` | GoldAnswerSpec 加载、原文与金标绑定、冻结评分依据 |
| 修改 `evals/answer_judge.py` | 原文绑定抽取、事实判定、独立 verdict、裁判版本和指纹 |
| 修改 `evals/answer_fields.py` | 单位/条件等字段原文依据校验，保留盲抽取边界 |
| 修改 `evals/formal_command.py`、`evals/command.py` | sidecar 参数与绑定、真实回放、付费账本及中断恢复 |
| 修改 `evals/review_v2.py` | 金标说明审核、事实级盲标注、裁判校准及报告 |
| 修改 `evals/report_v2.py`、`evals/tuning.py` | 新指标、未知分母、审核门槛、可比较性和参数选择 |
| 新增 `evals/acceptance_v2.py` | 只读验收门槛审计，不写生产配置 |

核心接口：

- `validate_grounded_facts(question: str, answer: str, facts: list[GroundedFact]) -> list[GroundedFact]`：非法来源抛具名校验错误，不返回修好的“正确答案”。
- `load_gold_answer_specs(path: Path, cases: list[GoldCaseV2], snapshot: CorpusSnapshot) -> dict[str, GoldAnswerSpec]`：缺失/重复/错绑定拒绝。
- `FormalRagasJudge.extract_grounded_facts(question: str, answer: str) -> list[GroundedFact]`：不接收金标。
- `FormalRagasJudge.evaluate_grounded(question: str, answer: str, facts: list[GroundedFact], spec: GoldAnswerSpec) -> dict[str, object]`：输出双向逐项判定、计数、P/R/F1、协议版本、sample_id 和错误状态。
- `audit_acceptance(report: dict, gold_audit: dict, calibration_audit: dict, policy: dict) -> dict`：输出各门槛状态与 blockers；缺失或未知均不能通过。

## 六 分阶段实施与验收

各任务按“先写失败测试 → 确认失败 → 最小实现 → 回归 → 更新恢复点”执行。每项完成都记录文件、命令、真实产物路径；本方案不要求自动提交 Git。

### Task 1: 固化真实反例与原文绑定抽取

文件：`evals/grounded_facts.py`、`evals/answer_judge.py`、`evals/answer_fields.py`；测试新增 `tests/unit/evals/test_grounded_facts.py`，扩充 `test_answer_judge_v2.py`。

- [x] 保存 S004 实际回答与错误拆解的最小回归样本；测试 `test_changed_entity_is_judge_error`：星河回答配星尘 quote 必须拒绝，不产生答案错误分。
- [x] 测试 `test_wrong_entity_in_answer_is_not_repaired`：回答本来就是星尘，则允许抽取该原文，由后续判定其与金标不符。
- [x] 覆盖重复 quote、Unicode 偏移、多公司多金额、否定词、跨句指代、漏事实、中文数字和单位换算。
- [x] 实现来源契约、确定性规范化和盲抽取，保留失败原响应/原因，不补金标答案。
- [x] 运行 `PYTHONPATH=src:. /Users/steven/miniconda3/envs/agentic-rag/bin/python -m pytest tests/unit/evals/test_grounded_facts.py tests/unit/evals/test_answer_judge_v2.py -q`；新增反例全部通过（46 passed）。单测不当真实评测成绩。

### Task 2: 冻结完整评分依据和双向事实分数

文件：`evals/gold_answer_spec.py`、`evals/answer_judge.py`；测试新增 `tests/unit/evals/test_gold_answer_spec.py`，扩充 `test_answer_judge_v2.py`。

- [x] 测试缺来源、错 case hash、按回答补 reference 均拒绝；S004 的公司/年份/城市绑定原金标及原文来源。
- [x] 测试必需事实缺失、可选说明、条件错误、矛盾、不同抽取粒度和重复事实。例：2 条回答事实被支持、3 条必需事实中覆盖 2 条，必须 P=1、R=2/3、F1=0.8。
- [x] 实现 GoldAnswerSpec 与独立 `grounded_factual_v1`，保留原 Ragas 诊断结果；所有规则、schema、提示词和 spec hash 进入新指纹。当前sidecar只接受从冻结金标生成的待人工审核草案，粒度与必答范围不冒充已审核。
- [x] 正确字段全部通过也不能掩盖漏答必需事实；抽取错误也不能被整题 verdict 的“通过”覆盖。
- [x] 运行对应新增测试与 `test_answer_judge_v2.py`；分数、失败类型和绑定符合断言。

### Task 3: 报告和零重复查询回放

文件：`evals/formal_command.py`、`evals/command.py`、`evals/report_v2.py`；测试 `test_formal_eval_command.py`、`test_report_v2.py`。

- [x] 新增 `--answer-spec` 参数，单独冻结 sidecar hash；同一实验恢复时任一输入变化必须拒绝。
- [x] 分别展示 system_failure、answer_incorrect、judge_error、gold_issue；有分样本均值、评分覆盖率和全量保守通过率分别展示。
- [x] 主评分 unknown 保留原样；保守通过率按“未证实成功”处理，不把它描述为已判系统答错。
- [x] 未人工审核的草案允许受限诊断回放，但 `eligible_for_tuning=false`；不得自动进入选参。缺字段/未知同样拒绝。
- [x] 新目录只读回放现有 20 个 Run，不改索引、不重复 Query POST。单题pilot与完整20题均已执行并保留原样；20题主评分仍有6个unknown，`scoring_complete=false`，不能当验收通过。
- [x] 验证同配置 `--resume` 新 Query POST=0、新 Judge 请求=0、Run ID 不变、账本 hash 不变。见`formal-v2-grounded-smoke-20261001/audit-resume.json`；首次305个Judge/Embedding请求单独计费。
- [x] 新增中断、未知付费响应、丢失 ledger、错 sidecar 指纹的失败测试；不得自动补发。

第一批v5.1工程交付时，6个真实回答因LLM对重复短词给错字符下标而被拒绝，只有3/9有正文答案获得主事实分数；历史失败保留。用户随后要求继续修复，v6.2已改为原句/分句编号与逐字quote选择，并修复冗余单位引用误拒、字段失败报告缺项。新目录20题回放中9条实际回答均获主评分、6条适用字段评分齐备，`scoring_complete=true`；11条系统失败与2条漏答必需事实仍保留。工程修复完成不代表质量门槛通过，人工审核、校准及全量验收任务仍未启动；详细恢复与审计结果见`docs/formal-rag-evaluation-v2-progress.md`末尾。

### 任务 4 人工金标审核

文件：`evals/review_v2.py`；测试 `test_review_v2.py`；产物：人工审核队列、审核报告、评分说明清单和差异记录。

- [ ] 队列展示原文位置、问题、参考答案、必需/可选事实、适用条件、Child AND/OR 证据集；金标审核不展示系统得分诱导修改。
- [ ] 全 4453 题执行来源和结构检查；完成既有 213 题风险分层人工审核，发现模板问题时扩查同模板全部题目。
- [ ] 213 题通过只标记“审核样本通过”。未逐题人工核验的其余题目保留原状态和总体抽样限制。
- [ ] 人工 graded qrels 未完成则 graded NDCG 继续 N/A；未标注候选不得默认当无关。已有二元指标继续独立报告。
- [ ] 涉及 test 的金标质量审核由隔离审核流程处理，不向调参流程暴露 test 答案或表现。
- [ ] 拒绝的金标不原地覆盖；发布新版本并记录受影响 case，原版本实验保留。
- [ ] 测试缺 reviewer、日期、理由或错误 hash 均不能通过。字段齐全只能证明格式有效，不能证明真人审阅；验收还需用户确认的审核包 hash 与批准记录，缺少该记录仍阻断，人工待办不由工程代理代签。

### 任务 5 校准裁判而不只校准整题 verdict

文件：`evals/review_v2.py`；测试 `test_review_v2.py`；产物：盲标注清单、分层校准报告、冻结裁判配置。

- [ ] 在 dev 内按公司/问题族划分裁判调整集和独立复核集；validation 不用于改裁判，test 禁止参与。
- [ ] 至少补齐 100 条已人工标注的真实问答，每类至少 20 条，含正确与错误样本。现有 9 条不足；无正文运行失败另做状态审计，不能拿来凑事实评分样本。
- [ ] 对拒答题核验实际拒答正文，不能用 research_round_limit 冒充正确拒答。若真实样本不足，继续收集，不伪造系统答案填额。
- [ ] 标注事实抽取是否保真/完整、字段适用对象、逐事实支持/矛盾/无依据、必需事实覆盖、整题正确性；盲标注阶段隐藏自动判定。
- [ ] 对争议与高风险样本安排第二位人工复核；记录分歧和裁决。受控改公司/金额的负例单独作为鲁棒性测试，不混称真实系统准确率。
- [ ] 报告裁判误放率、误杀率、unknown率、事实级和整题一致率、Kappa、分层样本量及置信区间；不只汇总一个 agreement。
- [ ] 阈值在复核集打分前批准并冻结。100 条仅是启动下限，不自动证明低错误率；若置信区间过宽或关键子类不足，增加样本而不是宣称通过。
- [ ] 测试 test 泄漏、校准集复用、单一标签、样本不足、缺主指标标注时均不得标 calibration_passed。

### 任务 6 候选预算比较和正式验收

文件：`evals/tuning.py`、新增 `evals/acceptance_v2.py`；测试 `test_tuning_v2.py`、新增 `test_acceptance_v2.py`。

- [ ] 将相同 gold/spec/Judge/代码/provider/语料/费用口径/题集和已批准审核状态作为可比较条件；未校准原生 Ragas 分数不得继续担当自动选参的决胜指标。
- [ ] dev 上先比较 RRF 30/50/80，其余保持 40/40→rerank10→Parent6；再按诊断结果逐级比较召回、rerank、Parent 数量，不直接跑全参数笛卡尔积。
- [ ] 使用同一批 dev 题做配对比较；冻结并记录并发条件，合理交错候选顺序。重放旧答案只能验证新裁判，不能代表新的检索参数实验。
- [ ] 候选比较优先主任务成功率/关键事实正确性，再看 P95 和真实用量费率估算；配对公司级重采样报告不确定性。单题 RRF80 不作为最优参数结论。
- [ ] dev 筛出少量方案后，用 890 题 validation 选方案；所有候选使用完整相同题集，失败和未知不得删除。
- [ ] 冻结胜出配置、评分器、金标、评分说明和验收门槛后，对 893 题 test 做一次正式验收。合法中断恢复可以继续同一实验，不重新抽样挑高分。
- [ ] 若 test 后修改系统、金标、裁判或门槛，原结果继续保留，不把反复看过的 test 当全新盲测；后续需独立验收数据。
- [ ] 验收门槛包括质量最低值、关键子类下限、错误拒答/错误放行、P95、预算、评分覆盖和审核状态。质量/SLA具体值须在运行前由用户批准，未批准即 blocker；不凭空当行业标准。
- [ ] 完整报告包含最终答案对金标、分阶段 Recall/MRR/NDCG、实际上下文、多跳完整性、Agent失败原因、成本/时延和限制。TTFT当前仍 unknown，不凭总耗时推算。
- [ ] `audit_acceptance` 只返回是否满足批准的测评门槛及 blockers；即便通过也不自动修改生产参数或部署。

### 任务 7 回归和交接

- [ ] 运行 `PYTHONPATH=src:. /Users/steven/miniconda3/envs/agentic-rag/bin/python -m pytest tests/unit/evals -q`，再运行同解释器 `-m pytest -q`；记录实际结果，不沿用上一轮 959 passed。
- [ ] 执行适用 lint 和 `git diff --check`；文档明确原生指标、自定义指标、校准范围及分母。
- [ ] 更新 `docs/formal-rag-evaluation-v2.md` 的真实命令、产物说明和失败恢复；更新 progress 账本逐项恢复点。
- [ ] 分别交付工程通过、人工审核/校准通过、validation完成、test验收四种状态，不用“全部完成”掩盖人工待办。

## 七 优先顺序和恢复规则

第一批完成任务 1—3：实体误改拦截、完整评分依据、旧 Run 回放。第二批完成任务 4—5：人工审核与校准；金标原文审核准备可以与第一批并行，但任何人工批准不能提前假定。第三批在前两批门槛满足后完成任务 6—7。

每个恢复点记录输入/代码/Judge/spec hash、完成步骤、测试命令、实验目录、Run ID 账本及下一项。新模型调用开始前明确预算；代码或绑定变更不能恢复到旧评分目录。下次“继续 rag 测评优化”时先读 progress 末尾，再看本方案经用户批准的阶段。

本方案的完成目标是让评分可信且验收可执行，不预先承诺 RAG 分数上涨，也不承诺已有系统一定通过生产门槛。

## 依据

- 当前源码：`evals/answer_judge.py`、`evals/answer_fields.py`、`evals/review_v2.py`、`evals/report_v2.py`、`evals/formal_command.py`、`evals/tuning.py`。
- 当前实测：`var/artifacts/evals/formal-v2-smoke-20261001-indexed-factual/`，尤其 S004 lodging 的原回答与 factual evidence。
- [Ragas Factual Correctness 官方说明](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/)：原生指标先拆解声明，再进行自然语言推断。本文的保真约束、自定义口径和验收流程为项目修改建议，并非官方默认实现。
