# 正式 RAG 测评 v2：可恢复实施账本

## 当前任务：100题系统端到端基线（2026-10-05）

- 用户要求重新测评整个 RAG 系统，不限于 IK 变化；已明确选择五类均衡100题。使用正式 API8000 / Outbox / Query Worker / QueryGraph；保持生产默认参数，不做 analyzer 双臂对照。
- 输入沿用 `var/artifacts/evals/formal-v4-rebound-20261003/{snapshot.json,gold.v2.2.jsonl,answer-spec.jsonl}`。确定性 `--split dev --smoke 100` 已核对为 single_hop、section_comprehension、table_calculation、unanswerable、multi_hop 各20题；validation/test不使用。
- 本轮只读预检已通过：1003份原文、4453题来源锚点，以及正式SQL/ES完整快照 matched；live fingerprint=`d8347a7b2e5adf522d413160d40b427f1bea901d63d7d985928b0d2ef80ea739`。正式 API 配置与本地匹配，base snapshot=`be482e1136095769a7f6d451b12cc421591ae2814776e19d634ee5d74c7973d4`，evaluation_requests_enabled=true；正式 Query Worker PID73998存在。只读预检没有调用模型或新建查询。
- 新目录：`var/artifacts/evals/system-rag-baseline-100-20261005`。先以完整100题身份和 `--max-wall-seconds 1` 执行首题链路探针；墙钟预算在题间检查，所以首题允许完成，其余保留pending。核实API/Worker/证据采集/评分正常后，在同一目录 `--resume` 继续，不重提首题。
- 首阶段命令：`./scripts/eval_rag.sh --dataset var/artifacts/evals/formal-v4-rebound-20261003/gold.v2.2.jsonl --corpus-snapshot var/artifacts/evals/formal-v4-rebound-20261003/snapshot.json --answer-spec var/artifacts/evals/formal-v4-rebound-20261003/answer-spec.jsonl --api-url http://127.0.0.1:8000 --split dev --smoke 100 --max-cases 100 --max-judge-requests 5000 --max-wall-seconds 1 --output-dir var/artifacts/evals/system-rag-baseline-100-20261005`。
- 拟续跑命令：`./scripts/eval_rag.sh --resume var/artifacts/evals/system-rag-baseline-100-20261005 --max-cases 100 --max-judge-requests 5000 --max-wall-seconds 28800`。初始链路尚待真实完成，不能把预检通过当作100题成绩。
- 评分包括最终答案对金标、独立来源约束事实评分、关键字段校验、Ragas、分阶段检索排名与证据完整性；失败/超时/评分未知保留分母。人工金标审核和Judge校准仍pending，graded NDCG缺标注不编造；此基线不等于上线验收，也不覆盖聊天、长期记忆或故障恢复测试。memory_policy=disabled，独立评测thread，不污染用户记忆。
- 旧IK对照目录保留原状：截至本次检查仅21/40有记录，另1条query账本为running、对应评测进程已不在；不得将其当全量完成，不自动重发未知查询。本轮使用独立新实验，不混入旧结果。
- [x] 全量输入、环境与API配置只读预检。
- [x] 确认100题五类各20题，以及生产Worker存在。
- [ ] 首题真实API链路与评分核验。
- [ ] 同目录继续剩余99题，保存逐题进度与失败。
- [ ] 全量后复核语料快照、评分覆盖率、零新增付费请求恢复。
- [ ] 汇总总体及五类成绩、系统瓶颈与未覆盖范围。

用户已授权实施：2026-09-30。方案：`docs/superpowers/plans/2026-09-30-formal-corpus-evaluation-v2.md`。

## 固定输入与边界

- 已有1003份正式文档、2237 Child，不生成/上传文档，不重建/删除索引。
- 输入：`var/artifacts/evals/formal-corpus-gold-v2-20260930-production/{snapshot.json,gold.v2.2.jsonl}`。
- 金标4453题；12105事实组中1组因真实PDF解析错误 unmapped，保留端到端分母。
- 人工语义审核和Judge阈值校准未完成，不能自动宣称生产发布通过。
- 当前 `codex/real-rag-evaluation` 有大量既有未提交实现；就地增量修改，保留所有旧成果，不另建缺少这些成果的工作树。
- 基线：全仓859 passed、47 skipped（29.49s）。这是代码测试，不是RAG质量成绩。

## 执行顺序

- [ ] 1. 严格金标契约、来源/版本/计算校验、人工审核入口。
- [x] 2. 正式v2命令、冻结快照预检、隔离记忆、可恢复查询账本（完整恢复实测待本轮结束）。
- [x] 3. 原始Child排名、实际上下文、最终输出与Agent轨迹；TTFT未持久化，仍明确unknown。
- [x] 4. 分阶段/分粒度排名、AND/OR事实覆盖与多跳完整性；人工graded qrels尚未标注。
- [x] 5. 盲抽取关键字段、真实Answer/Factual Correctness及ContextRecall接入；真实评分覆盖率待本轮确认。
- [x] 6. 失败不剔除、评分覆盖率、Agent诊断、分组与报告。
- [x] 7. 可配置候选预算、dev/validation/test隔离调参与Pareto选择工具；尚未完成参数实验，未发布参数。
- [x] 8. 全仓回归、真实分层smoke、独立审查与恢复说明；生产质量验收仍待人工审核/校准和validation/test。

## 接口裁决

- 新v2模型/结果与旧24题评测并存，不将旧Parent二元口径冒充新Child口径。
- 冻结gold.v2.2文件不覆盖；缺少人工graded qrels时，graded NDCG为null，不凭自动匹配编造等级。
- 关键字段抽取不接收金标值；规则验证与LLM事实评分独立。
- 正式快照完整性校验覆盖全部干扰文档，不只核对命中的正例。
- 未知POST/裁判响应不能自动重复收费调用；恢复必须复用run_id或明确对账。

## 早期恢复点（后续已完成工程实现，见本文末尾）

已实现并针对反例验证：严格v2加载（完整4453题通过）、来源/用户/版本/hash/公式校验；Child AND/OR与多跳覆盖、graded未知口径；真实Checkpoint投影；测评记忆读写隔离；盲抽取字段和真实Ragas扩展；候选预算贯穿与可配置reranker上限。CLI/持久化/报告/真实运行仍在实施，不能把单元测试当真实成绩。

新增回归发现并修复：Subagent原来继承进程固定snapshot，未继承当前Run的测评配置；现继承当前父Run全部受控配置。测评Event使用当前Run snapshot且不放宽其它快照不匹配保护。

## 历史真实运行（已停止，不能当完整成绩）

- 只读预检：1003原文/4453金标的原文hash、原文锚点和原Excel输入验证通过；SQL/ES全量快照 matched，live fingerprint `74a1cca873a67403306b2c9c93db37da237110c23de345f6d99b819a45bc6565`。
- 历史API/正式Query Worker进程已退出；2026-10-01检查正式8000未监听。测试API8001/Worker仍在，不能误停。新部署指纹覆盖provider配置和源码，待代码完成后启动正式API/Worker。
- 本地配置启用`AGENTIC_RAG_ALLOW_EVALUATION_REQUESTS=true`，默认模板false；评测仅接受eval-v2隔离thread、固定用户、memory_policy=disabled及校验预算，普通请求不变。
- 原评测独立venv补齐pypdf6.10.0（与金标构建解释器一致），不升级业务Conda；pyproject的eval extra固定pypdf/openpyxl。
- 历史20题输出`var/artifacts/evals/formal-v2-smoke-20260930`，在第8个查询轮询时已主动停止；不是完整成绩。主要发现HTTP/checkpoint投影误判及DeepSeek思考消耗导致Judge输出截断。已修复投影；官方DeepSeek裁判禁用thinking并给8192输出上限。
- 新代码/裁判/部署指纹禁止直接恢复混用历史成绩。`--replay-from`可在新目录只重收集历史run_id、重新评分，禁止补发源实验未提交题目；旧账本完整保留。未知POST/评分操作不能自动重复收费。
- 独立只读审查已完成，发现的问题正在收尾；不重复发起审查。人工金标语义审核/裁判校准/完整validation和held-out test仍待完成。

## 2026-10-01 实施恢复点

- 已补齐缓存核验：恢复核对API、SQL、checkpoint、实际上下文和独立Judge记录；缓存结果缺查询账本时禁止重新POST。历史失败/未知评分保留，不能静默重试收费。
- 已补齐真实SDK逐请求用量、缓存token及冻结费率估算；查询费用与Judge费用分开，未知用量/模型单价不报0。固定峰时费率仅用于同口径比较，不是实际账单。
- 已补齐provider/代码配置绑定、alias过滤定义校验、阶段失败/降级标注、实际上下文排名、全类别宏平均和pending分母。
- 最近针对性回归276 passed；仍需最终全仓测试、新部署只读预检、真实20题完整成绩及零新增收费恢复验证。
- 待收尾：可执行调参入口、人工审核/裁判校准交接、文档与实际报告。任何单元测试通过均不能替代RAG质量成绩或人工验收。

## 2026-10-01 新基线真实执行

- 全仓最终回归：951 passed、47 skipped；相关ruff和git diff --check通过。新增invocation_id使两项旧精确字典断言不匹配，已更新断言同时保留唯一ID检查；没有删测试。
- 正式API工具session40490（PID33224）、Query Worker session38048；测试8001不变。只读预检再次验证1003原文/4453金标与全量SQL/ES；指纹仍为`74a1cca873a67403306b2c9c93db37da237110c23de345f6d99b819a45bc6565`。
- 首轮20题：`var/artifacts/evals/formal-v2-smoke-20261001`，原工具session98883已停止；五类各4题、15公司。8题已评分，9个Run已提交，157个Judge/Embedding请求全部已完成，SIGINT发生在第9题轮询时，不在裁判调用期间。旧结果不可直接混用新版评分。
- 代码/配置指纹冻结后不要在运行中改evals或业务源码。完成后用同配置`--resume`验证零新增查询/裁判请求；如需修评分器，用新目录replay，不覆盖旧结果。
- 人工审核入口已实现并真实导出213题：`var/artifacts/evals/formal-v2-review-20261001/gold-review.jsonl`，audit确认0题已人工通过；含所有多跳、拒答、历史题及其它每类20题。不能自动代签。
- 调参CLI已执行生成RRF30/50/80预算：`var/artifacts/evals/formal-v2-tuning-20261001/rrf/`，尚未发起这些候选实验。默认参数文件`evals/budgets/baseline.json`，人工校准与比较命令见`docs/formal-rag-evaluation-v2.md`。

## 共享事实评分修正与当前接续点

- 首轮真实多跳题发现：Ragas三个Factual模式分别拆解/判定，出现precision=1、recall=.75、F1=1。三者来源是不同LLM样本，不能冒充同一混淆矩阵。
- 已改用Ragas0.4.2同一事实拆解/双向NLI样本计算TP/FP/FN及未四舍五入P/R/F1；保存claims、reason、counts、sample_id。漏判/错位unknown；中途保存不完整组禁止重复付费采样。裁判版本`formal-answer-v3-ragas042-shared-factual-fields-v1`。
- `--continue-from`已实现：仅评分器变化可创建新实验，复用已提交Run并补发未提交题；查询部署/provider/session/预算必须完全一致。与只读旧Run的`--replay-from`不同，不覆盖源实验，不自动混用旧裁判结果。
- 当前运行：`var/artifacts/evals/formal-v2-smoke-20261001-shared-factual`，工具session74764；命令`./scripts/eval_rag.sh --continue-from var/artifacts/evals/formal-v2-smoke-20261001 --output-dir var/artifacts/evals/formal-v2-smoke-20261001-shared-factual --max-cases 20 --max-judge-requests 1000 --max-wall-seconds 3600`。正式API/Worker未改，原9个Run不重复执行。
- 新增回归259项eval单测通过；全仓最终回归955 passed、47 skipped（27.30秒）。
- 实测旧历史Run回放已完成：`formal-v2-replay-check-20261001`，复用2026-09-30的Run，没有新查询或裁判请求；旧Run是clarify，不能算成功答题。
- 已验证首轮一条已完成回答在require_cached复算时结果完全一致，新增Judge请求0；完整20题的最终恢复验证仍待完成。
- 基线发现（不是参数优化结论）：S019跨年题BM25多轮覆盖全部事实，但RRF前30全部丢失；部分cannot_answer只有状态没有正文；这些均留在总分母，不能替系统补写回答。
- 共享版本已真实观察9/20题（01:32 UTC时点），其中S004 lodging的NLI声明校验失败，P/R/F1保留null，不删除、不自动重试。其它已评分多跳题三项分数共享同一sample_id/计数。
- 单独最多4请求的诊断保存在`formal-v2-factual-alignment-check-20261001`，诊断再次采样成功不代表原失败消失，未用于替换基线评分；观察到事实拆解可能改写公司名称，更说明人工校准不可省略。不能把这次诊断当生产成绩。

## 编号Schema事实裁判（当前恢复点，2026-10-01）

- v3共享版本20/20查询已完整采集、末尾快照matched；9个completed回答、8个research_round_limit、3个cannot_answer。S004 lodging、S015 lodging、S020 incident section-3共3题NLI声明校验失败，旧成绩保留，退出码1代表评分不完整。
- v4改用明确编号的Structured Output适配NLI，继续使用Ragas0.4.2事实拆解/计数口径。模型只输出claim_id/supported/reason，Python核对编号全集、拒绝重复/越界/漏判并保留原声明，不依赖模型逐字回显。rubric和schema独立指纹，版本`formal-answer-v4-ragas042-indexed-factual-fields-v1`。
- 4个新反例测试先失败后实现，263项eval单测通过；最终全仓工具session10505待结果。
- 当前重新评分工具session2759：`formal-v2-smoke-20261001-indexed-factual`，`--continue-from formal-v2-smoke-20261001-shared-factual`，全部复用已存在20个Run，不新增这些题的RAG查询。20题/1000裁判请求/1800秒预算。等待完成，再用该目录--resume证明零新增查询和裁判请求。
- 独立1题RRF80连通性试验工具session27588：`formal-v2-rrf80-pilot-20261001`，S019跨年题、预算40/40→80→10→6，最多100裁判请求。仅本Run预算变更，不改正式默认参数；与重新评分并行，不据此单题比较P95或认定最优参数。
- 之后导出v4真实回答的盲校准清单，更新报告和恢复说明。213题人工审核、graded qrels、100例校准/阈值、完整validation/test仍未完成，不可自动代签或宣称生产发布通过。

## 本轮最终核验结果（2026-10-01；下次从这里继续）

- v4基线`var/artifacts/evals/formal-v2-smoke-20261001-indexed-factual/report.{md,json}`：20/20真实Run采集完成，scoring_complete=true、snapshot_verified=true；scoring_failures为空。9条生成回答，8条研究轮限，3条cannot_answer无正文。
- 独立整题判定9/20（45%）；AnswerCorrectness 0.7796748383、Factual F1 0.5247863248，均为9条有正文样本的条件均值，不是20题总体正确率。关键字段6条适用回答均通过；其余题目不能冒充字段正确。
- 重要评分风险：S004的实际回答公司名是“虚构星河04公司”，Ragas声明拆解却改成“虚构星尘04公司”，且简短reference未展开公司/年份；因此Factual出现误判风险。v4解决的是输出对齐和共享计数，不是宣称LLM判定已校准。人工审核和拆解/参考答案口径校准仍为发布阻断项，不能直接用当前Factual分数选生产参数。
- Query API用量全部可核对；固定峰时估算查询12.3882419元、v4裁判0.29460288元（不代表账单，不含历史试错和基础设施）。查询P50=117.061856秒，P95=218.119724秒，TTFT仍unknown。
- 完整--resume实测成功：`resume-verification-query-posts.json`确认新建Query API POST=0，293条裁判请求账本内容/hash不变，20个run_id不变且全部复用v3源Run。初始统计把8个ES只读_search/scroll POST误算成查询，已保留为`resume-verification-all-posts-diagnostic.json`，不能误读为重复检索。
- RRF80独立单题`formal-v2-rrf80-pilot-20261001`完成，run_id=`01a0f52c-932e-7db4-9e5b-cb015c055171`；实际RRF输出53/70/70/70/58/60条，rerank输出10；两年证据和最终上下文覆盖率1，回答590→650、增加60正确。基线同题研究轮限、RRF覆盖0。仅1题且Agent检索轨迹不同，不能外推总体增益或认定最优参数；正式默认40/40→30→10→6未修改。
- 最新完整代码回归959 passed、47 skipped（31.09秒），ruff与git diff --check通过。未额外发起独立复审；原审查建议已逐项处理并回归。
- 已导出`formal-v2-review-20261001/gold-review.jsonl`（213题，人工通过0）和`calibration-v4.jsonl`（9条实际回答，人工通过0）；对应pending audit报告不授予通过。当前calibration-audit只比较独立整题verdict与人工标签，不代表Ragas每个子指标已校准。
- 正式API session40490/PID33224、Query Worker session38048/PID33215保持运行；测试8001未改。全部测评进程已结束，原文、分块和正式ES快照保持不变。

### 尚未完成及下一步顺序

1. 先校准事实拆解保真、完整reference语义口径及Judge误判；保留现有失败/低分证据，不能反复抽样覆盖到满意分数。生成新评分版本时使用新实验目录复用既有Run，不重做入库。
2. 完成213题原文/金标人工审核、人工graded qrels，以及至少100例（各类≥20）盲校准与门槛确认。不能用AI代填human approval；目前9个正文样本不足。
3. 扩充自然问法、困难负例和缺失类别的校准样本；维持公司/问法族split隔离，不修改冻结旧金标。
4. 校准通过后对相同dev集逐级比较候选预算，再跑890题validation选方案、893题held-out test验收。当前无全量4453题成绩，无生产发布结论；单题RRF80不是最优参数结论。
5. TTFT尚未持久化；Todo违规只检查可见checkpoint，target组重叠是Subagent重复工作的近似诊断，不应冒充完整分布式追踪。

## 2026-10-01 后续修改方案草案

- 应用户“生成一个修改方案”的请求，新增 `docs/superpowers/plans/2026-10-01-judge-fidelity-and-production-eval.md`，覆盖事实保真、冻结评分说明、旧Run回放、人工审核/校准、890题validation及893题test验收。
- 本次只编写方案和本索引，未修改评分器、启动真实测评、填写人工批准或改变索引/生产配置。方案待用户审阅，不能将其复选框当作已完成状态。
- 后续执行仍从上方真实核验结果恢复；先确认用户批准的方案阶段，不因本条新增记录自动发起付费测评或全量实验。

## 2026-10-01 第一批实施中

- 用户已批准第一批并要求在real-rag-evaluation分支修改；从main安全切到已有 `codex/real-rag-evaluation`，两分支起点树相同，保留全部未提交修改，无提交/推送。
- 新增原文绑定事实抽取、来源校验、独立GoldAnswerSpec草案、自定义grounded P/R/F1、失败分类、恢复账本接入。`--answer-spec`选择新版主评分，旧Ragas诊断不改名冒充新算法。
- 当前针对性评测测试309通过；完整回归1061通过/53跳过（29.46秒），基线1024通过/53跳过。尚在独立只读审查和真实回放阶段，不能宣称第一批完成。
- 生成4453条待审评分说明：`var/artifacts/evals/formal-v2-grounded-spec-20261001/answer-spec.jsonl`。说明hash `2c474a080257f633ff2eaf46c32ec549376f527d5918757dd0cb3edce15d2749`，不含人工批准。完整原文/金标/SQL/ES预检matched，语料指纹仍为`74a1cca873a67403306b2c9c93db37da237110c23de345f6d99b819a45bc6565`。
- 正式API当前部署与本地业务源码指纹一致为`56853c63afafba9c03aab84ad42e5ca5b4783256f8d65a26facd130628cca353`；只读回放沿用旧Run的历史查询快照，不重启/不重查。待回放目录 `formal-v2-grounded-pilot-20261001`，单题S004，60裁判请求/600秒预算；之后完整20题及零新增调用恢复验证。
- 本轮决策和测试细节记在 `.superpowers/sdd/2026-10-01-judge-fidelity-and-production-eval/progress.md`。第二批人工审核/校准与validation/test未启动。

## 原文绑定评分第一批交付与真实限制（2026-10-01；最新恢复点）

- 分支：`codex/real-rag-evaluation`（上游`origin/real-rag-evaluation`）。保留既有改动，未提交/推送；没有改正式解析/分块/索引或启动新RAG查询。第一批Tasks 1–3实现与回放验证已执行，不代表评分质量验收通过。
- 新裁判版本：`formal-answer-v5.1-grounded-factual-v1`。新增原文绑定事实/字段校验、独立评分说明、同一次判定的主事实P/R/F1、失败证据持久化、分类报告与草案选参阻断。旧Ragas仍为诊断，不冒充已校准主指标。
- 独立审查的3项问题已修复并先失败再通过：金额不能丢弃万/亿倍率或借用别处货币；恢复即使零模型调用也核验付费账本；缺失/未知选参资格默认拒绝。账本验证还覆盖负数token、缓存token超总量、成功评分缺对应付费操作、累计请求预算。
- 最新全仓验证：**1084 passed、53 skipped、19 warnings，29.04秒**；相关ruff及`git diff --check`通过。单测用于验证工程契约，不是质量测评分数。
- 待审评分说明4453条，路径`var/artifacts/evals/formal-v2-grounded-spec-20261001/answer-spec.jsonl`，SHA256=`2c474a080257f633ff2eaf46c32ec549376f527d5918757dd0cb3edce15d2749`。原gold哈希仍为`7066e120bc11b836d2e99d9522e5420cf8f3b77e949e9d1965a75edb5cf2bfb0`，未根据系统答案调整必需事实。

### 真实回放结果

- 旧单题pilot `formal-v2-grounded-pilot-20261001`保留失败原样：重复引用位置出错，38次真实Judge/Embedding请求，冻结费率估算0.0465696元。之后只对全局或已验证父证据内唯一的完全相同quote确定性修正位置，不模糊改写。
- 新20题目录：`var/artifacts/evals/formal-v2-grounded-smoke-20261001/`。全部20个历史Run采集完成，末尾1003原文/4453金标及SQL/ES快照仍matched，live fingerprint=`74a1cca873a67403306b2c9c93db37da237110c23de345f6d99b819a45bc6565`。
- **11个系统失败（8轮限+3空正文）、1个漏答、2个暂判通过、6个judge_error/unknown**。保守通过率2/20=10%，不等于系统真实准确率；6个未知不被描述为答错。`scoring_complete=false`、`eligible_for_tuning=false`、`production_release_accepted=false`，退出码1正确表示评分未齐备。
- 主事实指标仅3/9条有正文答案、3/20总题有有效评分：P=1、R=0.8333333333、F1=0.8888888889。这是极低覆盖下的条件均值，不可据此宣称提升。旧Ragas AnswerCorrectness=0.7699345785（9/20），仅供诊断。
- S004明确区分两类问题：原始公司“虚构星河04公司”未被改成“星尘”；金额/单位通过，但答案未表达冻结金标中的“含税”，故P=1、R=0.5、F1=2/3、整题不通过。草案是否必须显式说含税仍待人工提前审核，未因结果不好删除要求。
- 新20题Judge/Embedding账本305次请求，冻结费率估算0.4162028元；加pilot本轮合计343次、0.4627724元，不是实际账单。报告中历史Query费用12.3882419元是复用Run的原成本，不是本轮再次检索的开销。
- 同代码完整`--resume`实测：新Query POST=0、新Judge SDK调用=0、新账本请求=0；20个Run ID、query ledger及Judge ledger hash均不变。证据为`audit-resume.json`。初次`audit-first.json`的HTTP层Judge计数未覆盖SDK传输，不可将其中0误读为没有收费；真实305次以持久化SDK账本为准。恢复验证已同时拦截SDK边界和核对账本。

### 下一步优先处理（不要直接重跑到高分）

1. 降低原文定位的unknown：S007/S030跨年、S015/S019住宿、S020/S029事件共6题均被`source_quote_missing_or_ambiguous`拦截。离线复核确认涉及回答中重复“元”“目标”“不能”等短引用，LLM字符下标错误且其父证据范围仍过宽；S007/S030的字段提取也失败。原始模型输出全部在对应`judges/<case>/grounded_extract.json`或`fields.json`，不必收费重新采样才能排查。
2. 下一版应优先让程序生成可选择的原文位置/候选编号，减少LLM数Unicode字符；保留歧义拒绝，不能用最近位置猜测或金标补全。若改评分器，创建新版本/新目录，现v5.1成绩和账本保留，不在本目录重新付费补分。
3. 第二批人工金标说明审核、事实级裁判校准/阈值、graded qrels、890题validation与893题test仍未执行。当前自动草案不能代签人工审核，也不能用这3条有效主评分选生产参数。

当前回放及恢复进程均已结束；API/Worker未重启，测试8001/ES9201未改动。

## 原文位置选择修复 v6.1（历史执行记录；最终结果见下节）

- 用户要求“继续修复”，沿用`codex/real-rag-evaluation`，只处理事实/字段抽取定位，不改金标、主评分计数、生产检索或正式索引。
- 新模块`evals/source_selection.py`：模型选择整句证据编号、分句编号和逐字quote，程序计算位置并继续现有来源/角色/覆盖校验。错误实体、问题补答案、句外借数值、编号越界、重复或歧义引用保持拒绝。
- 字段单位/条件依据在新schema中必填；原S007/S030空unit_evidence仍作为旧失败保留，没有静默补正旧结果。
- 初始v6真实定位探针`var/artifacts/evals/formal-v2-numbered-extraction-probe-20261001`：10次真实裁判请求，5/6事实抽取、4/4字段抽取通过，费率估算0.06030152元，0个检索请求；这不是系统质量评分。S007剩余失败为跨分句共享主体超出仅选分句证据范围。
- v6.1把证据选择改为显式原始整句编号，角色仍在分句内严格定位，保留原有完整句证据范围而不猜主体关系。新增反例先失败后通过；独立只读审查无Critical/Important发现，语义校准与正式验收仍不在本轮范围。
- 最新完整回归**1138 passed、53 skipped、19 warnings（32.03秒）**；评测单测**348 passed**，相关ruff/diff-check通过。全仓数目包含工作区其他既有测试，不将所有新增测试数量归因于本修复。
- 已启动新目录20题只读回放`var/artifacts/evals/formal-v2-numbered-smoke-20261001`，最大600次裁判请求/1800秒；评分代码冻结，完成后再做零新增调用resume。旧v5.1/探针产物不覆盖；人工审核、890题validation和893题test未启动。

## 原文定位与字段校验修复 v6.2（2026-10-01；最新恢复点）

- 分支保持`codex/real-rag-evaluation`，HEAD仍为`144ff69e2738dc2de4a327b77745fd5bf9b4e002`，未提交/推送；既有与并行工作区修改保留。未重启API/Worker，未改正式ES、原文、解析或分块。
- v6.1完整回放结果：6个旧定位错误全部消除，9条有正文答案获得主事实评分；但S004/S024的字段抽取同时引用完整本句单位依据和下一句冗余说明，被旧规则误拒，`scoring_complete=false`。311次真实Judge/Embedding请求，冻结费率估算0.41983366元。该版本的零调用恢复已验证，失败和费用均保留。
- v6.2修复：所有引用继续逐字核验、保留审计，只让数值所在原句和允许的问题计数口径参与单位证明；冗余句外引用不会误拒，也不能补单位。新增反例同时阻止相同金额的其他句子提供计数口径。原S004/S024保存的失败响应经离线校验通过，无需为定位问题重新采样。
- 修复字段失败原因透传和报告缺项：`fields.reason`与`scoring_failures.critical_fields`现在可明确追踪失败。相关测试先失败后通过；同一只读审查者复核，无Critical/Important发现。语义关系、否定范围和事实完整性仍需人工校准。
- 最新完整回归**1143 passed、53 skipped、19 warnings（31.95秒）**；评测单测**353 passed**，相关ruff和diff-check通过。全仓数目包含其他工作区测试，不将全部新增数目归因于本修复。

### 最终真实回放

- 最新目录：`var/artifacts/evals/formal-v2-numbered-local-units-smoke-20261001/`，版本`formal-answer-v6.2-numbered-source-factual-v1`。20/20历史Run全部采集，9条实际回答主评分可用、6条适用回答关键字段全部通过，`scoring_complete=true`、`scoring_failures={}`。
- 当前自动整题判定**7/20（35%）**；11条系统失败仍在分母（8条研究轮限、3条cannot_answer无正文）。S004和S019住宿题漏答冻结评分说明要求的“含税”，仍判`answer_incorrect`，没有因金额正确而掩盖漏答。
- 主事实Precision=1、Recall=0.8888888889、F1=0.9259259259，AnswerCorrectness=0.7796748383；这些都是**9条有正文答案的条件均值**，不是20题总体正确率。该回放只验证新裁判，不能说检索系统性能得到提升。
- 本次回放311次真实Judge/Embedding请求，费率估算**0.39526352元**；本轮继续修复的探针、v6.1与v6.2共632次请求，累计**0.87539870元**。报告中的查询12.3882419元来自复用历史Run，不是本轮新增查询费用；所有估算均不代表账单。
- 运行期间代码指纹与冻结值一致：`a9b1bdc706ae0e6c8c13f1b54e647d6c72eef01e08dff84af718faaefe809573`。金标与评分说明hash不变，首轮审计新Query POST=0，末尾全语料快照核验通过。
- 完整resume退出码0；`audit-resume.json`确认新Query POST=0、Judge SDK调用=0、付费账本新增请求=0，20个Run ID、查询账本和裁判账本hash均不变，并与源v6.1 Run完全一致。金标/说明/实现hash再次核对通过，`snapshot_verified=true`。回放及恢复进程均已结束。
- 本轮编号定位与字段误拒修复完成。后续人工审核、至少100例事实级校准、graded qrels、890题validation和893题test未启动，`eligible_for_tuning=false`、`production_release_accepted=false`，不能当生产验收结论。下一步从已批准方案的任务4/5准备审核与校准，需真人审核和门槛确认，不能由AI代签，也不自动启动全量付费实验。

## 2026-10-03 主分支 IK 适配设计恢复点

- 用户要求根据主分支变化重梳金标和测评流程，已确认“保留事实、重绑v4、分层验证”的书面设计。本次已保存详细实施计划，尚未实现或执行新测评。
- 当前核对的 main 为`60d543c7c8d181fffce11f8cc5572d5277a0d0fc`，IK 已正式提交；此前“IK 未提交”的状态已失效。本次不切换分支、不提交/推送，既有`tmp/`保留。
- 设计：`docs/superpowers/specs/2026-10-03-ik-gold-retrieval-evaluation-design.md`。范围为v4等价金标重绑定、真实分词环境指纹、同一v4索引standard与IK混合BM25对照。
- 计划：`docs/superpowers/plans/2026-10-03-ik-gold-retrieval-evaluation.md`。六个任务依次为迁移证明、金标发布、分词环境、真实配对执行/恢复、指标报告、命令与真实验收；待审阅及确认执行方式。
- 当前只读分支检查：`codex/real-rag-evaluation` 是 main 祖先，落后 14 个提交、无独有提交；实施前再核查，在隔离评测工作区仅快进到上述批准基线，不在本轮切分支或修改 Git 状态。
- 第一轮拟使用dev五类各20题，只做真实ES读取/分词/BM25，不提交Query、不调用Embedding/生成模型/Judge。Dense/RRF/rerank/最终上下文与真实问答另设有预算的阶段，未执行阶段不能伪造分数。
- 4453题事实、答案、split、审核状态和既有unmapped事实保持不变；更新snapshot及sidecar绑定前必须证明全部原文与分块内容等价。新产物独立目录，旧v3快照和旧20题评分不能直接恢复成IK实验。
- [x] 核对主分支变化、运行索引与旧金标结构，确认拆分设计方向。
- [x] 保存第一批设计及验收条件。
- [x] 用户审阅并确认书面设计（2026-10-03）。
- [x] 编写详细实施计划、具名失败测试和评测分支同步步骤。
- [ ] 用户审阅计划及首轮只读预算，选择 Native 或 Subagent-driven 执行方式。
- [ ] 按批准的计划实现、测试、真实只读预检和限额A/B，记录实际结果。

下次从本小节和设计文档恢复。人工审核、事实级校准、自然问法扩充、完整validation/test、聊天及工具场景仍是后续待办，不能标记完成。本次没有修改业务代码、数据库、索引、金标文件或服务。

## 2026-10-03 Native 实施恢复入口（最新，以隔离工作区账本为准）

- 用户已批准计划和 Native 执行。评测代码位于 `/Users/steven/.codex/worktrees/ik-rag-evaluation/AgenticRAG`，分支 `codex/real-rag-evaluation`，批准基线 `60d543c7c8d181fffce11f8cc5572d5277a0d0fc`；新增代码未提交/推送，main 不承载这些实现。
- 任务1–5已实现；任务6的 `python -m evals.ik_evaluation_command rebind|bm25|resume` 已实现并经单测。当前全仓非在线回归1435 passed/35 skipped/145 deselected；定向434 passed/1 skipped。最终独立审查进行中，审查修复后的计数以隔离工作区账本为准。
- 已真实生成 `/Users/steven/LzmWorkSpace/AgenticRAG/var/artifacts/evals/formal-v4-rebound-20261003`：1003原文、2231 Parent、2237 Child、4453题、12105事实组（1 unmapped）；dev2670/validation890/test893，旧输入字节hash不变。不要重复生成或覆盖这个目录。
- 真实只读语料预检matched；实际IK进程/制品/词典及15探针核验通过。环境证明在 `var/artifacts/evals/ik-environment-preflight-20261003/environment-host-read.json`；先前沙箱无法读进程的unverified结果也保留。新增BM25/Query/模型调用仍为0。
- 测试ES9202未运行；已向用户询问是否允许启动隔离临时节点，未得到许可前不启动、不执行集成写入。100题/200次真实BM25对照和零请求resume仍未执行，不能把上述工程测试/重绑定当作质量成绩。
- 继续时先读隔离工作区 `.superpowers/sdd/2026-10-03-ik-gold-retrieval-evaluation/progress.md` 和该工作区同名进度文档，完成审查修复，再按用户对临时测试节点的答复执行。不要从旧main记录重做任务1–5。

补充最终状态：独立审查两项Important已完成9个RED→GREEN反例修复（历史失败核验不可被resume洗白、删除的已加载词典不可漏检）。全仓非在线回归1444 passed/35 skipped/145 deselected；真实环境修复后前后再次核验通过，保存于 `ik-environment-preflight-20261003/reviewed-pre.json`、`reviewed-post.json`、`audit-reviewed.json`。尚未启动9202、未跑真实BM25；下一步仅在明确获准启动隔离节点或提供可用测试URL后继续。代码全部保留在隔离worktree，未提交/推送；详细次要待办及执行取舍见该处最新账本。

## 2026-10-03 本批真实 BM25 已完成（最新）

用户批准复用9200创建专属临时测试索引，不再等待9202。新同集群授权/归属保护11测试先失败再通过，独立审查无阻断；真实集成1passed，临时索引及4条合成数据已精确删除，正式v4/alias未改。全仓非在线回归1455passed/35skipped/145deselected，2旧警告。实现仍在 `/Users/steven/.codex/worktrees/ik-rag-evaluation/AgenticRAG`，分支及基线不变，未提交/推送。

真实报告：`/Users/steven/LzmWorkSpace/AgenticRAG/var/artifacts/evals/ik-bm25-paired-v4-20261003-shared9200/report.md`（详细JSON同目录）。100题/200次BM25全部成功，100有效配对；standard→IK的FactRecall@40为64%→67%，完整证据覆盖54%→67%，MRR@40为0.481889→0.423347，binaryNDCG@40为0.513403→0.482907。覆盖提高但排名指标下降，不是整体质量提升或上线结论。

首次与resume均退出0；resume新增检索0、QueryPOST0、模型0，语料/词法前后核验通过、ledger/results哈希不变。实验与金标产物不要覆盖/重建。当前六项BM25工程任务完成；全链路问答、Ragas/AnswerCorrectness、人工审核/校准和validation/test仍未执行。下一步以隔离worktree同名进度文档最新小节和恢复账本为准，不再重复启动9202或重复200次实验。
## IK 查询分词兼容性对照（2026-10-03；已完成，main）

- 用户已批准查询端 `ik_max_word` 对照，并明确要求在 `main` 修改；此次授权覆盖旧计划的分支约束，不改变正式检索配置。
- 从已完成的测评工作区迁入 main 所需的八个只读评测模块及八个单测文件；不合并业务代码、服务配置、临时索引工具或旧实验数据。
- 固定同一 v4 物理索引、金标和五类各20道 dev 题，Top40、中文字段权重2、standard兜底权重1、best-fields最大得分语义、200次请求预算不变。仅实验臂 `.zh` 查询 analyzer 改为 `ik_max_word`，原 IK 臂重跑作为基线。
- [x] 单测 RED→GREEN：字段级覆盖、身份冻结、原始 adapter 漂移拒绝、两臂交替、报告方向与零请求恢复。新增反例先9 failed/30 passed，再39 passed。
- [x] main 全仓非在线回归与独立检查；不运行未授权的写入型或付费测试。1455 passed/34 skipped/144 deselected，2个已有警告；新增16个Python文件Ruff通过；独立审查无Critical/Important发现。
- [x] 新独占目录真实100题/200次只读检索；前后语料和分词环境核验；保存新报告，不覆盖旧实验。全部成功，100有效配对，4项严格核验均通过。
- [x] 验证完成后恢复新增检索为0，记录质量变化、范围限制和剩余问题。resume退出0，新增BM25/attempt/Query POST/模型调用均0，4项前后核验通过，ledger/results SHA256不变。

新报告：`var/artifacts/evals/ik-analyzer-paired-v4-20261003-main/report.md`，详细逐题结果见同目录 `report.json`。运行与恢复命令见 `docs/local-operations.md` 的“IK 查询 analyzer 对照”。恢复仍会只读扫描语料和分词环境；0指新增BM25实验检索，不是0次ES读请求。

| 指标 | 原 IK（ik_smart） | 查询 ik_max_word |
| --- | --- | --- |
| Fact Recall@40 | 67% | 68% |
| Complete Evidence@40 | 67% | 60% |
| Complete Evidence@3 | 43% | 60% |
| MRR@40 | 0.423347 | 0.518976 |
| binary NDCG@40 | 0.482907 | 0.553198 |

同一批100题的题目身份及顺序、金标包和快照与旧实验完全相同；原IK臂100题的请求、命中ID、排名、得分完全复现旧实验。旧实验ledger/results/report哈希未变。另做了4次只读 `_explain` 诊断（不计入200次实验search）：FAQ示例中2025/2026文档原同分44.500668，候选分别62.913967/65.63708，正确年份的`.zh:2026`开始贡献得分；未额外调用模型。

关键取舍：不是无条件提升，不自动切换正式配置。

- 单跳20题的完整证据@3由10/20提高到20/20；表格计算完整证据@40由0/20提高到8/20。
- 跨年多跳20题的完整证据@40由13/20降至0/20，Fact Recall@40由65%降至40%；章节理解由14/20降至12/20。因此总完整题数67→60，不能只看MRR/NDCG宣布改善。
- 已保存结果中的例子 `S012-2026-policy:year-comparison`：原两年证据排名19/20；候选2026年证据升至10，但2025年证据不在Top40。候选Top40被不同公司的2026年文档占据且同分56.583126，说明年份区分改善并未解决公司区分与跨年证据完整性；不能把这一原始BM25结果当作ResearchAgent实际多跳结论。
- 后续需单独设计并批准实体/公司上下文补齐、跨年份子查询与证据完整性对照；本批没有更改这些行为，也没有调整候选数量或重新生成金标。

代码保留在main工作区，未提交/推送；正式src、配置、mapping、alias、SQL和原文未改。本批只执行BM25；Dense/RRF/rerank/Parent/最终回答、Ragas/AnswerCorrectness、人工审核/裁判校准及validation/test仍未执行，不能作为上线结论。旧的main/隔离分支恢复指引以本节为准，旧实验产物仍保留。

## 2026-10-04 多跳20题全链路对照：执行边界预检

用户同意下一步比较上述20道多跳题在两种查询analyzer下的真实最终回答与多轮补救，而不是将单次BM25的0/20当作系统失败；继续使用main，不同时修改分块、元数据、grader规则或研究预算。

- [x] 从 `ik-analyzer-paired-v4-20261003-main/report.json` 确认原20个multi_hop case ID，不重新抽样或改金标。
- [x] 只读检查8000端口的 `/v1/runtime/summary`：测评请求已启用，所列依赖均available；此检查不代表完整语料/实现身份预检已完成。
- [x] 确认当前 `EvaluationMetadata` 只支持会话隔离、禁用记忆与候选预算，不支持逐Run的analyzer选择；`ik-smart-vs-max` 仅存在于单次BM25实验客户端。
- [x] 用户已批准执行边界：复用生产组成的真实QueryGraph，在独立测评进程中只对候选臂的中文BM25字段覆盖analyzer，真实调用模型/embedding/reranker并保留完整轨迹；明确不覆盖HTTP、Outbox、Worker队列。两臂必须都走相同入口，不拿API基线与Graph候选直接比较延时。
- HTTP到Worker不在本轮范围；若以后需要，须另行批准受控测评配置扩展及部署安排，不能自行切换正式配置、重启服务或绕过快照校验。
- [x] 补入口与防护测试：23个新增用例，安全回归1478 passed/34 skipped/144 deselected，Ruff与diff检查通过。独立审查发现的跨臂缓存替换、技术故障固定回复误送裁判均已复现并修复。真实composition/reranker加载验证通过，0付费调用。严格前置探测43次ES只读调用通过，正式语料/SQL/原文/IK绑定未变。
- [x] 开始冻结双臂实验：`var/artifacts/evals/ik-querygraph-paired-v4-20261004-main`，20题/每臂一次，QueryGraph独立入口；已批准启动真实模型与裁判调用（结果以逐题账本为准）。执行中不得修改evals/src实现以免身份漂移。
- [ ] 真实执行并保存最终答案、金标正确性、各次grader决策/gaps、后续任务/检索、累计与最终上下文覆盖、时延和用量。判别首次足够、多轮补齐、误放行、重复检索、打包丢证据和预算耗尽，不能只统计成功答案。
- [ ] 验证resume不重复已完成请求，生成两臂对照报告；模型缺失用量/价格时记unknown，不当作0费用。

已新增 `evals/paired_graph*.py` 与配套测试，使用真实生产composition；SQL只读、正式ES仅search，事件/工具账本写实验目录、禁用记忆与历史会话；与本批文档检索无关的第三方工具拒绝连接（两臂相同，明确报告限制）。默认6个研究动作/300秒不变；每题provider最多100次、每臂裁判最多500次、每题评分600秒。生产gateway原有有界重试保留并计费；测评器不重发已经开始且结果未知的整次Query/评分。下一步：观察原20题×2真实运行→后置核验与报告→完成态零网络resume验证。若中断，按local-operations中的同一命令加`--resume`，保留uncertain，不重开目录重跑已付费题。旧100题BM25报告和金标保持不变。

### 首题记录修复与获准续跑（2026-10-04）

- 原目录 `ik-querygraph-paired-v4-20261004-main` 首题S028基线真实完成：research_round_limit，6动作、7次检索、262.8598秒，15次provider请求（67491输入/24131输出tokens）。grader首次返回insufficient并指出公司主体不明和年份数值冲突；真实最终答案为空，不属于成功补救。
- 发现测评投影兼容缺陷：将未经答案审计的Parent IDs附到轮次上限终态，严格PublicAnswer拒绝；正式Worker会排除这种未经审计的引用。测评器自停，剩余39次未发送；源目录前后严格核验通过且保留不改。
- 已修复 `evals/collector.py` 与Graph投影，回归1491 passed/34 skipped/144 deselected，Ruff与diff检查通过；新增终态、不可变来源、代码/语料/区间漂移与离线恢复测试，独立审查无阻断项。
- 用户明确批准：新建续跑目录 `var/artifacts/evals/ik-querygraph-paired-v4-20261004-continued`，只继承首题真实结果并执行剩余39次。通过 `--continue-from var/artifacts/evals/ik-querygraph-paired-v4-20261004-main` 启动；不重跑首题，不更改其原始失败账本。
- 续跑身份固定源目录78个JSON产物哈希、旧运行快照/金标/预算，只有本次4个测评修复文件允许变化；`continuation-proof.json`记录离线重建证明。首题仍按轮次耗尽/无答案计0分，旧调用用量保留，恢复新增provider请求0。
- 后续恢复目标必须是 `...-continued` 目录加`--resume`（命令可保留`--continue-from`），不得对旧目录强行改身份或重跑首题。当前真实执行状态与最终结论以continued目录逐题账本和最终报告为准；还需完成后置核验、完整态零网络resume与结果总结。
