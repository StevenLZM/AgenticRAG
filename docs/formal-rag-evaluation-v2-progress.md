# 正式 RAG 测评 v2：可恢复实施账本

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
