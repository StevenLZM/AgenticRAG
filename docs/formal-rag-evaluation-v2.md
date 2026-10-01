# 正式环境 RAG 测评 v2

本入口复用已上传的文档与正式Child索引（API8000 / ES9200）；不生成测试原文、不上传、不重建索引。旧8文档测评入口仍兼容，结果与v2分开。

## 一条命令

先核对，不调用回答/评分模型，不写结果文件：

```sh
./scripts/eval_rag.sh \
  --dataset var/artifacts/evals/formal-corpus-gold-v2-20260930-production/gold.v2.2.jsonl \
  --corpus-snapshot var/artifacts/evals/formal-corpus-gold-v2-20260930-production/snapshot.json \
  --api-url http://127.0.0.1:8000 --split dev --smoke 20 --check
```

真实20题分层运行（输出目录必须不存在）：

```sh
./scripts/eval_rag.sh \
  --dataset var/artifacts/evals/formal-corpus-gold-v2-20260930-production/gold.v2.2.jsonl \
  --corpus-snapshot var/artifacts/evals/formal-corpus-gold-v2-20260930-production/snapshot.json \
  --api-url http://127.0.0.1:8000 --split dev --smoke 20 \
  --pricing evals/pricing/2026-10-01-peak-cny.json \
  --output-dir var/artifacts/evals/my-formal-smoke \
  --max-cases 20 --max-judge-requests 1000 --max-wall-seconds 3600
```

验证集全测：去掉`--smoke`，改`--split validation`，使用新输出目录，`--max-cases 890`。独立test应在dev调参、validation选方案之后运行，不用test挑参数。全4453题需`--split all --max-cases 4453`；跨split全量是诊断汇总，不是无泄漏调参。

恢复：`./scripts/eval_rag.sh --resume <已存在的输出目录>`。复用提交时的run_id，不重发已提交查询；已完成评分不重新调用。未知POST或评分响应保留running/failed账本，必须先对账，不能把删账本当正常恢复。

恢复并非只读结果JSON：重新核对API、SQL、checkpoint、实际上下文、字段抽取和逐项裁判记录，任何缺失/变化会拒绝混用。已有结果缺查询账本时禁止重发查询。源码、provider配置、裁判、费率和输入均绑定指纹，改变后必须新建实验。

若仅评估新裁判对历史回答的评分，使用`--replay-from <旧实验> --output-dir <新目录>`，可加`--pricing`。只复用旧run_id并实时核对，不重新执行旧查询，不补发旧实验未提交题目；后者列为pending。历史回放不代表新部署效果，不能参与新参数选择。

若升级的仅是测评评分器，需要同时接续剩余题目，使用`--continue-from <旧实验> --output-dir <新目录>`：查询部署指纹、provider配置、session和检索预算必须完全一致，已有Run复用并重新评分，未提交题目才发起新查询。业务源码/模型/预算改变会拒绝该模式。与resume不同，它是有来源绑定的新评分实验，旧成绩不覆盖；不能把它当免费重评或新的系统查询基线。

## 配置与隔离

- API明确启用`AGENTIC_RAG_ALLOW_EVALUATION_REQUESTS=true`并重启；默认配置false。只接受隔离`eval-v2-`thread、服务器固定用户、memory_policy=disabled及有限候选预算。不会清空Mem0或修改普通会话。
- 固定Ragas0.4.2，独立评测venv；`RAG_EVAL_PYTHON`可指定已配置的解释器。脚本不自动安装依赖。
- 默认裁判是`Settings.light_model`，与main回答模型不同。可用`RAG_EVAL_JUDGE_MODEL/BASE_URL/API_KEY`配置独立模型/供应商；密钥不写结果。
- 裁判temperature=0，schema/rubric/judge/embedding版本及指纹写入实验；人工校准未完成会明确标注。不是“换个模型就保证裁判正确”。
- 官方DeepSeek裁判默认禁用thinking，输出上限8192，避免推理耗尽输出预算后JSON被截断；配置纳入裁判指纹。
- `--retrieval-budget <JSON>`接收dense_k/bm25_k/rrf_k/rerank_k/parent_k/rrf_constant；输出预算不可超过上游，最大有限。用于本Run，不发布到生产默认参数。默认40/40→30→10→6不变。
- 相同generation仍可能数据变化。预检比对全部原文、版本、Child/Parent内容和成员、cluster/index UUID及alias，不只校验正例。严格模式漂移停止；`--snapshot-mode robustness`另记真实漂移，原本可回答题不会改成无答案。

## 看报告

- `report.md`：先看task_success、answer_correctness、factual_correctness P/R/F1、关键字段，再看阶段检索。
- `report.json`：每项value/有效n/总n/缺失数、类型/公司/格式/路由分组、公司bootstrap、执行失败、全量保守通过率。
- `failed_cases`表示未能形成有效逐题结果的测评采集异常；成功采集到的research_round_limit、空回答等仍在observed_cases中，答题失败须看outcomes/task_success。评分器自身失败见scoring_failures和各指标missing，不能仅凭failed_cases=0宣称系统全答对。
- `cases/`与`traces/`：金标事实→有序Child候选→Parent→实际裁剪上下文→最终回答；各次检索独立，跨轮只有union覆盖，不虚构全局RRF/MRR。
- `queries/`：提交/未知响应/run_id；`judges/`：字段、事实判定、逐项Ragas账本；`judge-requests.json`：真实SDK请求及provider usage，不保存提示词或密钥。
- 本地完整轨迹目录权限700；报告正文不导出个人原文。收费token以provider usage为准，缺失为unknown，不把字符预算写成计费token。没有完整用量/单价，不会假报费用。
- `--pricing`显式指定冻结费率，记录SDK每次模型/Embedding请求及缓存token；查询、裁判和总费用分别报告。示例费率是固定峰时人民币口径，适合配对比较，不代表实际账单（节假日/错峰/批量/优惠未自动推断），也不含本地rerank GPU、ES、MySQL资源成本。未知请求或未报价模型保持unknown。

## 口径和限制

- 每个事实的充分证据组合内AND、组合间OR；多跳必须满足全部事实。MRR高不等于整题完成。
- Child和Parent分别评分；最终上下文还验证实际文本，不能因为Parent ID出现就宣称被裁掉的事实存在。
- binary NDCG与graded NDCG分开；没有人工graded qrels时graded=null，未判断候选不是0相关。
- 字段抽取只收到问题/最终答案/字段定义，不收到金标数值。quote必须来自答案，金额/单位/对象/年份再由Python独立比对。
- Ragas AnswerCorrectness含事实与语义分量；FactualCorrectness和关键字段另行把关。[官方AnswerCorrectness](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/answer_correctness/)、[官方FactualCorrectness](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/factual_correctness/)。
- Factual P/R/F1共享同一份真实事实拆解和双向NLI判定，保存声明、理由、TP/FP/FN和sample_id，再用Ragas0.4.2的计数口径计算未四舍五入的三项分数；避免三次独立采样导致P/R/F1彼此不一致。v4保留Ragas事实拆解，NLI改用带claim_id的Structured Output适配：模型只判定原编号，Python校验完整编号集合并保留原声明，不要求模型逐字回显文本。漏号、重复号、越界仍为评分失败。它是明确版本化的Ragas适配器，不冒充未经改造的原版FactualCorrectness；各版本不能混用。
- 无答案题看正确拒答；research_round_limit/tool_error/timeout不是正确拒答，且不会从总分母删除。评分失败为null，不能伪造0/1。
- 当前金标原文程序核验通过但人工语义审核待完成。4453题模板占比高，须看类别宏平均；自然问法/困难负例、人工qrel和Judge分层校准仍需继续。
- 当前已知1组真实PDF阅读顺序导致qrel unmapped，保留题目和原文答案，阶段排名unknown，端到端仍评分；本次不修改解析器或重入库。
- 自动报告不等于上线验收：human review/calibration、独立test及成本约束未确认时production_release_accepted始终false。

## 候选预算的可执行调参入口

用`PYTHONPATH=src:. <项目Python> -m evals.tuning plan --stage rrf --base evals/budgets/baseline.json --output-dir <新目录>`生成RRF30/50/80等逐阶段预算；stage还支持recall/rerank/parent。命令只生成预算，不自动发起收费查询。

每个预算显式传给真实命令`--retrieval-budget <budget-XX.json>`，使用完全相同的dev题集和其余配置。取得报告后比较：

```sh
PYTHONPATH=src:. <项目Python> -m evals.tuning compare \
  --split dev --reports <实验1/report.json> <实验2/report.json> \
  --output <新比较报告.json>
```

只接收同原文/索引/模型/源码/裁判/费率/同题集合且 `eligible_for_tuning` 明确为布尔 `true` 的真实报告；该字段缺失、未知或为false时均不参与Pareto，旧报告不能因缺字段绕过门槛。评分不完整、费用或P95未知者同样排除。当前尚未完成人工审核/校准的v5报告全部不具备选参资格，不能手改标志代替批准。对将来合格的候选用validation再测，比较时改`--split validation`，按整题成功率→AnswerCorrectness→P95→查询费用选方案。test拒绝参与选参，任何命令都不修改正式默认参数。

## 人工审核和Judge校准交接

使用`PYTHONPATH=src:. <项目Python> -m evals.review_v2`，公共参数`--dataset <金标> --corpus-snapshot <快照> --output <新文件>`：

- `gold-export`：导出每类至少20题、全部多跳/无答案/历史题的审核JSONL，包含原文锚点、充分Child组合和待填写reviewer/reviewed_at/decision/reason。不是AI代签人工审核。
- `gold-audit --reviews <人工填写的JSONL>`：核对原金标内容指纹和审核记录，报告通过、拒绝和缺失。不改原冻结金标；发现错误需新版本与新实验。
- `calibration-export --experiment <真实实验目录>`：导出实际回答与金标，但隐藏自动裁判结论，避免先看到模型判断影响人工。人工填写human_fully_correct及有时区的审核时间、理由。
- `calibration-audit --experiment <同一实验> --reviews <人工填写的JSONL>`：严格核对问题/答案/实验绑定，统计一致率、Cohen kappa、误判为正确/错误及类别覆盖。至少100例且五类各20例、含人工正确和错误标签才标记样本齐备；阈值仍需人工批准。禁止held-out test校准裁判。

该audit针对独立整题verdict，不等于Ragas所有子指标已校准。仍要专项核查事实拆解是否保留实体/年份/数字、短reference是否缺少语义条件，并由人工确定各指标判定门槛。一次结构化评分成功不表示裁判判断必然正确。

审核导出只生成受限权限的本地交接文件，不自动填写标签、更新金标或授予发布通过。当前自然问法/困难负例扩充、人工graded qrels、完整validation与test验收仍是独立待办。

人工核对原文时，用审核记录的document_version_id在冻结snapshot.json的documents中定位source_path；canonical_ast_path仅帮助检查解析与分块，不能代替原文真值。

费率依据：[DeepSeek官方价目](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[阿里云Embedding官方价目](https://help.aliyun.com/zh/model-studio/embedding)。

恢复实施账本：`docs/formal-rag-evaluation-v2-progress.md`。

## 已完成的真实基线（2026-10-01）

报告目录：`var/artifacts/evals/formal-v2-smoke-20261001-indexed-factual`。正式索引20题、15公司、五类各4题；20条Run全部核验、适用评分齐备。9条生成回答，8条研究轮限、3条cannot_answer无正文；独立整题判定暂为9/20，人工校准未通过。

完整恢复在当时的 v4 代码和配置下已实测零新建Query API POST、零新增裁判请求。以下命令只适用于匹配该版本的环境；升级到 v5 后应新目录回放，不直接续写历史 v4 实验：

```sh
./scripts/eval_rag.sh --resume var/artifacts/evals/formal-v2-smoke-20261001-indexed-factual
```

这不是4453题全量成绩或生产验收。RRF80仅完成1题连通性试验，结果见`formal-v2-rrf80-pilot-20261001`，正式默认参数未改变。人工交接文件在`formal-v2-review-20261001`。最新基线已经暴露事实拆解改写实体的风险，后续应先校准，再做全量调参/验收。

## 原文绑定事实评分 v5.1

原文绑定抽取只选择回答/问题中的实际片段；来源和角色范围由程序校验。判定器再次核对抽取的关系、条件、否定和覆盖，发现抽取失真则保留失败证据、记unknown，不记系统答错。程序不能保证语义无误，仍需第二批人工校准。

原文quote不能改写。位置错误仅在全局唯一或已核验父证据内唯一时修正；重复“元”等无法唯一定位的引用保持unknown，不猜测指向。金额与完整数值/倍率/货币表达绑定，例如`500万元`不得抽成`500 CNY`；`0.05万元`经明确换算可以匹配`500 CNY`。不支持或有歧义的表达进入裁判复核，不擅自补成金标。

`grounded_factual_precision/recall/f1` 是项目自定义主事实指标：回答支持率与必需金标覆盖率分别用各自分母，三项共用一个判定记录。旧Ragas算法和名字保留作诊断。关键字段、主事实、独立整题判定分别保存，主裁判失败不能被另一项通过掩盖。

评分说明独立于冻结gold，绑定原case和原文来源。当前版本只接受可重建的待审草案：从reference按标点分出的事实候选、问题条件、来源事实和关键字段；其原子性及必答范围不冒充人工审核。所有草案报告 `eligible_for_tuning=false`。当前生成的正式v2报告在人工校准门槛完成之前也不参与自动选参。

生成一次说明文件（重复输出同路径会拒绝；不调用模型、不访问或修改ES）：

```sh
PYTHONPATH=src:. /Users/steven/miniconda3/envs/agentic-rag/bin/python -m evals.gold_answer_spec \
  --dataset var/artifacts/evals/formal-corpus-gold-v2-20260930-production/gold.v2.2.jsonl \
  --corpus-snapshot var/artifacts/evals/formal-corpus-gold-v2-20260930-production/snapshot.json \
  --output var/artifacts/evals/formal-v2-grounded-spec-20261001/answer-spec.jsonl
```

新目录回放已有20个Run（首次新裁判评分收费，但不提交新查询；`--check`仅预检）：

```sh
./scripts/eval_rag.sh \
  --replay-from var/artifacts/evals/formal-v2-smoke-20261001-indexed-factual \
  --answer-spec var/artifacts/evals/formal-v2-grounded-spec-20261001/answer-spec.jsonl \
  --output-dir var/artifacts/evals/formal-v2-grounded-smoke-20261001 \
  --max-cases 20 --max-judge-requests 600 --max-wall-seconds 1800
```

单题检查加 `--case-id S004-2026-policy:lodging` 并换独立新目录。恢复用 `--resume <新实验目录>`；gold、评分说明、Judge、代码、费率和索引必须一致。已有失败/未知调用保留，不能通过resume反复收费重抽；换评分器须新建版本化实验。

恢复开始时（包括零模型调用的缓存恢复）先核验`judge-requests.json`的实验绑定、请求记录及成功评分对应的操作证据，累计预算包含过去全部尝试。已有评分产物而账本丢失、异实验账本或记录损坏时拒绝恢复，不能从零重建付费历史。报告读取同一核验后的账本；未知用量不估成零。

不带 `--answer-spec` 的兼容路径只运行旧评分流程，不代表已启用主事实保真评估。出现真正金标错误时发布新gold，不放宽跨gold hash恢复校验。上述命令为当前入口说明，是否已跑完及实际成绩以恢复账本和对应report为准。

## 原文位置选择修复 v6.1（历史实验）

v5.1的真实回放有6条答案因模型数错Unicode字符下标而定位失败。v6.1把这一机械工作移回程序：

- 程序保留原文，将完整回答句子列为`segments`，句内分句列为`source_catalog`（回答r编号、问题q编号）。金额的千分位逗号和小数点不切断。
- 模型的`evidence_segment_ids`只选择原始整句编号，程序填充完整证据；同一句中后半句沿用前半句主体时不会丢上下文。角色位置只选择`span_id + quote`，不再让模型输出字符下标。
- quote必须是指定分句内唯一、完全相同的原文；同一分句的重复短词要求选择更长的唯一原文片段，仍有歧义就失败。程序不猜最近位置，不借金标补全，不做模糊匹配。
- 字段提取也使用编号来源；单位与条件证据在输出schema中为必填属性，缺失不能静默变成空列表。数值仍只来自回答，并继续执行金额倍率、单位和条件校验。
- 下游依旧使用原来的`SourceSpan`、`GroundedFact`与关键字段比较。新的schema、rubric、协议与版本`formal-answer-v6.1-numbered-source-factual-v1`进入实验指纹；不能用这版代码直接续写v5.1目录。

该修复降低的是机械定位失败，不证明LLM的语义关系、否定范围或事实完整性已经校准。错误的关系或跨主体拼接仍需主裁判核验，人工审核与校准门槛不变。

历史 v6.1 回放命令（目录已存在，保留原样；当前 v6.2 代码不能直接恢复该版本）：

```sh
./scripts/eval_rag.sh \
  --replay-from var/artifacts/evals/formal-v2-grounded-smoke-20261001 \
  --answer-spec var/artifacts/evals/formal-v2-grounded-spec-20261001/answer-spec.jsonl \
  --output-dir var/artifacts/evals/formal-v2-numbered-smoke-20261001 \
  --max-cases 20 --max-judge-requests 600 --max-wall-seconds 1800
```

实际完成状态、有效评分数量及零重复调用审计见恢复账本的最新小节。所有旧评分和失败探针保留，不能将改版后的高分回填旧实验。

## 当前修复版本 v6.2：本句单位依据与失败可见性

版本为`formal-answer-v6.2-numbered-source-factual-v1`，沿用上述编号原文选择。v6.1已消除6条旧位置错误，但真实字段抽取给出“本句完整单位证据 + 下一句重复单位说明”时，会被旧规则误拒。v6.2保留并验证所有逐字引用，只允许数值引用所在原句的依据以及问题中允许的计数口径参与单位证明；多余句外引用既不导致已有充分依据失效，也不能补齐缺失依据。其他句子即使出现同一个数值，也不能提供单位或计数口径。金额倍率、货币绑定和来源校验不放宽。

字段抽取失败原因现在同时保存在每题结果与报告`scoring_failures.critical_fields`中；不能再出现字段评分失败但报告失败列表为空的情况。

真实20题历史回答回放目录为`var/artifacts/evals/formal-v2-numbered-local-units-smoke-20261001`。9条实际回答的主事实评分、6条适用回答的关键字段评分均完成，`scoring_complete=true`；这表示适用评分齐备，不代表20题全部答对，也不代表裁判已人工校准。自动整题判定7/20，11条运行失败和2条缺失必需事实仍计入分母。

在完全匹配的代码、配置、金标、评分说明和语料快照下恢复：

```sh
./scripts/eval_rag.sh \
  --resume var/artifacts/evals/formal-v2-numbered-local-units-smoke-20261001 \
  --max-cases 20 --max-judge-requests 600 --max-wall-seconds 1800
```

同配置恢复已实测退出码0、新Query POST=0、Judge SDK调用=0、付费账本新增请求=0，Run ID及账本hash不变；审计见该目录`audit-resume.json`。若业务或评测代码后续变更，指纹校验会拒绝直接恢复；不要删除绑定或账本绕过检查，应使用新实验目录。历史v5.1、v6探针、v6.1的失败及费用保留；本轮没有新Query POST、上传、索引改动或参数调优。人工金标审核、事实级校准、完整validation/test仍未完成。
