# RAG 真实评测：可恢复执行记录

用户续接口令：**继续 rag 测评优化**（大小写、空格不限）。

恢复时先读本文件，再读关联设计与计划，核对 `git status --short`。不要从头重跑上传、模型请求或创建资源；先检查下方运行账本与已有结果。完成状态必须有实际验证证据。若上下文或 token 耗尽，先更新这里再交接。

## 已批准范围（2026-09-16）

- 移除业务评测的 fixture/标准答案回填/固定向量捷径；数学公式和接口的单元测试保留，但不作为业务成绩。
- 首轮生成 8 份中文 PDF/TXT/XLSX 测试文件、24 道预先标注的问题（含无答案问题）。
- 必须通过真实 HTTP 上传接口，等待真实解析、切分、Qwen embedding、索引发布；不得直接构造 Parent/Child 写库。
- 通过真实 HTTP Query API、Query Worker、生产 Graph、ES、reranker、模型生成/审计运行评测。
- 接入真实 Ragas judge。缺依赖、缺配置、请求失败记录为错误/未完成，不伪造分数，不冒充评测通过。
- 使用隔离评测用户、MySQL 数据库、Redis 隔离空间、索引代际、Mem0 集合、artifact/checkpoint 路径及 API 端口；保留业务文档和记忆。
- 按阶段依次实施。未经用户新请求，不混入 Chat/RAG 流式输出改造。

设计：[real-rag-evaluation-design](superpowers/specs/2026-09-16-real-rag-evaluation-design.md)

计划：[real-rag-evaluation-plan](superpowers/plans/2026-09-16-real-rag-evaluation.md)

## 状态

| 阶段 | 状态 | 完成凭据 |
|---|---|---|
| 0. 保存设计、计划与恢复入口 | 完成 | 本文件及关联文档 |
| 1. 真实评测契约、金标与检索指标 | 完成 | 独立审查发现的缓存金标问题已修复；108 passed、2 skipped；Ruff/diff check 通过 |
| 2. 生成并校验 8 份语料、24 道题 | 完成 | 实际文件通过安全扫描、PDF/XLSX渲染及文字/数值校验；回归147 passed、2 skipped |
| 3. 真实 Ragas 评分器与严格失败策略 | 完成 | 90 项契约测试通过；真实 judge/embedding 两样本预检 available；不计业务 RAG 成绩 |
| 4. 隔离运行栈与真实上传/入库 | 完成 | 8 份真实上传/入库完成；8 active 文档/版本、21 Parent/21 Child；39事实/24题金标映射完成；回归132 passed |
| 5. 真实检索排名、上下文与运行证据采集 | 完成 | collector、Run账本、逐轮逐阶段排名、真实运行计数、只读重新核验；tokens未持久化，明确unavailable |
| 6. 3 题预检后全量 24 题真实评测 | 完成 | session4898退出0；24题真实Ragas available，24个核验Run、0运行错误；正确拒答1/4 |
| 7. 报告、回归测试与交付 | 完成 | 旧入口退役、可信验收、恢复保护、最终报告完成；最终609passed4skipped，另live readonly E2E1passed；legacy/恢复保护独立审查通过，核验模块交叉审查无未解决重要问题 |

## 当前下一步

**2026-09-19 一键重测扩展完成（用户已批准）**：新增 `./scripts/eval_rag.sh`，默认新实验；`--resume` 显式续跑。只复用已有文档、冻结金标、已建索引和服务，不生成新语料、不上传/重建。新增 `evals/command.py` 做只读前检、输入绑定、实验锁、完整链路调用和 JSON/中文 Markdown 报告。原 evaluator/verifier 增加独立 output 参数，旧接口保留。`--check` 已在实际环境通过（8 文档/24 题/现有代际）；最新回归 156 passed（包含 live readonly E2E），Ruff/bash 语法/diff check 通过。独立审查无剩余重要问题；指出的“核验失败但 summary 尚未生成导致报告显示 0 题”问题已用 RED/GREEN 修复：从逐题账本生成诊断汇总，保留失败/损坏记录，可信查询计数置 0。另补上 `--resume --check` 的只读绑定检查。尚未用新入口发起新一轮付费 24 题；本次验证复用了既有真实结果，没有产生新业务成绩。命令及限制见 `docs/local-operations.md` 的“一条命令运行既有语料的真实 RAG 测评”。

**2026-09-18最新状态：本轮测评实施与交付完成。** 全量24题已完成，勿重跑上传、Query或judge。最终结果见[真实测评报告](./rag-evaluation-report.md)。分阶段排名、SQL/checkpoint collector、durable Run账本、live核验、旧入口退役和失败记录恢复保护均已完成。低分与tokens不可用等限制保留，未声称生产发布通过。

**最新进展**：Query账本与非阻塞文件锁已实现；轮询中断复用run_id，未知POST结果拒绝自动重提。HTTP run_id和公开答案与collector核对，按PublicAnswer模型归一化null默认字段。`evals/verification.py`对结果重新读取scoped MySQL/checkpoint，核对评分答案/上下文/排名，再写运行证据哈希和真实计数；重算确定性指标不重复judge。

**下一步不再是恢复未完成测评**：用户若继续优化，应以这份冻结结果为baseline，单独开展无答案拒答/研究停止策略A/B，不改gold或覆盖本轮成绩。仅需复核时运行报告里的只读命令。任何新模型实验使用新目录并明确范围。

隔离资源保留在 `var/artifacts/evals/real-corpus-v1-20260916/stack-allocation.json`。8份上传及金标映射均完成，勿重复创建资源/上传/评分预检；若需核对，仅运行下方可恢复命令。阶段1验证命令：

```sh
conda run -n agentic-rag pytest --import-mode=importlib tests/unit/evals tests/integration/evals tests/integration/api/test_query_runs.py tests/e2e/test_release_query_gate.py tests/e2e/test_backup_restore.py -q
```

当前全量 `real_query_count=24`，来自scoped MySQL/checkpoint重新核验。外部报告验收CLI要求`--run-dir`，冻结gold/上传/运行凭据/评分输入/汇总必须全部匹配。`--quality-only`核验退出0；不加此参数的发布验收退出1（预期，拒答低分且未做恢复/备份）。测试通过不等于效果达标。阶段1原始RED/GREEN见`.superpowers/sdd/2026-09-16-real-rag-evaluation/task-1-report.md`。

阶段 2 完成：2 PDF、4 TXT、2 XLSX已生成。`evals/corpus.py` 复制固定资产并校验哈希；相同输入续跑不重写，文件或金标变化会拒绝复用，中断复制不会留下半个正式文件。修复真实 XLSX 暴露的标准 `_rels/.rels` 分类缺陷，XML/DTD安全检查保留。完整验证见 `.superpowers/sdd/2026-09-16-real-rag-evaluation/task-2-report.md`。

执行方式补充：独立实现/审查代理在阶段 1 审查后遇到调用额度限制。主代理已本地复现缓存缺陷（6 个新测试先失败），完成修复并验证 108 passed、2 skipped；未谎称有第二次独立审查。后续暂由主代理本地逐项实施。基础设施只读预检报告已于 2026-09-17 完成，见 `.superpowers/sdd/2026-09-16-real-rag-evaluation/stack-preflight-report.md`：MySQL 3306 实际认证查询成功；Redis DB1 当时为空、DB15 已使用；业务 ES9200 禁止用于评测发布；独立 ES 镜像和容量仍待验证。分配资源前必须重新检查。

## 已核实约束

- 当前工作目录 `/Users/steven/LzmWorkSpace/AgenticRAG`，分支 `codex/real-rag-evaluation`（从 main 创建，完整保留已有未提交改动）；不能清理或整体提交。
- API 以 `settings.default_user_id` 确定用户，不支持任意 HTTP header/body 切换身份。独立评测必须有独立配置的 API/Worker。
- 入库 outbox 使用固定 stream `agenticrag:jobs:ingestion`；只改用户或索引不够，必须隔离 Redis 与数据库/dispatcher，防止现有 Worker 误领评测任务。
- 只读预检新发现：ES 发布使用共享活动别名 `agenticrag-children-active`，仅指定新 index_generation 不保证隔离。阶段 4 必须采用独立 ES 实例（优先）或经过测试的独立别名机制，未解决前禁止上传/发布；不得重指向业务活动别名。
- 上传允许 PDF、UTF-8 TXT、XLS/XLSX，不包括 DOCX/Markdown。
- 2026-09-16 探测：Conda `agentic-rag` 有 docling 2.118.0、openpyxl 3.1.5；未安装 ragas、datasets、langchain-openai、reportlab。不要仅安装依赖就宣称真实评分完成。
- 原 `run_real_query_acceptance.py` 使用固定向量和直接播种 Parent/Child，不能作为本任务的上传/召回质量证据。
- 原 API evaluator 的 `contexts=[]`、expected_route 回填及最终 evidence 代替 retrieval ranking 必须修正。

## 运行账本

### 最新恢复点（2026-09-18）

- 全量已完成，无后台评分待等待。正式结果`evaluation/results.jsonl`24行；`summary.json`24题；`failures.jsonl`为空。Ragas忠实度0.9833、回答相关0.9366、上下文精确0.8683、正确拒答0.25。
- `evaluation/verification-details.json`为本次只读live核验快照，包含24个Run、终态、SQL耗时、明确不可用的tokens；不是新一轮评分。所有Query/judge结果保持原值。
- 验收增加`evals/saved_report.py`，旧`run_real_query_acceptance.py`退出2且无副作用；固定向量仅保留contract-only协议fixtures，real_query_count=0。正式E2E通过`AGENTIC_RAG_EVAL_RUN_DIR`只读核验现有真实run。
- 初次审查`eval_verification_review`遭usage limit失败；之后`eval_legacy_finish`交叉审查其未实现的saved-verifier/runner，发现失败记录子集覆盖缺陷，已修复并经53项runner测试验证。最后独立审查`eval_completion_audit`通过legacy退役、fixture分类、E2E接线和恢复保护；其未审saved-verifier内部语义，不扩大审查结论。
- 最终相关范围回归528passed2skipped（含文档契约）；运行时/checkpoint回归81passed2skipped；live readonly E2E1passed；Ruff/diff check通过。总609项相关回归通过、4项按需服务/演练测试skip，并非全仓全量测试或生产演练通过。
- 失败记录与成功记录都纳入subset防护，已有状态时空dataset/limit=0拒绝，完整重试成功才清除同case失败；扩展题集中断保留已有后续评分。原24题不受本次保护修复影响。
- 资源继续保留，不关闭业务服务、不改`.env.local`、不自动提交或删除恢复账本。

### 历史恢复点（2026-09-17，以下运行状态已被上方最新记录取代）

- 原续跑session49776在API runtime summary请求遇到502，尚未执行评分；只读确认旧API/Worker PID均不存在、8001没有监听，隔离ES容器仍在运行。未修改业务服务。
- 恢复隔离API：PID22542/session52269；Query Worker：PID22551/session5353。API readiness全部available，snapshot仍为`a4e9f142025f5d2b523ebfa1741ee63225b9e46dbf6a32fab5ec95c5857899f4`。Ingestion旧PID已失效；8份上传已完成，本次不需要恢复ingestion。
- 三题预检session12660退出0，completed_cases=3、real_query_count=3、resumed_cases=2、query_failure_count=0、Ragas available=3。无答案题21的correct_refusal=0是真实质量失败，不等于评分器故障。01/13未重复Query或评分，21复用原Run完成评分。
- 回归重新执行：342 passed in 4.23s。
- **全量24题正在执行：session4898**。命令：`NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost PYTHONPATH=src RAGAS_DO_NOT_TRACK=true var/eval-venv-ragas042/bin/python -u -m scripts.run_real_rag_evaluation --run-dir var/artifacts/evals/real-corpus-v1-20260916 --stage evaluate`。NO_PROXY仅作用于本命令，未修改全局配置。
- 恢复时先读session及`evaluation/queries`账本；不要用三题子集覆盖全量结果。summary在全量完成前仍可能显示上次三题预检；以逐题持久化结果/账本判断运行进度。

2026-09-17 已创建隔离运行资源并启动真实上传。首次真实 judge 连通性评分已完成（不是业务 RAG 测评，real_query_count=0）。详细资源/PID/session 见 `stack-allocation.json`；每份上传的 job/document/version ID 见 `uploads/*.json`，恢复前先检查，禁止重复提交。

- Judge 独立环境：`var/eval-venv-ragas042`，继承应用系统包但安装变更仅在该 venv；固定 `ragas==0.4.2`、`langchain-community==0.4.1`，已验证 import 与 `pip check`。未升级业务 Conda。
- Judge：`deepseek-v4-pro`；embedding：`text-embedding-v3` / 1024 维；禁止记录密钥。
- 预检：`PYTHONPATH=src RAGAS_DO_NOT_TRACK=true var/eval-venv-ragas042/bin/python -m scripts.preflight_ragas --output var/artifacts/evals/real-corpus-v1-20260916/judge-preflight.json`；启动 session 15033。
- 预检已退出 0：回答样本 faithfulness=1.0、answer_relevancy=0.8509521450490493、context_precision=0.9999999999；拒答样本 correct_refusal=1.0。输入为专用连通性样本，不是实际检索输出，因此真实 Query 数仍为 0。
- 隔离资源计划账本：`var/artifacts/evals/real-corpus-v1-20260916/stack-allocation.json`，allocation_id=`e7a49c260917`。仅名称计划，不代表服务已创建。ES 镜像拉取 session 22809；磁盘可用 186Gi，Docker VM 总内存约 8.32GB。
- `evals/upload.py` 已实现真实 HTTP POST/轮询及持久化续跑；契约验证 6 passed（不是实际上传证据）。不确定 POST 不自动重试；超时/poll 失败用原 job_id 恢复。
- 当前资源：MySQL `agentic_rag_eval_e7a49c260917` 已迁移至 `0007_query_run_answer`；Redis DB1 owner=`e7a49c260917`；ES 容器 `agentic-rag-eval-es-e7a49c260917` / 9201，独立 cluster UUID `TThca-p0QyGzxHYXob16JA`，与业务不同。
- API8001 readiness 全部 available；API PID30805/session69656、ingestion PID30813/session15840、query PID30819/session66538。不要重启业务8000或ES9200。
- 真实批量上传 session80480，按 manifest 顺序串行，一份失败/超时即停止；第一份 `remote_2026` job=`01a0ad1a-050b-7808-9fff-32848cd7ac2d`。尚未确认入库完成。
- 新增 `evals/real_stack.py` 隔离环境校验 6 项通过；上传测试增至8项；最新评测回归104 passed，Ruff与diff check通过。
- **阶段4最终状态**：批量上传 session80480 已退出0，8个 job 全部 completed；隔离库8个 active documents/versions，21个 active parents；独立 ES21个 children。业务9200的活动别名仍指向 `agenticrag-children-index-v3`。
- 金标产物：`var/artifacts/evals/real-corpus-v1-20260916/gold-mapping.json`，39条预标注事实全部唯一匹配对应版本Parent，保留AST定位，24题映射完成。无金标修改、无检索结果反推。
- 阶段3/4最终回归：`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/evals tests/integration/evals tests/integration/api/test_query_runs.py -q` → **132 passed**；Ruff与diff check通过。Query数仍为0，24题RAG质量评测尚未运行。
- 可恢复核对：`PYTHONPATH=src conda run -n agentic-rag python -m scripts.run_real_rag_evaluation --run-dir var/artifacts/evals/real-corpus-v1-20260916 --stage upload`（复用已完成账本，不重新POST）；将stage改为`map`校验/复用金标映射。
- 实施记录：`.superpowers/sdd/2026-09-16-real-rag-evaluation/task-3-4-report.md`。API/两个Worker/ES保留用于阶段5/6；恢复时核对PID和端口，不要盲目重复启动。
- 独立审查 `review_eval_stages34` 因代理usage limit失败，无有效审查意见，不能称审查通过。主代理继续本地TDD。
- 阶段5当前验证：`conda run -n agentic-rag pytest tests/unit/evals tests/integration/evals -q` →120 passed；retrieval graph10passed；Ruff/diff check通过。尚无真实Query请求。注意SQL agent_runs.route当前未被Worker赋值，collector使用checkpoint实际route并与公开答案投影核对，不能假设SQL route非空。
- 阶段5扩大回归：`conda run -n agentic-rag pytest --import-mode=importlib tests/unit/evals tests/integration/evals tests/unit/retrieval tests/unit/query tests/integration/api/test_query_runs.py -q` → **332 passed**。Stage5还在进行中，不得将以上测试数写成真实Query数。
- 2026-09-17 评测Query Worker已定向重启：旧PID30819/session66538已正常退出，新PID75597/session99927；业务API8000/ES9200未动。重启时普通kill受系统权限限制，经精确PID权限审核后成功。
- **第一题真实全链路完成**：`real-v1-01` Run=`01a0ae30-94d2-737c-ae9a-2e6546fafa4b`；Query完成，Ragas available；Recall@6/MRR/NDCG@10=1，faithfulness=1，answer_relevancy=0.9854466184278233。不是全量成绩。
- 首次collector比对因API补null字段而停止在评分前；经PublicAnswer归一化修复，复用同一Run完成评分，没有重新Query。之后再次执行验证resumed_cases=1、real_query_count=1，没有重复judge。
- 运行结果：`var/artifacts/evals/real-corpus-v1-20260916/evaluation/`；其中queries为POST/轮询账本，runtime-evidence为已校验的运行证据，results.jsonl/summary.json为当前选定题集结果。**不要用不包含已完成题的子集覆盖当前results；下一次应含01/13/21或完整24题。**
- 当前命令：`PYTHONPATH=src RAGAS_DO_NOT_TRACK=true var/eval-venv-ragas042/bin/python -u -m scripts.run_real_rag_evaluation --run-dir var/artifacts/evals/real-corpus-v1-20260916 --stage evaluate --case-id real-v1-01 --case-id real-v1-13 --case-id real-v1-21`。
- 3题首次预检 session57202 已退出：01和13完成Query及Ragas并持久化；21 Run=`01a0ae39-d7b7-7fcc-aa78-52ce0df9ea53` 达到 `research_round_limit`，公开答案只有终止状态，旧评分进程未支持状态文案导致停止。已补现有console状态文案的真实投影（answer_origin=terminal_status），不使用参考答案填充。正在用同一命令恢复；01/13评分复用，21不重新Query。
- 多跳13 Run=`01a0ae36-83a0-799d-8887-08f8e398b765`，2次独立检索，Ragas faithfulness=1、answer_relevancy=0.7872108975306844、context_precision=0.8333333332916666。无答案题21的研究轮次耗尽是实际质量问题，不能改金标或删掉该题。
- Runner已增加逐题API失败文件failures.jsonl及requested_cases/query_failure_count，不因一题操作失败抹掉后续题；状态型答案按现有控制台文案评分并标记来源。最新扩大回归342 passed（Ruff通过）。
- 新增测试验证POST不确定/轮询中断/并发重复/题目变化/null字段/错答案等；评测模块128 passed，Ruff/diff check通过。独立审查仍未成功，不声称审查通过。
- 评分失败显式记为 failed，不返回伪造零分；更换 judge 或重试失败评分可复用已有 query 答案与真实上下文，不重复 Query。
- 2026-09-17 回归：`conda run -n agentic-rag pytest tests/unit/evals tests/integration/evals -q` → 90 passed；预检防重复 2 测试从缺模块 RED 到 GREEN。

- 语料输出：`var/artifacts/evals/real-corpus-v1-20260916/corpus/manifest.json`
- Manifest SHA256：`530123af07065eb4ca70878c512e9cf7500a6f9e0ec3489c8ee65e577519f9e9`
- 题库/源事实：`evals/datasets/real_corpus/source.json`，SHA256 `474d21ffedce4bef8e353cae25c103ac1907081785e908259182fc7da4f8cbef`
- 8 个文件的固定哈希：`evals/datasets/real_corpus/assets.sha256.json`；预览和校验结果在输出的 `qa/`，上传只取 manifest 的 documents 列表。
- 复用语料命令：`conda run -n agentic-rag python -m evals.corpus var/artifacts/evals/real-corpus-v1-20260916/corpus`

## 恢复规则

1. 逐项核对状态与实际文件；“测试通过”与“真实评测通过”分别记录。
2. 复用相同文件 SHA256 已完成的上传；失败任务记录原因，不能改变金标迎合输出。
3. 缓存键须包含语料/题库哈希、runtime snapshot、judge/metric 版本。不能只凭 case_id 跳过。
4. 每题必须落入成功、失败、超时、合理拒答之一；报告保留分母与缺失指标原因。
5. 不自动删除业务数据。评测资源保留以供复核，清理只针对账本中明确的评测资源。
