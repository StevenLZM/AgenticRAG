# IK 金标重绑定与真实 BM25 对照 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 证明现有金标与 v4 语料等价，冻结真实分词环境，完成可恢复、可审计的 standard/IK 配对 BM25 诊断。

**Architecture:** 保持 GoldCaseV2 和现有评分器兼容，用独立审核文件保存迁移证明、词法环境与实验身份。直接调用真实 ElasticsearchBm25Index，通过只读审计代理记录请求；不用 Query API，也不伪装成完整问答测评。

**Tech Stack:** Python 3.11/3.12、现有 Pydantic/SQLAlchemy/Elasticsearch SDK、pytest/pytest-asyncio；不新增运行依赖。

**Spec:** `docs/superpowers/specs/2026-10-03-ik-gold-retrieval-evaluation-design.md`（2026-10-03 用户已确认）。

## Global Constraints

- 本计划待审阅；尚未实现、生成 v4 金标或运行对照。所有下述新接口、命令都是待实现接口。
- 在 `codex/real-rag-evaluation` 实现，不在 main 修改业务代码。批准基线为 `60d543c7c8d181fffce11f8cc5572d5277a0d0fc`；不自动提交、推送、重置或清理用户文件。
- 不上传、重分块、重嵌入，不写 SQL/ES 业务数据、不切 alias、不改变生产配置、不重启服务。可丢弃集成测试仅操作其专属测试资源。
- 旧输入保持字节不变；新绑定和新实验使用独占新目录，目录 0700、文件 0600。公开摘要不含个人原文、凭据或带认证的 URL。
- 第一批：dev 五类各 20 题，共 100 题；协议 `ik-bm25-paired-v1`；top_k=40，并发 1，检索请求总预算 200，每请求 30 秒，累计检索阶段墙钟预算 900 秒。历史尝试也计入预算。
- 两臂同一物理 v4 索引、语料、查询、作用域与实现；只变 lexical profile 及其字段/权重/查询结构。原文 query 不改写，不增加公司/年份过滤。
- 不提交 Query POST，不调用 Router、Dense、Embedding、RRF、rerank、生成或 Judge；未执行阶段标记 `not_executed`。
- 始终 `eligible_for_tuning=false`、`production_release_accepted=false`；不代签人工审核，不以本批替代自然问法、校准、validation/test。
- 失败/unknown 不静默删除；unmapped 保持 null；不输出超过 40 的 K，不将未标注命中全部认作无关，不用历史答案回放冒充 IK 成绩。

## Review Focus

1. v3/v4 同时被修改后彼此仍相等：必须同时匹配迁移 journal 冻结哈希，不能只做双边比较（任务 1）。
2. 相同数量/ID 下向量、发布状态、manifest 非代际字段或干扰文档变化：阻止等价重绑定，旧标签不自动修补（任务 1、2）。
3. 多节点中只取到一个节点的词典，或在线词典/文件证据无法绑定实际进程：明确 unverified，不靠版本号和探针放行（任务 3）。
4. 请求已发送、结果落盘前崩溃，或 SDK 暗中重试：保留 uncertain 与预算，不重发、不重置账本（任务 4）。
5. 单侧成功、返回不足 K、部分 graded qrels、拒答证据被召回：报告适用范围与有效配对分母，不能误算完整问答或等级排名质量（任务 5）。

---

## 文件边界与执行准备

新增模块均放在 `evals/`，避免把评测配置暴露到公共 Query API：

| 文件 | 单一职责 |
| --- | --- |
| `evals/migration_evidence.py` | 独立捕获物理索引、SQL、原文件/AST/manifest，验证迁移等价和内容漂移 |
| `evals/gold_rebinding.py` | 保持原题/sidecar 非绑定字段，发布经过完整验证的新输入包 |
| `evals/lexical_environment.py` | 实际 ES 环境、探针、节点文件证明和前后漂移核验 |
| `evals/paired_retrieval_identity.py` | 固定抽样、共同/两臂身份和显式差异白名单 |
| `evals/paired_retrieval_ledger.py` | 独占目录锁、持久化尝试、限额和恢复验证 |
| `evals/paired_bm25.py` | 真实 adapter 的审计代理和顺序配对执行 |
| `evals/paired_retrieval_report.py` | 纯检索逐题、分层、配对与失败覆盖报告 |
| `evals/ik_evaluation_command.py` | `rebind`、`bm25`、`resume` 的薄编排入口 |

复用且不降低契约：`evals/gold_v2_models.py`、`gold_v2_validation.py`、`gold_answer_spec.py`、`formal_gold.py`、`retrieval_metrics_v2.py`、`report.py` 和 `src/agentic_rag/retrieval/adapters/elasticsearch.py`。不修改 `tuning._comparable`、旧 resume/replay 或现有正式问答报告结构。

本地测试命令中的 `PY` 须替换为 `/Users/steven/miniconda3/envs/agentic-rag/bin/python`，每次执行使用 `PYTHONPATH=src:.`。这是文档缩写而非可执行命令名。`CorpusSnapshot/GoldCaseV2` 来自 `evals.gold_v2_models`，`GoldAnswerSpec` 来自 `evals.gold_answer_spec`，SDK 响应 `ObjectApiResponse` 来自 `elastic_transport`，其余库类型使用现有依赖。

- [ ] 执行任务 1 前重新查看 `git status --short`、`git worktree list --porcelain`、附加 worktree 及分支祖先关系。当前只读结果为评测分支落后 main 14 个提交、没有独有提交。
- [ ] 按 using-git-worktrees 流程复用或准备隔离工作区，绑定用户指定的 `codex/real-rag-evaluation`；仅在祖先关系仍成立时 `git merge --ff-only 60d543c7c8d181fffce11f8cc5572d5277a0d0fc`。若分支已变化，停止同步并解释差异，不自动 merge/rebase。保留 main 中本次文档及既有 `tmp/`。
- [ ] 用文件编辑工具把经批准的设计、计划和恢复点带入执行工作区，不复制凭据或大语料。CLI 显式接受原工作区的 `--settings-file`、`--artifact-root` 和输入绝对路径；通过 Settings 读取配置但不记录密钥、不构建 LLM 客户端。

## Task 1: 全语料迁移证明与只读内容捕获

**Files:** 新建 `evals/migration_evidence.py`；测试 `tests/unit/evals/test_migration_evidence.py`。参考 `scripts/migrate_ik_index.py:100`、`:118` 和 `scripts/export_formal_eval_snapshot.py:20`，不调用迁移 apply/rollback/preflight（后者要求目标不存在）。

**Interfaces:**

- `async capture_corpus_evidence(template: CorpusSnapshot, *, engine: AsyncEngine, es: AsyncElasticsearch, artifacts: LocalArtifactStore, index: str, generation: str) -> tuple[CorpusSnapshot, dict[str, Any]]`：独立读取指定物理索引，不用旧 alias 验证来阻止合法代际迁移。
- `async capture_migration_evidence(old: CorpusSnapshot, journal: dict[str, Any], *, engine: AsyncEngine, es: AsyncElasticsearch, artifacts: LocalArtifactStore) -> tuple[CorpusSnapshot, dict[str, Any]]`：读取旧/新物理索引与当前 SQL/Artifact，返回目标快照及原始取证摘要。
- `verify_migration_equivalence(old: CorpusSnapshot, target: CorpusSnapshot, journal: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]`：成功返回审核对象；失败抛 ValueError，原因含确切字段/ID，不含原文。
- `assert_corpus_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None`：比较内容指纹，排除采集时间而不排除业务字段。

`evidence` 固定包含 `protocol="ik-migration-evidence-v1"`、`journal_sha256`、`source_inventory`、`target_inventory`、`parents_inventory`、按版本的 `originals/asts/manifests`、`alias_before/alias_after`、`content_fingerprint`、采集方法和时间。full-source 按 ID 的摘要另存，向量不得出现在可读报告；使用 migration 原有序列化方式复算 journal 的聚合哈希，不混用不同 hash 算法。

非秘密 JSON、字节 SHA 与参与实现文件清单均可审计；规范化白名单命名 `ik-rebind-normalization-v1`。原始/目标清单的字段差异必须逐项列出，禁止用忽略所有 metadata 的方式通过。

- [ ] **写失败测试：** `test_generation_only_is_equivalent` 断言仅白名单变化通过；参数化 `test_same_ids_do_not_hide_mutation` 修改正文、向量、AST、Parent、user、active、manifest parser 字段、增删干扰文档各自拒绝。
- [ ] **写审计反例：** `test_both_indices_changed_must_match_journal` 令两边同改向量且保持计数，仍因 journal 哈希不符拒绝；`test_capture_alias_change_or_partial_scan_rejected` 验证 alias 漂移、重复 ID、分片失败和 scan 中断都不能产出 verified。

```python
# test_both_indices_changed_must_match_journal 的核心断言；fixture 含已冻结 journal。
assert evidence["source_inventory"] == evidence["target_inventory"]
with pytest.raises(ValueError, match="journal"):
    verify_migration_equivalence(old, target, journal, evidence)
```

- [ ] **运行 RED：** `PYTHONPATH=src:. PY -m pytest tests/unit/evals/test_migration_evidence.py -q`；预期因缺少新接口失败，记录失败原因。
- [ ] **实现捕获：** SELECT 全量有效文档/版本、完整 Parent 行；ES 扫描 `_source` 全字段，包括向量，验证 `_id==id`、用户/文档/版本/父关系、active/search_type/generation。复制原快照字段布局生成 v4 `CorpusSnapshot`，附 alias 完整定义；额外全源证明保存在 evidence。完整扫描只固定物理索引，清理 scroll，关闭 SDK 自动重试。
- [ ] **实现等价核验：** 旧快照投影必须与保留的 v3 实况一致，v3/v4 完整 Child 库存和 SQL Parent 都必须匹配 complete journal；全量文档及版本集合匹配旧快照。验证全部原文/AST 实际 SHA；读取前后 manifest，仅允许 `index_generation` 差异，路径/哈希必须等于 journal 的 before/after。scope 仅容许物理索引/UUID 的声明迁移，alias 过滤/路由与用户语义不能变化。全部证据前后重复读取核对；无法建立一致视图即拒绝，不自动暂停服务。
- [ ] **运行 GREEN 并记录：** 上述测试与 `tests/unit/evals/test_corpus_snapshot_v2.py` 全通过；进度记录实际命令、结果和剩余项，不提交 Git。

## Task 2: 不改语义的 gold 与 sidecar 发布

**Files:** 新建 `evals/gold_rebinding.py`；测试 `tests/unit/evals/test_gold_rebinding.py`。

**Interfaces:**

- 消费任务 1 的 `verify_migration_equivalence(...)` 和现有 `load_gold_v2`、`load_gold_answer_specs`、`verify_originals`、`bind_anchor`。
- `rebind_gold_rows(rows: list[dict[str, Any]], *, snapshot_id: str) -> list[dict[str, Any]]`：按原顺序只更新 snapshot_id。
- `rebind_spec_rows(rows: list[dict[str, Any]], *, cases: list[GoldCaseV2]) -> list[dict[str, Any]]`：只更新 snapshot_id/gold_case_sha256。
- `publish_rebound_bundle(*, source_snapshot: Path, source_gold: Path, source_specs: Path, journal_path: Path, target: CorpusSnapshot, evidence: dict[str, Any], output: Path) -> dict[str, Any]`：只写新目录，返回完整 `rebind-audit.json` 的内容。
- `load_rebound_bundle(root: Path) -> tuple[CorpusSnapshot, list[GoldCaseV2], dict[str, GoldAnswerSpec], dict[str, Any]]`：验证 complete 标志、每个文件 SHA、源输入绑定及全部模型校验后才供 runner 使用。

- [ ] **写失败测试：** `test_only_binding_fields_change` 对 raw JSON 去绑定字段后逐行全等（包括可选字段缺省与顺序），sidecar 金标哈希按当前 `canonical_hash(case.model_dump(mode="json"))` 重算；`test_sidecar_semantic_drift_is_not_a_rebind` 改 claim/condition/rule/review_status 必须拒绝。
- [ ] **写恢复与充分性测试：** `test_incomplete_or_existing_bundle_refused`，`test_unmapped_and_splits_preserved`，`test_cross_child_coverage_and_scope_revalidated`；AND 集合缺一 Child 不可通过，错误版本/父定位拒绝，unmapped 不自动转 mapped。

```python
# test_only_binding_fields_change；每个 fixture 原始字段的存在性也须保留。
rebound = rebind_gold_rows(rows, snapshot_id=target.snapshot_id)
assert [r["case_id"] for r in rebound] == [r["case_id"] for r in rows]
for before, after in zip(rows, rebound, strict=True):
    assert after == {**before, "snapshot_id": target.snapshot_id}
```

- [ ] **运行 RED：** `PYTHONPATH=src:. PY -m pytest tests/unit/evals/test_gold_rebinding.py -q`。
- [ ] **实现迁移和验证：** 保留 raw 行的字段存在性；先加载旧绑定，再验证新绑定及原文/算式。对每个事实用既有 `bind_anchor` 的文本与 AST 范围规则复核原标注的充分集合，语义集合采用集合比较、持久化仍保留原顺序；不能用新检索结果造 qrels。sidecar 与确定性构造器及原 sidecar 非绑定字段都一致，否则停止。
- [ ] **实现原子发布：** mkdir(exist_ok=False)，先 `bundle-state.json` incomplete，再写 `snapshot.json`、`gold.v2.2.jsonl`、`answer-spec.jsonl`、`migration-evidence.json`、`rebind-audit.json`；重新加载全部文件并校验源字节未变后写 complete。复用 atomic_write_text，校验输出不等于或覆盖任一输入/既有目录。失败保留受限 incomplete 目录与原因，禁止 runner 接纳。
- [ ] **运行 GREEN 并记录：** 新测试及 `test_gold_v2_validation.py`、`test_gold_answer_spec.py` 全通过。审计输出题目/事实、split/category/mapping 计数与每题非绑定哈希；真实 4453/12105/1 为验收基线，不写成只支持该数量的模型限制。

## Task 3: 词法环境清单与节点文件证明

**Files:** 新建 `evals/lexical_environment.py`；测试 `tests/unit/evals/test_lexical_environment.py`。

**Interfaces:**

- `async collect_local_node_evidence(node: dict[str, Any], *, es: AsyncElasticsearch) -> dict[str, Any]`：node 固定字段 `provider="native_local"/node_id/node_url/es_home/config_dir`，可选 package_path；通过 ES 响应及本机进程证明绑定，只读，无安装或修改行为。未知 provider 返回 unverified，本批不实现远程 SSH 或 Docker 执行器。
- `async capture_lexical_environment(*, es: AsyncElasticsearch, index: str, alias: str, nodes: list[dict[str, Any]], implementation: dict[str, str]) -> dict[str, Any]`：返回清单及 `verified/missing_evidence/environment_sha256`。
- `assert_lexical_environment_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None`：内容寻址排除 observed_at，但比较所有影响检索的内容与取证节点集合。

清单为 `lexical-environment-v1`；分离稳定内容与采集时间/请求计数，避免每次采集时间变化导致假漂移。所有有可能处理该索引分片的相关节点均须证明，动态分配新增节点会使后验核验失败。

- [ ] **写失败测试：** `test_same_versions_different_dictionary_changes_identity`；`test_missing_node_or_process_binding_is_unverified`；`test_remote_dictionary_without_loaded_state_is_unverified`；`test_default_remote_switch_without_sources_is_not_remote_usage`；`test_config_override_and_dictionary_modified_after_start_rejected`；`test_probe_equality_cannot_replace_artifact_hash`；`test_capture_time_not_content_drift`。保持 ES/plugin 版本字符串不变也必须检出文件变化。

```python
# test_probe_equality_cannot_replace_artifact_hash；mock ES 只用于验证拒绝契约。
result = await capture_lexical_environment(
    es=es, index=index, alias=alias, nodes=[], implementation=implementation)
assert result["verified"] is False
assert "node_file_evidence_missing" in result["missing_evidence"]
```

- [ ] **运行 RED：** `PYTHONPATH=src:. PY -m pytest tests/unit/evals/test_lexical_environment.py -q`。
- [ ] **实现环境捕获：** ES info/nodes/plugins、物理索引 mapping/settings（含 defaults）、alias 定义与 UUID。记录实际标准根字段、zh 字段的 index/search analyzer、similarity、分片/路由配置；只排除显式说明的运行统计量，不忽略整个 settings 对象。
- [ ] **实现文件取证：** 当前文档为本机原生 ES 部署。通过节点直连 `_nodes/_local` 的 ID/进程信息、cluster UUID、本机监听 PID 及启动 home/config 参数交叉验证，不信任操作者填写的 node_id。只读实际安装的全部 JAR、descriptor、权限文件、有效 XML、内置/扩展/停用词典，记录相对路径、size、SHA 和配置来源。处理目录扩展和 symlink 的解析目标，越出明确授权的安装/config 根或无法读取时 unverified；文件在进程启动后变化也不可直接认定已加载一致。
- [ ] **实现配置解析：** 按安装制品对应实现确定 ES config 下插件覆盖目录与 JAR 所在目录/config 的优先级，不能只散列其中一个。分别记录 `remote_allowed`（8.19.0 默认 true）和 `remote_sources`（有效远程 URL 列表）；启用开关但无来源不是实际远程词典。已核验的仓库 ZIP SHA 为 `bcacbcea8dab5555cc8ad4bc59c0403cd583ed1ee169b7764e5bbb81464c7673`，仅是安装包线索；必须再与运行安装制品核对。未知实现/优先级或有效远程来源非空时本批 strict unverified，不安装、重启或下载远程词典来补证明。
- [ ] **实现固定探针：** 协议 `ik-probes-v1`，五个输入固定为 `中华人民共和国劳动合同`、`星河智采量子枢纽`、`Redis缓存A17`、`2026年10月3日报销上限560元`、`研究研究生生命起源`；逐一 standard/ik_max_word/ik_smart，共 15 次 analyze，保存 token/position/start_offset/end_offset 与哈希。探针用于诊断，不替代进程/文件证明。
- [ ] **运行 GREEN 并记录：** 新测试全通过。采集请求统一 30 秒超时，单次预检阶段墙钟上限 300 秒、ES 元数据/扫描/探针请求合计上限 250；显式分开 audit 与 retrieval 计数，预算耗尽输出未验证而不放宽。

## Task 4: 配对身份、真实请求账本与零检索恢复

**Files:** 新建 `evals/paired_retrieval_identity.py`、`evals/paired_retrieval_ledger.py`、`evals/paired_bm25.py`；测试分别为 `tests/unit/evals/test_paired_retrieval_identity.py`、`test_paired_retrieval_ledger.py`、`test_paired_bm25.py`。

**Interfaces:**

- `select_paired_cases(cases: list[GoldCaseV2], *, per_category: int = 20) -> list[GoldCaseV2]`；类别固定为 `section_comprehension/single_hop/table_calculation/unanswerable/multi_hop`，按名称排序，类内按 `sha256(("ik-bm25-paired-v1:" + case_id).encode()).hexdigest()`、case_id 排序，固定 dev，不足或出现未知类别拒绝，不静默改变样本量。
- `build_paired_identity(*, bundle_audit: dict[str, Any], environment: dict[str, Any], cases: list[GoldCaseV2], config: dict[str, Any], implementation: dict[str, str]) -> dict[str, Any]`；`assert_paired_comparable(left: dict[str, Any], right: dict[str, Any]) -> None`。
- `PairedLedger.open(root: Path, identity: dict[str, Any], *, resume: bool) -> PairedLedger` 为上下文管理器；`start(request: dict[str, Any]) -> str` 返回 attempt_id；`finish(attempt_id: str, outcome: dict[str, Any]) -> None`；`records() -> list[dict[str, Any]]`；`checkpoint_elapsed(seconds: float) -> None`。
- `AuditedSearchClient(es: AsyncElasticsearch, ledger: PairedLedger, *, snapshot: CorpusSnapshot, case: GoldCaseV2, arm: Literal["standard", "ik"], identity: dict[str, Any])`，提供 adapter 使用的 `async search(**kwargs: Any) -> ObjectApiResponse`。该代理记录并转发真实 SDK kwargs，按全语料快照验证命中后才写成功 FINISH，返回原响应，不造排名；不能要求命中只来自该题金标文档。
- `async run_paired_bm25(*, es: AsyncElasticsearch, snapshot: CorpusSnapshot, cases: list[GoldCaseV2], identity: dict[str, Any], ledger: PairedLedger) -> list[dict[str, Any]]`。

identity 固定协议、输入文件/快照/语料/环境哈希、代码提交及参与模块文件哈希、脱敏配置、case_ids/query/filter、top_k/timeout/budgets、两臂 profile/DSL 模板与独立指纹。敏感配置不落盘；无关未执行的模型阶段明确标识，不调用旧全问答 runtime 来产生假身份。

- [ ] **写失败测试：** `test_sample_is_fixed_dev_20_per_category`；`test_pair_allows_only_declared_lexical_changes` 参数化改变 user、generation、doc filter、query、top_k、budget、model 配置、代码、gold、case 顺序均拒绝；声明差异只能是 profile 和 `bool.must` 的标准/IK 查询及其派生哈希。不得跳过整个 opaque provider/runtime fingerprint。
- [ ] **写请求与账本反例：** `test_adapter_kwargs_equal_saved_request`，`test_native_tied_order_and_hit_scope_preserved`，`test_crash_after_send_is_uncertain_without_retry`，`test_sdk_retries_are_disabled`，`test_missing_or_tampered_ledger_refused`，`test_complete_resume_adds_zero_searches`。query API/model 边界装拒绝调用的 spy，断言计数为 0。

```python
# test_complete_resume_adds_zero_searches；预检读取另计，不能混入 BM25 计数。
before = search_spy.call_count
rows = await run_paired_bm25(es=es, snapshot=snapshot, cases=cases,
                            identity=identity, ledger=resumed_ledger)
assert search_spy.call_count == before
assert rows == completed_rows
assert query_post_spy.call_count == model_spy.call_count == 0
```

- [ ] **运行 RED：** `PYTHONPATH=src:. PY -m pytest tests/unit/evals/test_paired_retrieval_identity.py tests/unit/evals/test_paired_retrieval_ledger.py tests/unit/evals/test_paired_bm25.py -q`。
- [ ] **实现身份：** 显式冻结所有非秘密配置及组件哈希，分别重算共同和两臂指纹。实现清单包括本计划新增模块、现有 gold/schema/validation/anchor/metrics 模块、ES adapter/filter/mapping 模块及 Python/ES SDK 版本，保存文件路径→SHA；不能仅记录 HEAD 而漏掉未提交代码。A 为当前根字段 match，B 为当前 best_fields、zh^2/root^1、tie_breaker=0；记录 effective operator=OR（即使 adapter 未在 DSL 写明）。校验不是任意 DSL；保持现有 adapter 代码和默认值不变。
- [ ] **实现持久化：** 使用独立 flock，ledger 文件含版本、identity hash、单调序号/hash-chain 的事件数组；每次 START/FINISH 复用原子写入并 fsync。START 在 I/O 前持久化实际 request、attempt、预算扣减；FINISH 保存成功原始 hits/失败原因、响应摘要、took/耗时/状态。无结果 START 视为 uncertain；完成结果与 ledger 双向校验，不能通过删文件重置额度。多进程同时 resume 拒绝。
- [ ] **实现预算/恢复：** 异步全阶段 deadline + 每请求 timeout，SDK `max_retries=0,retry_on_timeout=False`。累计墙钟含之前运行片段，持久化片段起止；进程崩溃导致无法确认片段时长时保守记满该片段剩余预算并停止新增请求，不能以重启重置 900 秒。failed/uncertain 本批无自动重试开关；人工同意重试仍须新 attempt 和显式新预算审批，不能覆盖历史。正常耗尽保留 pending。
- [ ] **实现真实执行：** 显式构建 SearchFilter 与固定物理索引的 ElasticsearchBm25Index，注入上述只读代理。偶数题 A→B，奇数题 B→A，并发 1；每个 (case,arm) 只发一次。拒绝 duplicate/foreign ID、错 parent/version、非有限 score、超窗口命中、timed_out 或 failed shards；保留 ES 顺序，不按得分或 ID 重排。代理仅接受预期的 BM25 search，禁用别的查询/写入能力。
- [ ] **运行 GREEN 并记录：** 三个新测试文件及 `tests/unit/retrieval/test_elasticsearch_search.py`、`tests/unit/evals/test_tuning_v2.py` 全通过；完整 resume 即使无待执行请求也必须校验账本和前后内容/环境，不提前短路。

## Task 5: 明确分母的纯检索报告

**Files:** 新建 `evals/paired_retrieval_report.py`；测试 `tests/unit/evals/test_paired_retrieval_report.py`。

**Interfaces:**

- 消费 `run_paired_bm25` 的真实 records 及 `PairedLedger.records()`（同一 record 结构，runner 直接返回该结构）。
- `build_paired_report(*, cases: list[GoldCaseV2], records: list[dict[str, Any]], identity: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]`。
- `render_paired_report(report: dict[str, Any]) -> str`；不接收现有 formal answer report 类型，不调用 Pareto tuner。

每条已完成 record 至少有 case_id、arm、attempt_id、identity_sha256、request/query/filter/index/size、ordered hits（child/parent/document/version/user/rank/score）、status、elapsed_seconds；缺失字段不是默认成功。verification 必含 pre/post corpus 和 lexical 核验状态。

- [ ] **写失败测试：** `test_cross_child_and_complete_fact_metrics` 使用 AND={c1,c2}、OR={c3} 的充分集合验证前缀；`test_short_results_do_not_shrink_qrel_denominator`；`test_partial_pair_has_no_verified_delta`；`test_unmapped_is_null_not_zero`；`test_partial_graded_labels_are_na`；`test_refusal_evidence_is_not_answer_correctness`。

```python
# test_cross_child_and_complete_fact_metrics；单事实的充分集合为 [[c1,c2],[c3]]。
partial = score_retrieval_stage(case, ["c1"], ks=(1, 3, 5, 10, 20, 30, 40))
complete = score_retrieval_stage(case, ["c1", "c2"], ks=(1, 3, 5, 10, 20, 30, 40))
assert partial["metrics"]["fact_recall@40"] == 0
assert complete["metrics"]["fact_recall@40"] == 1
assert complete["metrics"]["mrr"] == 0.5
```

- [ ] **运行 RED：** `PYTHONPATH=src:. PY -m pytest tests/unit/evals/test_paired_retrieval_report.py -q`。
- [ ] **实现指标：** 复用 score_retrieval_stage，显式传 `ks=(1,3,5,10,20,30,40)`，mrr 命名 `mrr@40`，注明其为首次完整命中某事实充分集合的倒数排名。binary NDCG 是已标注证据命中诊断，不声称无标签即真实不相关。本批无独立完整等级审核证明，graded NDCG 统一 null/`not_available`，不因局部 qrels 自动启用。拒答题评的是拒答依据召回，不是系统是否正确拒答。
- [ ] **实现配对/汇总：** 逐指标给 B−A 和 improve/equal/regress（固定绝对容差 1e-12）；仅两侧成功且前后核验通过才算 verified pair。条件均值显示有效 n；另给 `[0,1]` 指标保守下界/上界，未知值分别取 0/1，分母为全部请求题目，明确只是界限不是把 unmapped 判错。按 category 和来源格式分组并列宏平均，跨格式题计为独立 `mixed` 类避免重复计数；组为空或无有效值为 null，不当 0。
- [ ] **实现可读报告：** 记录 100 题/200 预期请求、实际尝试/成功/失败/uncertain/pending、有效配对和 metric 覆盖率、各组耗时、输入身份、解析缺陷及缺证原因。只披露案例/证据 ID 和脱敏诊断；所有非执行阶段 not_executed，两项资格始终 false。并列排序波动、本地延时非生产 P95、模板题非自然中文代表性三项限制写入报告。
- [ ] **运行 GREEN 并记录：** 新测试及 `tests/unit/evals/test_retrieval_metrics_v2.py` 全通过；快照测试确保无 AnswerCorrectness/Ragas/完整 RAG 伪分数。

## Task 6: 一条命令编排、真实验收与恢复交付

**Files:** 新建 `evals/ik_evaluation_command.py`；测试 `tests/unit/evals/test_ik_evaluation_command.py`、`tests/integration/evals/test_ik_paired_bm25.py`；更新 `docs/formal-rag-evaluation-v2-progress.md` 和 `docs/local-operations.md`，只追加新入口与已验证结果。

**Interfaces:** `main(argv: list[str] | None = None) -> int`；`async run(args: argparse.Namespace) -> int`。子命令固定：

- `rebind` 必填 settings-file、artifact-root、es-url、user-id、source-snapshot、source-gold、source-specs、journal、output-dir。
- `bm25` 必填 settings-file、artifact-root、es-url、user-id、bundle、node-evidence-config、output-dir；默认且首批固定 20/类、40 候选和上述预算，不接受自定义模型/任意 DSL。
- `resume` 必填 settings-file、artifact-root、es-url、user-id、experiment、node-evidence-config；从既有身份恢复其他参数，禁止替换金标、重抽样、加预算。

`node-evidence-config` 是显式节点 ID→只读取证位置/方式的 JSON，不含凭据；由执行时只读部署核对生成在私有实验输入目录，路径必须来自实际部署，不能猜。CLI 缺参拒绝，没有 9201/allocation fallback。读取原 Settings 后显式覆盖 artifact_root/es_url，核验 user/generation/cluster 与输入一致；不输出完整 Settings。

- [ ] **写失败测试：** `test_cli_has_no_legacy_or_paid_fallback`、`test_preflight_failure_sends_no_bm25`、`test_postflight_drift_invalidates_pairs`、`test_resume_rechecks_even_when_finished`、`test_no_corpus_writes_or_query_posts`、`test_exit_codes_and_private_outputs`。用小内存语料只测协议，不标为真实质量。

```python
# test_preflight_failure_sends_no_bm25；fixture 将文件取证设为 unverified。
assert await run(args) == 1
assert search_spy.call_count == query_post_spy.call_count == model_spy.call_count == 0
```

- [ ] **运行 RED：** `PYTHONPATH=src:. PY -m pytest tests/unit/evals/test_ik_evaluation_command.py -q`。
- [ ] **实现编排：** rebind 调任务 1/2；bm25/resume 先读绑定包，前验任务 1/3，冻结任务 4 身份/账本，执行检索，后验任务 1/3，最后任务 5 报告。已绑定包在 strict 输入不齐时拒绝，不先跑后补证；finally 关闭连接/scan，保留失败账本并输出未完成报告。exit 0=工程流程完整且严格核验通过（不是质量上线通过），1=运行/核验不完整，2=输入/恢复契约拒绝。
- [ ] **实现 opt-in 集成：** 仅 `AGENTIC_RAG_TEST_IK_ELASTICSEARCH_URL` 显式指向可丢弃 IK 节点，且确认与正式 cluster UUID 不同，才允许建立 `ik-eval-test-<uuid>` 专属索引；只清理本测试创建的准确索引名。验证真实 adapter 两臂 DSL、scope、native rank、持久化和恢复零 search。集成不具备节点文件证明时只测诊断分支，strict 环境证明由实际部署验收；不能伪造 verified。
- [ ] **运行 GREEN 和全仓回归：** `PYTHONPATH=src:. PY -m pytest tests/unit/evals tests/unit/retrieval/test_elasticsearch_search.py -q`，再 `PYTHONPATH=src:. PY -m pytest -m 'not integration and not e2e and not live_model' -q`；对本计划文件边界表的八个 Python 文件及本计划列出的九个新测试文件执行 `PY -m ruff check`，另执行 `git diff --check`。在确认可丢弃节点后单独执行新 integration 文件；未运行/skip 不能写成通过。
- [ ] **执行只读真实重绑定：** 使用下列命令形式。执行前将输出日期/后缀改为当日唯一目录，确认已批准本批实现和这组只读预算；路径存在就停止，不覆盖。核验 1003 原文、2231 Parent、2237 Child、4453 题、12105 事实组、1 unmapped，split 2670/890/893。任一计数变化先解释内容差异，不改断言求通过。

```bash
PYTHONPATH=src:. /Users/steven/miniconda3/envs/agentic-rag/bin/python -m evals.ik_evaluation_command rebind \
  --settings-file /Users/steven/LzmWorkSpace/AgenticRAG/.env.local \
  --artifact-root /Users/steven/LzmWorkSpace/AgenticRAG/var/artifacts \
  --es-url http://localhost:9200 --user-id default_user \
  --source-snapshot /Users/steven/LzmWorkSpace/AgenticRAG/var/artifacts/evals/formal-corpus-gold-v2-20260930-production/snapshot.json \
  --source-gold /Users/steven/LzmWorkSpace/AgenticRAG/var/artifacts/evals/formal-corpus-gold-v2-20260930-production/gold.v2.2.jsonl \
  --source-specs /Users/steven/LzmWorkSpace/AgenticRAG/var/artifacts/evals/formal-v2-grounded-spec-20261001/answer-spec.jsonl \
  --journal /Users/steven/LzmWorkSpace/AgenticRAG/var/migrations/ik-v4-20261002/journal.json \
  --output-dir /Users/steven/LzmWorkSpace/AgenticRAG/var/artifacts/evals/formal-v4-rebound-20261003
```

- [ ] **执行真实 100 题 BM25：** `bm25` 复用相同四个 settings/artifact/es/user 参数，另传 `--bundle <上一步目录> --node-evidence-config <已核验的私有节点配置绝对路径> --output-dir <新的绝对实验目录>`。冻结代码后不得边跑边改；缺少安装制品/词典证明则停在预检并交付缺项，不自行安装/重启或降级为 verified。
- [ ] **执行完整 resume：** `resume` 复用四个配置参数并传 `--experiment <刚完成的实验目录> --node-evidence-config <同一配置>`；必须重新核验前后环境/内容，新增 BM25 search=0，Query POST=0，模型调用=0，历史 attempt/结果 hash 不变；预检读取与 15 个分词探针按阶段另计。保存 `audit-first.json`/`audit-resume.json`，明确计数边界。
- [ ] **验收与交付：** 保存报告绝对路径、实际分数/有效样本数、退出码、请求数、耗时、前后核验和源文件 SHA。不预写“IK 提升”；若阻断，写明最后完成任务/具体缺项，不把模板数据或单元测试当实测补齐。由独立审查者检查整批代码后处理发现，再重跑相关回归；保持不自动提交/推送。

## 自检与恢复约定

计划自检：设计的金标/环境/配对/恢复/指标/验收分别落在任务 1–2/3/4/4/5/6；五类 Review Focus 均有具名反例。共享输入为现有模型和明确 JSON 字段，新增函数签名在使用前定义；只读 adapter 代理避免生产行为改动，旧正式测评契约不放宽。

每完成一个任务，勾选实际完成步骤并在进度文档追加测试证据。半途停止记录分支、HEAD、工作区、执行到哪一步、输出目录及账本状态；恢复先验证这些身份，不重复已完成真实请求。代码修订改变实现身份时创建新实验，不原地恢复旧成绩。

人工金标审核、至少 100 例事实级裁判校准、自然语言题/困难负例、Dense/RRF/rerank/上下文、候选数调参、890 validation/893 test 和聊天/工具场景依然未完成。本计划完成也不能把这些项目勾为完成。

## 审批与执行方式

- [x] 用户确认书面设计（2026-10-03）。
- [x] 保存本实施计划并完成主代理自检。
- [ ] 用户审阅本计划及首轮只读运行预算。
- [ ] 用户选择执行方式：Native（建议，接口耦合较紧，由当前代理顺序实施并在末尾独立审查）或 Subagent-driven（逐任务实现与独立审查，额外上下文成本更高）。
- [ ] 开始任务 1；当前尚未修改业务代码、生成绑定或执行新测评。
