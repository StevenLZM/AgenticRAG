# 信息来源路由实现与验收记录

2026-10-01。本次解决非知识库请求误入 RAG、缺少外部能力仍升级 Research，以及缺少真实路由回归三项问题。代码已接线，真实模型分类门槛通过；真实服务端到端验收尚未完成，未部署或重启生产服务。

## 实现与决策

- 单次 Router 输出 RouteAssessment，服务端 `decide_route` 决定路径。能力描述来自实际组合根，当前没有实时/外部查询执行器，不能由用户、记忆或模型自行开启。工具权限、RBAC 与 MCP credential/scope 管理延期。
- Fast RAG 和研究后的 Evidence Grader 共用 `decide_grade`。区分可继续检索的 missing_facts 等缺口、需要澄清、外部能力缺失和技术失败；外部能力缺失不再启动 Research，仍保留正常文档问答的研究与引用审计。
- 最近上下文读取同 user/thread、当前 Run 创建时间之前已完成的公开问答，固定时间、最多 6 条/8000 字符；不依赖生产尚未填充的 messages 表。原始 query 保留，normalized_query 供检索和推理。新 Run 重置旧路由、记忆缓存与证据；空证据包必须符合当前索引代次的 PackedEvidence 契约。
- Chat 能力说明由固定模板生成，不再调用回答模型。内部证据不丢失，但 Worker 不把无关 parent 引用附加到 Chat。事件记录 initial_route、最终 route、executed_path 和有界 gap_type/response_mode，UI 随实际路径更新。
- 保留 v1 Prompt/schema 供旧调用方；生产组合根传入 v2 能力策略。Prompt 加载器支持安全的版本化文件名，仍拒绝路径穿越。研究重入防护位于主图节点，不改变独立 ResearchAgentLoop 的工具契约。

## 自动化验证

解释器为 `/Users/steven/miniconda3/envs/agentic-rag/bin/python`，以下命令在仓库根目录执行。

```sh
python -m pytest tests/unit/query tests/unit/runtime tests/unit/observability \
  tests/unit/evals/test_routing.py tests/integration/persistence/test_sqlite_checkpoint.py \
  tests/integration/persistence/test_conversation_context.py \
  tests/integration/runtime/test_query_worker.py tests/integration/api/test_query_runs.py -q
python -m pytest -q
python -m mypy src/agentic_rag/query src/agentic_rag/runtime src/agentic_rag/persistence/conversations.py
python -m pytest tests/e2e/test_capability_routing.py -m 'e2e and live_model' -v -rs
```

专项回归与全量复跑结果在最终审查后更新。首次全量测试发现旧组合根 Fake Router 输出 v1，以及新 Run 重置时使用空字典导致研究包校验失败；已更新测试并以失败用例驱动修复，保留拒绝损坏 checkpoint 的校验。Ruff 对本次变更相关文件检查通过，退出码 0。

Mypy 退出码 1：9 个错误位于本次未修改的 `ingestion/models.py:120,124`（7 个 kwargs 类型问题）和已有未提交评测改动的 `retrieval/graph.py:283,287`（2 个类型问题）；本轮涉及的 query/runtime/conversations 文件没有报告新错误。未替用户修改这些无关内容。

真实服务 E2E 收集 6 项，全部 skip（pytest 退出码 0，不视为验收通过），缺少显式 `AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1` 测试 opt-in。测试还要求独立本地 MySQL/Redis/ES、测试 admin DSN 和已有 backup/restore opt-in。没有从生产配置推定测试清理权限。新 fixture 复用隔离生命周期和真实组合根，但不沿用直接种子向量或 Mem0 写入 smoke；真实上传文本天气报告及两份简历，经摄取发布后运行查询，查询记忆策略 disabled。误分类后的检索止损另由确定性图契约测试验证，不冒充真实分类命中。

## 真实模型分类对照

```sh
python scripts/eval_routing.py --dataset evals/datasets/routing_v2.jsonl \
  --repeats 3 --output-dir /private/tmp/agentic-rag-routing-eval.yJIuaI
```

退出码 0，状态 PASS。产物为该目录内 `samples.jsonl`、`manifest.json`、`report.json`、`report.md`；启动日志在同级 `agentic-rag-routing-eval.yJIuaI.log`。首次命令在没有安装源码包的环境发现导入路径缺失，补上项目 src 路径后重跑；首次未产生任何模型样本。另有不依赖安装包的 CLI help 回归。

请求模型 `deepseek-v4-flash`，360 次响应实际模型标识均为 `deepseek-flash`；protocol=auto，SDK retry=0，Gateway retry=2，client timeout=30s，并发 1。两版共享相同 question/history/time/capabilities，按 case/repeat 交错；旧 Prompt 未指导使用新增上下文字段，这属于受控输入对照，不是旧生产请求原样回放。数据集从第一次真实请求前冻结，无删除难例或改标签。

| 指标 | 旧版 | 新版 |
|---|---:|---:|
| 有效样本 | 180/180 | 180/180 |
| 综合正确数（含响应模式与指代实体） | 118/180，65.56% | 180/180，100% |
| 无需 KB 却检索 | 59/93，63.44% | 0/93 |
| 需要 KB 却不检索 | 0/87 | 0/87 |
| 澄清率 | 0/180 | 15/180，8.33% |
| 8 个核心用例各 3 次 | 未通过 | 24/24 |
| 分类平均耗时 | 1517.53 ms | 1272.93 ms |

| 用例组 | 旧版正确数 | 新版正确数 |
|---|---:|---:|
| 通用与会话 | 22/30 | 30/30 |
| 外部实时与查询 | 0/30 | 30/30 |
| 知识库单次与多步 | 60/60 | 60/60 |
| 多轮指代与主题切换 | 27/30 | 30/30 |
| 混合需求与边界 | 9/30 | 30/30 |

每类结果从相同 samples 重新聚合，未请求额外模型。该成绩只反映冻结题集上的分类表现，不是生产置信保证；分类平均耗时不是端到端首 token 延迟。本轮没有测量上线后的首 token 改善幅度。

版本证据：

- canonical dataset SHA256：`9cc8aca9a9f44d1297ada117ac147c441a79b05e6fb0ea76523467b678057edb`
- JSONL file SHA256：`c60006f73878a8ec70ad8ed4f871c25a8ae16e749f388f8728bc3b6f92ca41bf`
- router_v1 SHA256：`c9787f995de34d0fcfbfff181fed57d1e123f25cefdef1f7beaf846c4b88b83d`
- router_v2 SHA256：`fd3c32d5de3e46d0df87f937f8a28ce1ac46a672f42de5db4d785451ac219e43`

## 发布边界

没有新增天气或联网工具，没有修改 PDF/chunker，没有重建 ES 或 MySQL Parent，没有业务表结构迁移。未授权的工具/MCP 权限治理继续待办；现有数据权限和引用审计不放宽。正式切换须先补齐隔离端到端验收、排空旧活动 Run、核对 query-v2/prompt-v2/routing-v2 快照，在明确授权后重启 API 与 Worker，步骤见[本地运行说明](local-operations.md#信息来源路由回归与切换)。
