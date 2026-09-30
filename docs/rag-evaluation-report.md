# 真实 RAG 测评报告

核验日期：2026-09-18。结论：**8文件→真实上传/入库→24题HTTP Query→真实Ragas→报告核验已经完成；效果未全面达标，不能视为生产发布验收通过。**

## 范围与证据

- 合成中文语料8份：2 PDF、4 TXT、2 XLSX；题库24题：12单跳、8多跳/比较/计算、4无答案。39条源事实先冻结，再匹配真实文档版本和AST，不从召回结果反推gold。
- 本机隔离API `127.0.0.1:8001`、ES `127.0.0.1:9201`；索引`agenticrag-children-eval-e7a49c260917`，21个Child；MySQL `agentic_rag_eval_e7a49c260917`、Redis DB1、专用user/Mem0/checkpoint。不是生产机业务数据评测，不修改业务9200别名。
- 真实Qwen `text-embedding-v3`/1024维；judge `deepseek-v4-pro`；Ragas0.4.2。真实Query/Worker/Graph/reranker/生成均使用项目主链路。
- runtime snapshot：`a4e9f142025f5d2b523ebfa1741ee63225b9e46dbf6a32fab5ec95c5857899f4`。
- manifest SHA256：`530123af07065eb4ca70878c512e9cf7500a6f9e0ec3489c8ee65e577519f9e9`。
- 产物根目录：`var/artifacts/evals/real-corpus-v1-20260916/`。`corpus/`是实际上传文件，`uploads/`包含8个job/document/version ID，`gold-mapping.json`保存事实与Parent绑定；`evaluation/queries/`包含24个HTTP Run ID，`runtime-evidence/`保存逐Run核验哈希。

## 最终成绩

| 指标 | 均值 | 有效样本数 |
|---|---:|---:|
| Parent Recall@6 | 1.0000 | 16 |
| MRR@6 | 1.0000 | 16 |
| 二值NDCG@10 | 1.0000 | 16 |
| 最终上下文证据覆盖率 | 1.0000 | 20 |
| Ragas Faithfulness | 0.9833 | 20 |
| Ragas Answer Relevancy | 0.9366 | 20 |
| Ragas Context Precision | 0.8683 | 20 |
| 无答案题正确拒答率 | 0.2500 | 4 |

24题全部有真实评分，评分不可用/失败0、Query执行错误0。业务终态：20正常回答、1 `cannot_answer`、3 `research_round_limit`。耗尽轮次是质量失败，不是运行错误；不能用“零执行错误”替代正确率。

排名指标使用真实hydrated Parent列表，不是答案引用列表；16个样本是“可回答且实际只检索一轮”的题，不等同于12个标注单跳题。多轮排名保留dense/BM25/RRF/rerank各阶段顺序，不拼接伪全局MRR。无答案题不套Recall/NDCG；20个可回答题另统计最终证据覆盖。小规模合成语料的满分召回不代表生产准确率100%。

同一16题的分阶段Parent排名（Child按首次出现去重成Parent，保持原顺序；从已核验的retrieval_rounds重算，不调用模型）：

| 阶段 | Recall@6 | MRR@6 | NDCG@10 |
|---|---:|---:|---:|
| Dense | 1.0000 | 0.9583 | 0.9637 |
| BM25 | 1.0000 | 1.0000 | 0.9950 |
| RRF | 1.0000 | 0.9688 | 0.9769 |
| Rerank | 1.0000 | 1.0000 | 1.0000 |

## 低分与后续产品优化

- 21（实际生产上线日期）、23（GPU型号）、24（法定代表人）：文档无答案，系统最终显示“已达到全局研究轮次上限”，没有明确给出资料不足说明，correct_refusal均为0。
- 22（2027营业收入）：明确资料不足，correct_refusal为1。
- 18（上海住宿标准跨年比较）：Faithfulness=0.6667；15的Context Precision=0。保留原回答/上下文/评分，后续复核judge判定与检索上下文，不修改gold或覆盖本轮成绩。
- 推荐下一轮单独优化“证据缺失时及时停止研究并明确拒答”，然后使用新的实验目录做A/B；本轮不为了提高分数修改生产Graph。

## 核验与限制

`verify_saved_evaluation`只读重新检查冻结语料、上传任务、版本内事实映射、scoped SQL/checkpoint、公开答案/上下文/真实排名、结果指纹、证据哈希及汇总。不会重新Query或judge。评分行是受本地权限保护的实验记录，不是提供商签名的防篡改收据。

本次只读核验快照保存在`evaluation/verification-details.json`，包含逐题Run ID、终态和耗时；需判断当前状态时应重新执行下方命令，不能仅凭旧快照放行。

SQL started_at/finished_at可重新采集逐Run耗时；本轮约10.34–168.86秒，不含外部Ragas评分耗时。每Run模型token用量未持久化，明确为`null/unavailable`，不能计算可信token成本或事后补0。旧summary的`unaudited_answer_count=4`包括4个没有展示生成答案的安全终态，不表示泄漏了4个未审计草稿；`citation_coverage=0`是最差样本硬门禁，不是引用平均准确率。

本轮未执行备份/恢复演练、生产鉴权/RBAC对抗、大规模负载或真实业务语料泛化验证。其门禁保持未通过，不把测试skip或配置flags当成演练证据。

## 工程交付验证

- 相关eval/retrieval/query/ingestion/safety/API/legacy/文档回归：528 passed、2 skipped。
- 运行时及SQLite checkpoint回归：81 passed、2 skipped。
- 指向本轮真实数据的只读E2E：1 passed；Ruff和`git diff --check`通过。
- 独立只读审查通过legacy退役、fixture分类、E2E接线和runner恢复保护；saved-verifier由另一代理交叉审查并实际live验证，不把首次因额度失败的审查称为通过。
- 审查发现并修复了subset覆盖失败记录的缺陷；成功/失败case均受保护，空选择不能清报告，完整重试成功才清对应失败，扩展题集中断不丢后续已评分行。

这些是本轮受影响范围的测试，不是全仓所有测试或生产发布门禁全通过。工作树保留原有未提交修改，没有自动提交、合并或删除评测资源。

## 复核命令

在仓库根目录运行，保留既有隔离MySQL与checkpoint：

```sh
PYTHONPATH=src conda run -n agentic-rag python -m scripts.verify_acceptance \
  --report var/artifacts/evals/real-corpus-v1-20260916/evaluation/summary.json \
  --run-dir var/artifacts/evals/real-corpus-v1-20260916 --quality-only

AGENTIC_RAG_EVAL_RUN_DIR=var/artifacts/evals/real-corpus-v1-20260916 \
conda run -n agentic-rag pytest --import-mode=importlib tests/e2e/test_query_evaluation_api.py -q
```

`--quality-only`成功只证明测评执行和来源核验完成；去掉该参数，本轮发布硬门禁应失败。旧`run_real_query_acceptance.py`已停用，退出2且不创建资源；确定性fixtures只保留协议测试标记、真实测评计数0。禁止使用子集覆盖已有全量成绩；需要新实验使用新目录。评测资源和全部未提交改动保留，未自动合并、提交或清理。
