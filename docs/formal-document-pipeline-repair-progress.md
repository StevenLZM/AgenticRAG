# 正式文档流水线与金标定位修复（2026-09-30）

用户要求：解决592组Parent-only金标定位，所有文档按正式上传/解析/分块/发布模式处理。不是执行整套答案评分器改造，不删除业务原文或旧版本，不直接向ES/SQL伪造分块。

## 已完成

- [x] 根因核验：592事实组中589为表格表示差异，另3涉及段落/标题。837份当前文档原本就由正常ingestion-v2/Docling/HybridChunker/index-v3生成，无测试专用分块。
- [x] 金标定位使用实际Child的ast_locator覆盖范围和真实heading/context；支持充分证据集合（集合内AND、集合间OR），不拼接有缺口的范围。不修改源文档、Parent或Child文本。
- [x] 正常原文重处理入口 `POST /v1/documents/{document_id}/reprocess`：同一ID/原文hash/filename、新版本/任务/Outbox、正常Worker；保留旧有效版本直到Publisher切换。用户隔离、行锁、重复pending拒绝、不可替换原文。
- [x] 两份legacy原文通过正式API8000重处理completed；最新账本位于 `var/artifacts/evals/formal-corpus-gold-v2-20260930-production/reprocessing/`。没有删除旧版本或原文。
- [x] 三份failed历史上传原文哈希均已有active成功副本；保留失败记录，不重复提交制造额外文档。deleted文档不恢复。
- [x] 最终全仓测试：859 passed、47 skipped、18 warnings（27.99秒）；本轮涉及文件的ruff全部通过。独立代码审查无遗留Critical/Important。
- [x] 抽样5份实际上传的AST，用当前同一生产ChunkingPipeline重新计算，完整Parent/Child产物一致。

## 全量完成结果

- [x] 2026-09-30 06:51:29 UTC正常批量上传1000/1000 completed，上传脚本退出0；`verification-1000.json`确认API/SQL/ES一致，600 TXT、200 PDF、200 XLSX，2200 Parents/2200 Children。
- [x] 完整生产快照：1003份active原文全部searchable（1000合成+3历史），2231 Parents/2237 Children；统一docling-v1/ingestion-v2/text-embedding-v3/index-v3；当前无processing。保留3份历史failed和1份deleted。
- [x] 正式流水线逐份审计：原文/AST/manifest哈希、当前生产ChunkingPipeline重新计算的完整Parent/Child对象、SQL/ES字段与ID/数量全部一致；最大Child为348 tokens（上限384），没有测评专用分块器。
- [x] v2.2金标4453题、12105事实组；12104 mapped、parent_only=0、unmapped=1。所有1003原文均覆盖；原文、唯一ID、用户/版本与Child/Parent绑定复核通过。
- [x] 原版592组逐条全部修复；历史重处理事实按同一document ID、问题、原文claim/anchors关联，未以旧version ID误判丢失。证据见`gold-binding-verification.json`的592条关联记录。
- [x] 更新完整语料README及原始测评实施方案；未声称人工语义审核或新端到端答案评分已经完成。

最终目录：`var/artifacts/evals/formal-corpus-gold-v2-20260930-production/`。
snapshot ID：`9679120983ec8b0ae213c190dc273997ffc1d7ab7f72d5437549cbd5413c406c`。
gold SHA256：`7066e120bc11b836d2e99d9522e5420cf8f3b77e949e9d1965a75edb5cf2bfb0`。

## 新发现的独立解析缺陷（保留真实结果，未修复）

- `S047-2026_知识服务产品说明.pdf`，document=`01a0f105-04b2-7e9a-a3ac-3d99d532386c`，version=`01a0f105-04b3-7430-b0fd-e71c8722e249`。
- 金标事实`S047-2026-product:section-4:F3`：“产品介绍不承诺未列明的GPU型号、供应商报价或外部客户上线日期。”
- Docling将第一行末尾的“介”分离为`#/texts/15`，bbox位于同一正文行右侧（x=530.014647..540.014647）；正文`#/texts/12`已有多行且缺该字，body阅读顺序将这个孤立字排在关联文档之后。Canonical AST忠实保留该结果，分块器不是原因。
- 使用相同原文在新Docling进程中转换仍复现，因此不是旧Worker的Broken pipe，也不能靠盲目重复上传解决。
- 本轮未改Docling阅读顺序或硬编码该字，也未删题、缩短锚点掩盖缺陷。故`fully_mapped=false`是诚实状态，不能宣称全部qrels完备。该题仍是原文可回答题，不能改成无答案。
- 后续若修复正式解析器，须先用此PDF及多栏/表格/扫描件做失败回归测试，通用解析修复后经正式reprocess入口发布新版本，另建快照重新绑定；不能直接修SQL/ES或覆盖本冻结产物。

## 进程/恢复

只读检查进程和日志后再操作；不重复启动worker、不重复POST。重处理账本completed复用，未知响应先核对。

- 正式API：长运行工具会话53434，`scripts/run_api.py --grace-seconds 30`；日志 `var/log/api-20260930.log`。
- 入库Worker：会话47017，`scripts/run_ingestion_worker.py`；日志 `var/log/ingestion-worker-20260930.log`。
- 批量续传会话12394已退出0，无后台上传继续；日志 `var/artifacts/corpora/synthetic-business-1000-20260927-v1/upload-run-20260930.log`。已有completed会复用；API重启导致一次poll_failed停批，已按原job ID恢复。
- 两份重处理脚本会话53969；日志 `var/log/formal-document-reprocessing-20260930.log`，两份已completed。
- 生产只读审查formal_pipeline_review已完成，59 focused tests passed；反馈问题修复后全仓859通过。审查不替代实际全量流水线审计。

## 本轮已执行命令（冻结文件已存在，不要原目录重跑）

```sh
PYTHONPATH=src:. /Users/steven/miniconda3/envs/agentic-rag/bin/python -m scripts.export_formal_eval_snapshot --output-dir var/artifacts/evals/formal-corpus-gold-v2-20260930-production
PYTHONPATH=src:. HF_HUB_OFFLINE=1 /Users/steven/miniconda3/envs/agentic-rag/bin/python -m scripts.audit_formal_document_pipeline --output-dir var/artifacts/evals/formal-corpus-gold-v2-20260930-production --expected-active 1003
PYTHONPATH=src:. /Users/steven/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m scripts.build_formal_eval_gold --output-dir var/artifacts/evals/formal-corpus-gold-v2-20260930-production
```

生产expected_active=1003（1000合成+3历史active）是本轮已核对范围，不适用于未来新增文档；未来应新快照、新目录。当前snapshot和gold生成器拒绝覆盖旧产物，已存在时先核验，不能删文件重跑掩盖漂移。
