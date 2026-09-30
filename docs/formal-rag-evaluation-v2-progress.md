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
- [ ] 2. 正式v2命令、冻结快照预检、隔离记忆、可恢复查询账本。
- [ ] 3. 原始Child排名、实际上下文、最终输出与Agent轨迹。
- [ ] 4. 分阶段/分粒度排名、AND/OR事实覆盖与多跳完整性。
- [ ] 5. 盲抽取关键字段、真实Answer/Factual Correctness及ContextRecall。
- [ ] 6. 失败不剔除、评分覆盖率、Agent诊断、分组与报告。
- [ ] 7. 可配置候选预算、dev/validation/test隔离调参与Pareto选择。
- [ ] 8. 全仓回归、真实分层smoke、独立审查与恢复说明。

## 接口裁决

- 新v2模型/结果与旧24题评测并存，不将旧Parent二元口径冒充新Child口径。
- 冻结gold.v2.2文件不覆盖；缺少人工graded qrels时，graded NDCG为null，不凭自动匹配编造等级。
- 关键字段抽取不接收金标值；规则验证与LLM事实评分独立。
- 正式快照完整性校验覆盖全部干扰文档，不只核对命中的正例。
- 未知POST/裁判响应不能自动重复收费调用；恢复必须复用run_id或明确对账。

## 当前恢复点

开始任务1。先写反例测试，再实现；每个已验证增量更新本账本。
