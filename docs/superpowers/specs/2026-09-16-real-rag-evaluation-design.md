# 真实上传到 RAG 评分的闭环设计

日期：2026-09-16。用户已批准 8 份文件、24 道题和隔离运行方案，并要求记录可恢复待办后逐项执行。

## 目标与边界

业务评测仅接收真实部署 HTTP API 的运行结果。保留确定性数学/接口单元测试，不把 fake client、固定 embedding 或参考答案回填包装为 real provenance。使用合成非个人敏感语料；不改写现有业务文档，不涉及本轮之外的流式输出功能。

## 数据集

8 份中文 TXT/PDF/XLSX：虚构企业的项目政策、产品规格、季度/部门数据以及相似但不同条件的干扰内容。24 题：12 单跳、8 多跳/比较/计算、4 文档无法回答。答案与来源锚点在运行查询前冻结。每条 gold 包含 case_id、question、reference_answer、answerable、expected_route、source anchors、tags；无答案题相关 Parent 集合为空，不将 NDCG/Recall 的不适用值伪造成 0 或 1。

上传后以 document/version 与 source anchor/AST span 对应实际 Parent，标注绑定过程不使用查询返回的排名。记录语料、题库与映射的哈希。生成 PDF/XLSX 时使用适用文档技能检查渲染/数值；不伪造摄取产物。

## 执行隔离与真实链路

启动独立端口的 API、Ingestion Worker、Query Worker。采用独立 MySQL 数据库、Redis 空间、user_id、index_generation、Mem0 collection、artifact/checkpoint 路径。旧任务/业务 Worker 不得消费此栈队列。资源名称进入运行账本，不修改全局 `.env.local`、不关闭业务服务。若当前凭据无权创建隔离资源，只报告所缺授权，不改用业务数据混跑。

每份文件经 `POST /v1/documents` 上传，轮询 `/v1/ingestion-jobs/{job_id}` 至真实完成，确认文档版本、Parent/Child 与活动索引属于评测 scope。真实查询经 HTTP Query API，由现有 Worker 执行，不使用 ASGI fake transport、不直接调用模拟 Graph。

预检补充：当前 ES 发布流程使用共享活动别名 `agenticrag-children-active`，新代际仍可能重指业务别名。因此采用独立 ES 实例优先；如需复用集群，必须先有独立别名配置及验证。未满足这一约束不得执行评测上传/发布。

## 评测契约

- CLI 移除 fixture；业务默认且正式入口为 API。Graph 客户端如用于内部验证，不自动获得真实端到端认证。
- 检索日志分别记录实际候选/融合/重排/Parent/最终上下文的顺序和关联 query/round，不把最后回答中引用过的 Parent 冒充检索 Top-K。
- 主指标明确为指定阶段的 Parent Recall@6、MRR@6、二值 NDCG@10。NDCG ideal denominator 使用 `min(K, total_relevant)`，与实际返回长度无关。多跳另报最终证据覆盖，不将多轮列表无定义拼成单轮排名。
- 分题记录实际 route、answer status、引用、contexts、run_id、snapshot、输入语料版本、耗时、tokens；失败/超时也保留，不仅统计成功题。
- 保留现有公共 API 安全边界；评测采集通过服务器端按 run/user/snapshot 限定的可信 collector 读取必要运行产物，不开放匿名任意文档读取接口。

## Ragas

配置真实 judge 与所需 embeddings，明确固定依赖及模型配置；传递 question/user_input、response、retrieved_contexts、reference。首轮运行 Faithfulness、Response Relevancy、Context Precision；无答案/拒答题用独立正确拒答判定，不强行套全部 Ragas 指标。缺依赖/配置立即预检失败；单题 judge 失败记录错误、有限重试，不能返回虚构分数。区分“离线批评测”与“不能联网”：真实 judge 允许调用已配置的模型服务。

## 持久化与恢复

准备、上传、gold 绑定、查询、judge、汇总各阶段记录状态；最小重试单位为阶段/案例。断点键包含 dataset/corpus hash、runtime snapshot、judge/metric 版本。逐题结果原子落盘，报告列出总题数与成功/失败/超时/拒答数。所有执行器均先写运行账本，避免中断后重复创建/上传/计费。

## 验收

先完成 3 题预检（单跳、多跳、无答案），检查端到端来源与评分输入；再跑全量 24 题。报告必须附实际 HTTP run IDs、上传 job IDs、各阶段指标、Ragas 成功率/有效样本数、失败原因与配置版本。低分如实报告，不修改 gold，也不把“跑完”称为“效果达标”。不删除业务数据；评测资源默认保留用于复核。
