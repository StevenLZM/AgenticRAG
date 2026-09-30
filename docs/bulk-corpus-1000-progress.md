# 1000 份合成文档真实上传恢复记录

## 已批准范围

- 用户要求生成 1000 份各类 RAG 文档，经正常上传流程写入日常系统。
- 用户明确选择日常系统并确认地址 `http://localhost:8000`。
- 保留现有文档，不修改业务配置、不直接写 ES/MySQL、不覆盖旧测评语料。
- 生成 50 个虚构企业场景 × 2025/2026 两年 × 10 类文档：600 TXT、200 PDF、200 XLSX。
- 文件标注合成测试；保留结构化源事实用于后续金标建设。本轮不声称已有新金标或测评分数。

## 路径与执行边界

- 批次：`synthetic-business-1000-20260927-v1`
- 输出根：`var/artifacts/corpora/synthetic-business-1000-20260927-v1/`
- `documents/`：1000 份上传文件；`source.json`：源内容；`manifest.json`：文件 SHA256。
- `uploads/`：逐份 HTTP job/document/version 账本；`verification.json`：最终只读核验。
- 已检查 API runtime 可用，index_generation=`index-v3`，snapshot=`d0bd5b21d22f1e89c50b6837aeab2f826001ae918854506a89f29cecf829cd0e`。
- 服务检查日期：2026-09-27。恢复必须重新检查在线状态。

## 待办

- [x] 生成器与上传批处理回归测试（先 RED 后 GREEN）：12 focused passed；全仓 835 passed、47 skipped、18 warnings。
- [x] 生成源内容和 1000 份文件，校验数量、唯一性、跨文档引用、文件可读性：1000 unique SHA256、24,596,804 bytes、200 PDF pages、1000 Excel 公式缓存与独立计算一致。
- [x] PDF/Excel 渲染抽样检查通过；200 个 Excel 均生成预览，人工抽查首尾两种台账；PDF 两类抽样通过。未声称逐份人工审阅。
- [x] 小批次覆盖三种格式，确认真实上传完成且 ES 可见：2026-09-28 首10份全部completed，6TXT/2PDF/2XLSX，22 Parents/22 Children通过API/SQL/ES核验，见verification-10.json。
- [x] 限并发续传至1000份，不重复未知结果的POST；2026-09-30全量完成。
- [x] 按本批次document/version ID核对API、MySQL、ES；`verification-1000.json`全部通过。

## 恢复原则

先读取本文件、manifest、uploads 账本和正在运行的进程/会话。completed 复用；已有 job_id 只轮询；未知 POST 结果先查证，不能盲目重传。不得因 token/会话中断重新生成或重传整个批次。不得把 HTTP 202/排队算入库完成。

## 2026-09-27 阻塞点（待用户授权重启）

- 试上传在首组两份后停止，无后台批量上传继续运行。1 TXT completed，1 PDF failed，998 未提交。
- TXT `S001-2025-project`：document=`01a0e151-f4ba-7770-a916-c526cc8efa9c`，version=`01a0e151-f4bb-70fa-bf74-98e5cffd2b74`，job=`01a0e151-f4d5-7190-aa4f-f31215d80c70`。
- PDF `S001-2025-policy`：document=`01a0e151-f4bc-721e-9ab2-bc54f7d1a2bc`，version=`01a0e151-f4bd-7ac6-a0be-6a422037c93c`，job=`01a0e151-f4d4-7090-9c72-06380eedda31`。
- 精确读取该 PDF 的 `var/ingestion_checkpoints.sqlite`、thread=`ingestion:01a0e151-f4d4-7090-9c72-06380eedda31` 的 `__error__`：三次 `RuntimeError: Failed to load model ... docling-layout-heron ... [Errno 32] Broken pipe`。
- 同一PDF用当前Conda环境新进程 `DocumentConverter().convert(...)` 成功，17 text items。无业务代码/配置变更。
- 旧 ingestion worker PID 93380（wrapper 93375），已运行约14天，stdout/stderr为PIPE；API PID92564，未修改或重启。
- 需要用户授权仅重启 ingestion worker，并对已确认 failed 的 PDF 重新走上传接口。现有 API 没有 retry job 路由；不得直接改SQL状态。旧失败账本必须保留，不能盲目删除账本重复POST。未知提交与确认终态失败不同。
- resume 命令（解决失败账本并获授权后）：`PYTHONPATH=src:. conda run --no-capture-output -n agentic-rag python -u -m scripts.upload_bulk_corpus --stage upload --limit 10 --concurrency 2`；试批验证通过后去掉 `--limit 10`。
- 校验命令：同上 `--stage verify`；已完成TXT可用 `--stage verify --limit 1`。
- source.json 中 facts 是场景参数集合，并非声明所有参数都出现在每一份文档；未来金标必须依据每份实际内容标注。
- 已完成 TXT 经 `--stage verify --limit 1` 复核：API completed、SQL active version、1 active Parent、ES 1 active Child 一致。证据为批次目录下 `verification-1.json`。
- 独立只读审查无 critical/important；minor：JS 公式预检需未来增加 `Number.isFinite`，避免 NaN 与容差比较漏报。当前批次已通过独立 Python 重开所有 Excel 并逐个核对 1000 个数值缓存，未发现该问题。审查不替代真实上传/视觉核验。

## 2026-09-28 授权重启与续传

- 用户明确要求“先重启然后重新尝试上传文档”。仅重启 ingestion worker。
- 旧 PID93380 收到 SIGTERM 后仍存活；SQL 确认该用户无 queued/running 入库任务后 SIGKILL，未停止 API/ES/MySQL。
- 首次 nohup 启动 PID56772 未存活；没有重复受理任务。改用长运行工具会话44366，新 Worker PID57005，直接使用 agentic-rag Python，stdout/stderr写入 `var/log/ingestion-worker-20260928.log`，不再依赖会话 PIPE。存活状态和实际文件描述符需恢复时复核。
- 已再次通过 API 确认旧 PDF job failed，将原账本保留到 `upload-history/S001-2025-policy.failed-job-01a0e151-f4d4-7090-9c72-06380eedda31.json`；旧失败文档和任务保留，没有删除或直接改数据库。
- 首批10份重试会话73191，完成TXT会复用，不重传。新PDF任务身份以 uploads 最新账本为准。
- 新PDF job=`01a0e7af-96c8-74a4-a3b6-4c8b79514064`，document=`01a0e7af-96bf-7983-a6f4-ce5e88462251`，version=`01a0e7af-96c0-72b2-be38-9c54e7817075`；日志已显示正常加载模型、进入Torch首次编译，未再出现 Broken pipe。
- 恢复前重新检查进程和账本，不重复启动 Worker 或并行上传脚本。
- 新 Worker PID57005 存活，lsof确认 FD1/FD2 均为上述日志的 REG 文件。首个PDF重试 completed，第二个PDF也 completed；原TXT复用。
- 首批会话73191已退出1：第4份后发现 `S001-2025-operations` XLSX HTTP415 rejected（没有job_id），安全停批。当前批次3份completed（2PDF+1TXT）、1份rejected、996份未提交；另保留先前1份失败PDF历史任务。
- `verification-3.json` 于2026-09-28 11:06:43 UTC核验3份：API completed、SQL active、11 Parents、ES11 Children一致。不是1000份已完成。
- 新阻塞：API PID92564已运行约15天，早于09-16/17 XLSX校验修复。当前磁盘scanner对该XLSX accepted；仅在独立内存中还原旧 `_rels/.rels` 后缀判断即 rejected，理由unsupported_container/mime_signature_mismatch，与在线415一致。高度指向API未加载现有修复，不是生成文件损坏。
- 待用户授权重启API加载已有修复。未修改安全规则、未重启API、未绕过上传接口。获得授权后先检查活跃查询，正常停启API并持久化日志；确认后归档明确rejected账本再重试（没有受理job，仍保留历史）。

## 2026-09-28 API 重启

- 用户明确授权“重启api”。重启前SQL显示36 completed、1 cancelled，没有活跃查询。
- 旧API PID92564正常SIGTERM退出，端口8000释放。
- 新API启动会话69333，命令仍为 `scripts/run_api.py --grace-seconds 30`，stdout/stderr写入 `var/log/api-20260928.log`。未改配置或安全规则。
- 新API PID58512，`/health/ready`全部available，runtime snapshot与index-v3不变。
- 415账本归档至 `upload-history/S001-2025-operations.rejected-415-before-api-restart.json`。同一Excel重新经POST上传后已completed，job=`01a0e7b8-a668-7278-8976-8694aab2a0e9`，确认重启加载修复有效。
- 首批10份续跑会话49756；恢复先检查会话/账本，不重复提交。
- 会话49756已退出0，10份通过完整核验。随后启动全量续传（复用已完成10份），concurrency=2，运行日志 `upload-run-20260928.log`。全部完成后脚本自动写verification-1000.json；有失败自动停批，不把启动成功当成全量完成。
- 全量续传工具会话51880。API会话69333/PID58512；ingestion会话44366/PID57005。继续时先读日志、账本与会话状态；禁止重复启动批量上传。

## 2026-09-30 全量完成与正式处理核验

- 续传前重新检查在线服务并恢复正常API/Worker。已有836份completed复用，剩余164份走同一普通上传入口，concurrency=2；一次API重启产生的poll_failed按原job ID恢复，没有重复POST。
- 2026-09-30 06:51:29 UTC，1000/1000 completed，工具会话12394退出0；当前没有后台上传任务继续运行。
- `var/artifacts/corpora/synthetic-business-1000-20260927-v1/verification-1000.json`：API8000任务完成、SQL active version和Parents、正式ES index-v3的active Children一致；1000文档、2200 Parents、2200 Children；600 TXT、200 PDF、200 XLSX。
- 两份历史index-v2原文通过正式版本化reprocess接口迁移，保留旧版本与原文。加上3份历史active文档，完整快照共1003文档、2231 Parents、2237 Children，全部在ES9200当前Child索引可见。
- 完整生产分块重算审计及4453题金标位于`var/artifacts/evals/formal-corpus-gold-v2-20260930-production/`；原来的592组定位缺口全部修复。另发现1处真实Docling PDF阅读顺序错误，金标保留unmapped，详见`docs/formal-document-pipeline-repair-progress.md`，未掩盖或删除该题。
- 正式API会话53434（`var/log/api-20260930.log`）、ingestion Worker会话47017（`var/log/ingestion-worker-20260930.log`）；未来操作前仍需实时核对健康状态，不因历史会话号推断存活。
