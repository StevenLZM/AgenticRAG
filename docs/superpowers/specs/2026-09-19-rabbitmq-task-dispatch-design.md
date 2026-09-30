# RabbitMQ 异步任务分发升级：讨论草案

日期：2026-09-19。状态：待用户讨论确认，尚未授权按本草案实施。

用户已明确：MQ 在项目外通过本地 Docker 启动，由项目连接使用；当前没有既定 MQ。本文建议 RabbitMQ + Quorum Queue。多 Worker 与共享 checkpoint 的配套范围尚待确认，不能将本草案当作已经通过的实施规格。

## 1. 目标与成功标准

替换 Query Run 和 Ingestion Job 的 Redis Stream 分发，形成可验证的发布、认领、执行、重试、恢复和死信闭环。不能只替换 Broker 客户端后保留 Redis 的认领语义。

可靠性承诺：至少一次消息投递、数据库授权的任务执行、受租约代际约束的业务提交。外部模型请求和节点内副作用可能因崩溃重放，不承诺端到端 exactly-once。

若本次包含多 Worker，完成标准还必须包括两个独立进程竞争消费、失败接管、共享 checkpoint 恢复和过期 Worker 写入隔离；单进程模拟测试不能替代这些验收。单节点 RabbitMQ 不作为集群高可用证明。

## 2. 当前源码证据

| 位置 | 当前机制 | 本次含义 |
|---|---|---|
| `persistence/redis_queue.py` | XADD / XREADGROUP / XAUTOCLAIM / XACK；Lua 发布去重 | RabbitMQ 不应伪装成支持 XAUTOCLAIM 的 StreamBroker |
| `persistence/outbox.py` | 发送后标记 dispatched，仅捕获 RedisError | 改成 Broker 无关的错误分类和确认契约 |
| `persistence/repositories.py` | Run 与 Outbox 同事务；任务租约；Outbox 使用 next_attempt_at 临时占用 | 保留原子创建，补全发布租约的 owner/token/过期检查 |
| `runtime/query_worker.py` | 一批通知逐个执行；DB claim；心跳；终态后 ACK | 改成有容量上限的消费循环；明确所有投递分支的处置 |
| `runtime/query_worker.py::_invoke_with_heartbeat` | 构造 new_query_state 后调用 ainvoke(state, config) | 当前入口没有显式区分新 Run、同 Run 恢复、已完成 checkpoint；须专项验证 |
| `ingestion/graph.py::run` | 查询 snapshot.next，恢复时更新 owner/generation 并 ainvoke(None) | 保留恢复意识，进一步检查新旧 owner 的并发写入边界 |
| `persistence/checkpoint.py`、`config.py` | SQLite + 每类 worker_count=1 | 不能仅放宽配置便宣称多 Worker 安全 |
| `persistence/artifacts.py` | LocalArtifactStore | 同机可共享受控目录；跨主机接管需要共享 Artifact 存储 |
| `runtime/concurrency.py` | asyncio.Semaphore | 限额属于单进程，不是集群全局额度 |

以上为本轮静态检查，不代表已执行故障验证。

## 3. 选型建议

推荐 RabbitMQ：本项目主要是耗时任务通知，手动 ACK、发布确认和消费预取与需求匹配；Python 采用 aio-pika 接入 asyncio。RocketMQ 在已有相关基础设施时是可行候选。Kafka 已有 Share Groups，不能以“不支持队列”排除，但当前项目没有必须新增事件日志平台的要求。

开发环境使用带 management 的 RabbitMQ 4.x 镜像，落实实施规格时固定精确版本及镜像摘要，应用不启动或删除用户的 Broker。独立 vhost、应用用户和持久化卷由运维步骤配置；连接凭证仅从配置注入。

Quorum 队列可在本地单节点演练。生产故障容忍需要至少三节点与合适的副本分布；由单节点扩为集群后，还必须检查队列成员是否实际扩充。

## 4. 职责和消息拓扑

链路：API -> MySQL 任务 + Outbox -> Dispatcher -> RabbitMQ -> Worker -> MySQL Claim -> Graph -> 提交结果 -> ACK。

MySQL 是任务、执行权、重试时间和业务终态的事实源。MQ 是通知与背压通道。checkpoint 是 Graph 进度事实源。SSE 仍从持久化事件读取，不改成直接订阅任务队列。

建议固定拓扑（每个部署使用独立 vhost；测试使用隔离命名空间）：

| 类型 | 名称 | 用途 |
|---|---|---|
| Durable direct exchange | `agenticrag.tasks` | 投递任务 |
| Durable quorum queue | `agenticrag.query` | routing key `query` |
| Durable quorum queue | `agenticrag.ingestion` | routing key `ingestion` |
| Durable direct exchange | `agenticrag.dead` | 业务死信和 Broker 隔离消息 |
| Durable quorum queues | `agenticrag.query.dead` / `agenticrag.ingestion.dead` | 分类型保存死信，无自动回主队列环路 |

业务重试以 SQL 到期调度 + Outbox 实现，不依赖 TTL 重试队列或延迟插件。Broker 死信用于坏消息和反复异常重投的隔离，不能替代业务失败状态。

主队列配置可靠死信策略：`dead-letter-strategy=at-least-once`、`overflow=reject-publish`，绑定有效 DLX，验证所需 feature flags。Broker delivery-limit 独立于业务执行次数，并显式配置；达到该限制不代表任务已经业务失败，必须通过隔离记录与对账处理。不得将默认丢弃行为当成可靠死信。

消息采用有版本的 JSON 信封：schema_version、event_id、task_type、task_id、dispatch_generation、created_at、trace_id。AMQP message_id=event_id。消息不携带 prompt、文档正文、模型凭证或可执行客户端对象。任务所有者和配置快照从 DB 回查，不信任消息中的权限声明。

Broker 不保证按 message_id 去重。相同 Outbox 事件重发时 event_id 稳定；下一次合法业务调度使用新的 event_id 和 dispatch_generation。AMQP delivery tag 仅在所属 channel 内有效，不能当成持久化任务 ID。

## 5. 发布与认领契约

### Outbox

任务创建与首条通知同事务提交。Dispatcher 用短事务领取到期 Outbox，记录 publisher owner、单调 lease token、lease_until，再在事务外发送。

只有 publisher confirm 成功且 mandatory 路由没有返回消息，才以匹配的 owner/token 标记 dispatched。NACK、不可路由、连接中断和确认超时均不得标记成功。确认不明允许重发同一事件。

发布租约过期后允许另一个 Dispatcher 接管，旧 Dispatcher 不能覆盖新 owner 的发布状态。发布尝试数独立于业务执行次数；长期无法投递应告警并保留记录，不能静默删除。

### Worker

消费前获取本进程执行容量；prefetch 与容量配合，不无限预取。收到消息后严格校验信封、回查任务并原子 Claim。认领与续租统一使用 DB 时间。

需区分：dispatch_generation（通知有效性）、claim_generation（执行权 fence）、execution_attempt（真正开始的执行预算）、publish_attempt（通知发送尝试）。连接波动导致的重复消息不得消耗业务执行预算。

DB Claim 返回结构化结果，而不只返回 None：

| 结果 | 处置 |
|---|---|
| claimed | 运行任务；保持消息未 ACK |
| terminal | 直接 ACK，不重跑 Graph |
| stale_dispatch | 直接 ACK；新代际已由持久化 Outbox 承担通知责任 |
| busy / not_due | 只有 DB 中任务及恢复/到期调度责任成立才 ACK；由 Recovery Scheduler 兜底，不立即 NACK 循环 |
| unknown / invalid | 可靠隔离并告警，禁止无限重投和执行 |
| DB unavailable | 不宣称完成；暂停消费并退避恢复，必要时关闭 channel 让投递重回 Broker |

busy 分支 ACK 的前提是本方案同时交付可靠 Recovery Scheduler；不能脱离该组件单独移植此行为。

## 6. 执行、重试和死信

成功：DB 中以活跃 owner/claim_generation 提交答案或入库终态，事务提交后 ACK。若 ACK 丢失，再投递读取终态并 ACK，不再次生成答案。

可重试执行失败：在同一 DB 事务中结束当前执行租约、记录 next_attempt_at、推进 dispatch_generation，并插入唯一的下一轮 Outbox；提交后 ACK 当前消息。若事务失败，不 ACK。下一轮到期再发送，不能依靠进程 sleep 维持重试可靠性。

耗尽执行预算或确定性失败：同事务提交失败终态、失败原因与死信 Outbox；提交后 ACK 原消息。DLQ 暂时不可用不撤销已提交失败状态，Dispatcher 后续补投。DLQ 消费/重放按 event_id 幂等，重放须经过业务入口校验，不能把旧消息直接复制回主队列。

clarify、refuse、cannot_answer、audit_failed、research_round_limit 等现有业务终态保持业务含义，不按基础设施异常盲目重试。

取消：持久化取消请求；尚未执行的任务可以原子结束；执行中由心跳观察并停止图执行。取消与成功竞争通过条件更新收敛为一个终态，不能生成互相矛盾的结果。

Recovery Scheduler 周期扫描：过期运行租约、到期重试、长时间未被认领的 queued 任务、未完成的发布/死信意图。每次恢复必须在锁定任务后复核状态，并以唯一调度代际生成通知，多个 Scheduler 不重复产生同一轮重试。

DB 已提交但消息丢失或误隔离时，对账可修复通知；Broker 故障不应丢失业务调度事实。对账频率与超时应可配置，避免把正常队列等待误判成执行失败。

## 7. 超时、失联与进程生命周期

分别配置 AMQP 连接心跳、DB 任务租约、执行心跳、任务截止时间、ACK 超时、优雅退出宽限期。连接活着不代表任务在进展；保持 DB 心跳也不代表任务可以无限运行。

Query 当前默认 300 秒执行超时。升级后把端到端执行截止时间持久化，明确它是否包含排队，不能每次接管都获得全新完整预算。建议执行截止时间从首次执行开始计算，排队另设过期策略；Ingestion 设独立、有限且可配置的执行时限。

ACK 超时覆盖执行上限和短暂清理时间，但不能依靠无限 ACK 超时解决卡死。文档解析、重排等阻塞计算不得阻塞承担 AMQP/租约心跳的事件循环。

AMQP channel 丢失时旧 delivery 无法在新 channel ACK；取消关联执行，停止续租并交由恢复流程处理。租约不明时禁止写业务终态。所有 Graph 任务应在 shutdown 内有界等待或取消，停止消费先于关闭连接，不能退出时盲目 ACK。

## 8. 多 Worker 配套：建议纳入本次，待整体审阅

推荐纳入同机两个独立 Worker 的认领与故障接管验证。当前 SQLite 单 Worker 限制不能简单删除。方案选择需要同时满足：

1. 所有 Worker 读取相同的已提交 checkpoint；保留完整 State、pending writes、版本与父子 checkpoint 关系。
2. checkpoint 写入也校验执行权，不能只在最终答案提交时检查租约。
3. 过期 Worker 暂停后恢复运行，不能污染新 Worker 可恢复的 checkpoint。
4. 恢复时读取当前 Run 的 checkpoint：无 checkpoint 才初始化；同 Run 有未完成节点时恢复；同 Run 已完成图时从保存结果完成业务提交；旧会话 Run 的 checkpoint 不可误认成当前 Run。
5. 同一用户会话维持单活动 Run 约束；权限命名空间和运行时快照继续校验。

共享数据库并不自动满足第 2、3 项。推荐复用现有 MySQL，采用带任务租约校验的 saver 适配层，不额外引入 PostgreSQL。这样 checkpoint 写入与任务执行权检查可在同一个数据库事务里完成。该建议与 MQ 方案一并审阅，不表示已获实施批准。

### 8.1 MySQL saver 适配边界

本机依赖读取结果为 LangGraph 1.2.10、langgraph-checkpoint 4.1.1、SQLite saver 3.1.1。第三方 `tjni/langgraph-checkpoint-mysql` 提供 AsyncMySaver；本轮读取的主分支中，写 checkpoint 和 pending writes 共用 `_cursor(pipeline=True)` 事务边界，未内置本项目的任务租约检查。依赖下限满足不等于运行时兼容性已验证，主分支也不能替代固定发布版本。

建议优先复用此库的 checkpoint 编解码与存储实现，增加项目拥有的租约事务适配层。实施时先锁定已发布版本并验证上述事务边界，禁止仅在调用 saver 前单独执行 assert_lease。若选定发布版本没有稳定可测的扩展点，须返回设计评审调整实现，不能静默退化为无写保护 saver。

每次执行使用不可变的执行上下文构造 saver：task_type、task_id、user_id、owner、claim_generation、允许的 checkpoint thread。不能把所有并发 Run 的执行权放进一个可变全局对象，也不能只从模型可影响的 state 读取执行权。

`aput` / `aput_writes` 写入事务遵循以下顺序：

1. 取得同一 MySQL 实例和业务 schema 的连接，开启短事务。
2. 锁定对应 Run/Job 行；锁到后读取数据库当前时间，复核运行状态、owner、代际和租约。
3. 验证 checkpoint thread 与任务绑定；写入 checkpoint、关联数据或 pending writes。
4. 在提交前复核当前租约，失败则整体回滚并抛出 LeaseLost。
5. 提交后释放行锁。新的 Claim 也更新同一任务行，因此新旧执行权切换与 checkpoint 写入具有明确先后关系。

SQL 事务不跨越 LLM、ES、文件解析或其他 Graph 节点执行。锁等待和 DB 操作时间均有限。超时/取消必须回滚，失败连接丢弃，不能复用含未完成事务的连接。

读取接口也限定可信 thread；管理侧的 delete/prune 与执行侧分开，禁止 Worker 调用无范围维护操作。所有状态保存使用同步持久化模式，使下一节点开始前已完成 checkpoint 提交。

验收覆盖 saver 读写、pending writes、并行节点部分成功、checkpoint 重开、父链、取消回滚、用户隔离以及旧 owner 的 aput/aput_writes 拒绝。若使用库的受保护事务接口，要固定依赖并用契约测试阻止升级时绕开校验。

### 8.2 Graph 恢复入口

Query 继续保留会话的服务端 checkpoint thread，但必须比较保存状态中的 run_id 和 runtime snapshot：

| Checkpoint 情况 | 行为 |
|---|---|
| 无记录 | 构造新 QueryState，启动新图 |
| 同一 Run、有下一节点 | 通过 `ainvoke(None, ...)` 恢复，保留研究次数和中间结果 |
| 同一 Run、图已结束 | 从已保存且审计完成的结果结束业务 Run，不重新生成 |
| 上一个已结束 Run | 在确认同会话没有其他活动 Run 后显式初始化当前新 Run |
| 其他活动 Run / 快照不匹配 | 拒绝恢复并记录不兼容原因，不能覆盖 |

Ingestion 以 Job 为 thread，恢复前绑定新执行上下文；状态中的旧 owner 字段只用于历史观察，新的写权限来自 saver/运行时持有的可信 Claim。已有完成 checkpoint 也应直接进行幂等业务收尾，不能因为 snapshot.next 为空就重新解析文档。

### 8.3 外部副作用的边界

MySQL fence 只能原子保护 MySQL 写入，不能原子保护一次独立的 Elasticsearch 或文件写入。现有 before_side_effect 检查与远程写入之间有时间窗口，多 Worker 设计必须承认并消除可见结果上的影响。

建议将非幂等中间产物按 Job/执行代际暂存，只有当前租约允许提交 accepted manifest；检索回源以 DB 已接受版本为准。旧执行可留下不可见暂存物，但不得覆盖已接受产物或改变其他版本可见性。清理与发布需要按已接受版本对账，不允许旧 Worker 直接删除新执行产物。全局索引 Alias 切换应与单个 Job 执行权分离，不能由任意过期入库 Worker 切换代际。

这部分需要在实施前逐项核对 IndexWriter、VersionPublisher、Artifact 路径和检索版本校验，写出各操作的幂等键及可见性规则，并纳入暂停旧 Worker 后恢复的真实测试；不能仅凭“有 assert_lease”宣布外部副作用安全。

本地多进程可共享已有受控 Artifact 目录；跨机器故障接管需要共享存储与可验证来源，不能用两个本地进程测试宣称跨机已经可用。进程级模型并发额度也不能冒充集群额度。

## 9. 迁移与回退

所有目标路径最终以 RabbitMQ 为唯一任务 Broker，替换 StreamBroker/reclaim 抽象、Redis 专用错误、启动接线、readiness、脚本、隔离测试夹具、评测入口及运维文档。不能只迁移 Query 而保留 Ingestion。

迁移采用受控停写切换：暂停任务入口和旧 Dispatcher；有界等待旧执行结束，记录剩余任务和 checkpoint；关闭旧消费者，确保旧执行失去写权限；以 DB 非终态任务生成唯一的新 Broker 调度记录，再启动新消费者和入口。不能把双发 Redis/RabbitMQ 作为去重机制。

保留旧队列、checkpoint 和备份直至核验完成，不自动删除用户现有数据。单机 SQLite 未完成 Run 的切换，必须选择并验证 checkpoint 导入或让旧 Run 安全完成；不能重置状态伪装迁移成功。

回退不是直接启动旧 Worker：必须停止新写入、清点新执行代际与 checkpoint 格式，判断旧版本能否读取。若不能，走已验证的恢复方案，不承诺无条件零成本回退。

## 10. 验收矩阵

| 故障注入或场景 | 必须证明 |
|---|---|
| API 提交成功，Broker 不可用 | 任务与 Outbox 保留，恢复后可执行 |
| 发布成功，确认丢失 | 同一事件允许重发，业务不重复提交 |
| 无法路由 / confirm NACK | Outbox 未被标记 dispatched，有可定位错误 |
| Dispatcher 租约过期后恢复 | 旧 owner 不能更新新发布状态 |
| 正常投递重复 / 旧代际消息 | 不增加执行次数、不重跑终态任务 |
| 执行中强杀 Worker | 其他进程接管当前 Run 的有效 checkpoint |
| 图完成后、DB 终态提交前崩溃 | 使用已完成 checkpoint 提交结果，不重新发起整图 |
| 终态提交后、ACK 前断网 | 再投递只确认终态 |
| 旧 Worker 暂停、租约被接管后恢复 | 旧结果和旧 checkpoint 写入均被拒绝 |
| SQL 重试事务提交后崩溃 | 下一轮通知按计划恢复，无内存定时器依赖 |
| DLQ 不可用 | 失败状态及死信意图保留，恢复后补投 |
| poison / unknown 消息 | 被可靠隔离，不无限循环 |
| busy 通知 ACK 后原执行者崩溃 | Recovery Scheduler 可重新通知并恢复 |
| 大量任务排队 | 内存和执行并发有界，Query/Ingestion 互不占满执行池 |
| 取消与完成并发 | 唯一业务终态，无取消后的越权发布 |
| 本地 Broker 重启 | 持久化队列恢复、连接恢复、未确认消息可再投递 |
| Redis -> RabbitMQ 切换 | 两类任务均迁移，旧通知不能造成重复执行或遗失任务 |

协议单测覆盖错误分类、token fence 和消息格式；真实 MySQL/RabbitMQ 集成测试覆盖事务、确认及恢复；真实多进程测试覆盖杀进程和暂停旧 owner。Graph 的恢复测试必须使用真实 LangGraph/checkpointer，即使节点用确定性桩，不能只用 FakeGraph 证明恢复。外部模型无需在每次故障测试重复计费，另安排小规模真实 RAG 冒烟验证。

关键观测：队列 ready/unacked、最老 Outbox 年龄、发布重试、任务等待与执行耗时、租约接管、过期写拒绝、checkpoint 恢复、业务重试和死信积压。日志保留 run/job/event ID，不记录凭证或原始上下文。

## 11. 当前待讨论决策

- 建议采用 RabbitMQ + Quorum Queue；用户已明确部署方式，尚未明确选型确认。
- 推荐本次包括多个 Worker 的竞争执行与崩溃接管，使用 MySQL 共享 checkpoint 和同事务租约校验；第 8 节提供可审阅的建议边界。
- 详细设计确认后才进入实现计划与代码迁移。本草案没有修改运行时、启动容器、连接用户数据服务或执行数据库迁移。

## 参考

- [RabbitMQ Publisher Confirms / Consumer ACK](https://www.rabbitmq.com/docs/confirms)
- [RabbitMQ Quorum Queues：复制和可靠死信策略](https://www.rabbitmq.com/docs/quorum-queues)
- [RabbitMQ Consumers：prefetch 与 ACK 超时](https://www.rabbitmq.com/docs/consumers)
- [RabbitMQ 本地 Docker 安装](https://www.rabbitmq.com/docs/download)
- [aio-pika](https://docs.aio-pika.com/)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [第三方 MySQL saver 项目](https://github.com/tjni/langgraph-checkpoint-mysql)
- [MySQL saver 事务边界源码](https://github.com/tjni/langgraph-checkpoint-mysql/blob/main/langgraph/checkpoint/mysql/aio_base.py)
