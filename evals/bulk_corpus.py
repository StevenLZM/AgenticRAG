"""Parameterised fictional business corpus, separate from frozen evaluation gold."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals.report import atomic_write_text

BATCH = "synthetic-business-1000-20260927-v1"
ROOT = Path("var/artifacts/corpora") / BATCH
KINDS = {
    "policy": ("差旅与远程办公制度", "pdf"),
    "product": ("知识服务产品说明", "pdf"),
    "operations": ("季度服务运营台账", "xlsx"),
    "inventory": ("设备库存与采购台账", "xlsx"),
    "project": ("知识库迁移项目计划", "txt"),
    "meeting": ("项目验收会议纪要", "txt"),
    "incident": ("检索服务故障复盘", "txt"),
    "faq": ("用户常见问题与解答", "txt"),
    "sla": ("服务等级与升级流程", "txt"),
    "training": ("新员工知识库培训手册", "txt"),
}


def build_source():
    documents = []
    cities = ["上海", "北京", "杭州", "深圳", "成都", "南京", "武汉", "苏州", "西安", "广州"]
    industries = ["工业设备", "物流仓储", "零售运营", "企业软件", "能源管理"]
    for company in range(1, 51):
        org = f"虚构星河{company:02d}公司"
        city = cities[(company - 1) % len(cities)]
        industry = industries[(company - 1) % len(industries)]
        for year in (2025, 2026):
            delta = year - 2025
            key = f"S{company:03d}-{year}"
            project = f"舟桥-{company:03d}"
            product = f"云册-{company:03d}"
            limit = 400 + company * 10 + delta * 60
            remote = 1 + company % 3
            retention = 30 + (company % 4) * 30 + delta * 30
            response = 15 + (company % 3) * 5
            capacity = 5000 + company * 200 + delta * 1000
            owners = ["平台组", "实施组", "业务运营组", "信息安全组"]
            owner = owners[company % 4]
            facts = {"company": org, "city": city, "year": year,
                     "lodging_limit_cny_per_night": limit, "remote_days_per_week": remote,
                     "log_retention_days": retention, "p1_response_minutes": response,
                     "document_capacity": capacity, "project_id": project, "product_id": product}
            sections = {
                "policy": [
                    ("生效范围", f"本制度适用于{org}{city}办公点的正式员工，自{year}年1月1日起执行，覆盖{industry}项目出差及远程协作。其他企业、其他城市的制度不得直接套用。本年度出差日期决定适用版本，报销提交日期不改变标准。"),
                    ("住宿与交通", f"{year}年{city}住宿报销上限为每人每晚{limit}元人民币，含税，以实际合规票据金额与上限两者的较低值报销。餐费不计入住宿额度。特殊展会期间确需超标的，须在预订前取得财务主管书面批准；未批准部分由个人承担。"),
                    ("远程办公", f"每周最多远程办公{remote}天，需提前一个工作日在排班系统登记。试用期员工前两周需要现场培训。远程办公不等同于休假，核心协作时段为10:00至16:00。设备借用记录以本年度设备库存与采购台账为准。"),
                    ("申请及例外", "出差申请由直属主管审核，财务负责票据复核，跨部门项目由项目经理确认费用归属。紧急出差可先电话报备，两个工作日内补齐审批。境外补贴和家属随行费用不在本文规定范围，不能从住宿额度推导。"),
                ],
                "product": [
                    ("产品定位", f"{product}是{org}用于{industry}业务的内部知识服务，{year}版由{owner}维护。系统以文档检索和可追溯回答为主要功能，不作为合同审批或费用支付系统。产品部署于{city}业务节点。"),
                    ("容量和文件", f"本年度单空间文档容量为{capacity}份。支持文本型PDF、UTF-8文本和Excel工作表。扫描件需先通过OCR验收；受密码保护的文件必须解密后上传。上传受理不表示可检索，只有处理完成并发布的版本才能进入查询。"),
                    ("版本与权限", f"审计日志保存{retention}天。文档访问按空间成员角色控制，历史版本可供管理员审阅，默认检索应使用已发布版本。用户不得通过共享链接获取其他空间证据。删除后的可检索性需要通过实际查询复核。"),
                    ("交付边界", f"{project}迁移项目负责初始化资料，验收记录见同年度会议纪要。故障响应以服务等级与升级流程为准。产品介绍不承诺未列明的GPU型号、供应商报价或外部客户上线日期。跨版本比较必须同时查阅两年的产品说明。"),
                ],
                "project": [
                    ("目标与范围", f"{org}在{year}年实施{project}项目，将{industry}相关操作规程迁移至{product}。项目由{owner}牵头，交付对象是{city}业务团队。历史资料保留原生效年份，迁移不自动更新制度内容。"),
                    ("里程碑", f"2月完成资料盘点，3月完成上传与质量抽检，4月完成用户试运行。计划抽检{20 + company}份资料，至少覆盖PDF、文本、表格三类。实际验收结果以会议纪要为准，计划中的日期不等于真实完成日期。"),
                    ("依赖与验收", "资料盘点完成后才能冻结迁移清单；权限核对完成后才能开放试运行。文件处理成功率、标题层级、表格单位和引用定位分别验收。发现缺失页或字段错位时退回资料责任组，不能仅凭上传成功标记通过。"),
                    ("风险与回退", f"主要风险为旧版制度混入、同名文件重复、扫描文字识别偏差。出现检索空结果时先查文档状态，再查访问范围。回退负责人为{owner}，回退需要保留原始文件和任务编号；禁止删除审计记录来掩盖失败。"),
                ],
                "meeting": [
                    ("会议信息", f"{org}{project}项目验收会于{year}年4月18日在{city}召开。参与角色为项目经理、{owner}代表、业务验收代表和安全审核代表。会议仅评价本企业的迁移批次，不代表产品全部场景通过。"),
                    ("验收结论", f"本次抽检{20 + company}份文档，发现{company % 4}项非阻断问题，均为标题层级或页码标注问题。核心制度和运营台账可检索，准予受控试运行；未覆盖扫描手写材料，不应对其识别效果作出结论。"),
                    ("行动项", f"{owner}在4月22日前补齐引用定位，业务运营组在4月25日前确认FAQ。信息安全组复核跨空间访问样本。资料责任组负责旧版文档标识，修订必须关联原始文档编号，不能只上传新的同名文件。"),
                    ("争议处理", "会议批准的是受控试运行，并非正式对外商用。若项目计划与本纪要日期不同，以本纪要记录的实际事项为准。尚未决议的采购品牌、法定代表人和外部合同金额保持未披露状态。"),
                ],
                "incident": [
                    ("故障概况", f"{org}的{product}在{year}年5月12日发生检索延迟升高，影响{city}节点。故障持续{18 + company % 17}分钟，未发现已确认的数据越权。用户表现为部分查询超时，不代表所有文档无法访问。"),
                    ("原因分析", f"原因是批量迁移与交互查询共享资源，后台并发超过当时的安全阈值。{owner}通过任务队列和检索阶段耗时确认瓶颈。文档总量上限{capacity}份是容量限制，不是允许同时处理的任务数量。"),
                    ("处理过程", f"先暂停新的批量提交，再降低任务并发，最后复核查询延迟。首次响应目标为{response}分钟，但目标不能替代实际事件时间。本事件未公布精确首响时间，因此不能判断首响是否违约。"),
                    ("改进与限制", "为上传设置背压，区分处理失败与排队，保存重试任务标识。新增重复提交检查和不可确定状态人工核对。故障恢复后抽样检查原有资料，没有以清空索引作为修复手段。跨月可用率需查运营台账，不能由一次故障推算。"),
                ],
                "faq": [
                    ("上传后搜不到怎么办", f"{org}{year}年FAQ：先确认文件处理完成，再确认当前空间和有效版本。{product}上传返回任务号只说明受理。若仍无法检索，向{owner}提供文档编号、任务号及问题描述，不要连续上传相同文件。"),
                    ("旧制度能否用于新差旅", f"不能仅依据上传时间判断制度新旧。{city}{year}年的住宿额度应查本年度差旅制度。跨年出差按实际住宿日期分别适用对应年度标准，本文不重复金额以避免FAQ与正式制度发生不一致。"),
                    ("权限与无答案", "有权限的文档仍可能没有用户所问的事实。没有检索结果不能证明事实不存在；应区分权限范围、资料缺失和服务故障。系统明确无法确定时，可补充实体名称、年份或上传相应原始材料。"),
                    ("支持渠道", f"普通问题提交服务台，紧急故障按同年度SLA升级。日志留存时长见产品说明。{project}项目进度以会议纪要核实，产品容量以产品说明核实，台账中的数量不等于承诺的服务能力。"),
                ],
                "sla": [
                    ("适用服务", f"本约定适用于{org}{year}年{product}内部服务，由{owner}值守，覆盖{city}节点。工作日普通请求在9:00至18:00受理，P1严重故障采用全天候值班。"),
                    ("分级目标", f"P1定义为主要业务流程整体不可用，首次响应目标{response}分钟；P2为部分功能受影响，首次响应目标60分钟；P3为咨询或低影响缺陷，两个工作日内响应。响应不等于解决，禁止将首响目标解释为修复承诺。"),
                    ("升级流程", "值班工程师先确认影响范围，无法恢复时升级平台负责人；疑似权限泄漏同步通知安全审核。所有升级记录包含任务号、发生时间、受影响空间和处理动作，不在工单中粘贴密钥或个人敏感数据。"),
                    ("统计口径", "可用率按约定服务窗口内的有效监测分钟计算，经批准的维护窗口单独披露。季度运营台账列出监测与中断分钟，可据此计算对应季度比例。本约定未给出合同赔偿金额，也未授权自动付款。"),
                ],
                "training": [
                    ("培训目标", f"{org}{year}年新员工培训面向{city}的{industry}团队，讲解{product}检索、文档上传和证据阅读。由{owner}组织，每位学员完成三个练习后方可独立维护资料。"),
                    ("练习一：版本查找", "分别找到2025和2026年的差旅制度，记录适用城市、生效日期及额度。不要只复制第一个检索结果。跨年比较需要两份对应资料，缺少任何一份时明确说明比较依据不足。"),
                    ("练习二：表格计算", "从运营台账读取季度监测分钟和中断分钟，再计算可用率。核对单位和列标题，不能用工单数量作为中断分钟。库存练习按期初加采购减领用计算期末，数量不足时应报告缺口而非编造库存。"),
                    ("练习三：上传排障", f"上传一份无敏感信息的练习文件，保存任务编号并等待完成。检查引用是否指向原文，不把任务排队视为成功。发生问题查阅故障复盘和SLA，{project}相关问题另附迁移批次编号。"),
                ],
                "operations": [
                    ("统计范围", f"{org}{year}年{product}季度服务运营记录，适用{city}节点。所有数据为合成测试观测，不代表真实企业经营或本项目运行情况。"),
                    ("字段口径", "监测分钟为剔除批准维护窗口后的服务窗口；中断分钟为其中不可用时长；工单数是服务请求数量，与分钟数不可互换。"),
                    ("计算规则", "季度可用率等于一减中断分钟除以监测分钟。年度汇总需要先加总分钟再计算，不能对季度百分比直接求和。"),
                    ("关联文件", f"分级响应承诺见本年度SLA，事件原因见故障复盘。台账只记录合成观测，不推定每张工单均已关闭。责任组为{owner}。"),
                ],
                "inventory": [
                    ("台账范围", f"{org}{year}年{city}设备台账，服务{project}项目与远程办公。只统计已验收入库设备，不包含采购申请中尚未交付的数量。"),
                    ("数量口径", "期初、采购、领用、期末单位均为台。领用是从可用库存中扣减，不等于资产报废。设备型号为测试代号，不对应真实品牌或市场报价。"),
                    ("计算规则", "期末库存等于期初库存加本期采购减本期领用。盘点差异由资产管理员另行记录，本表不包含未经审批的调整。"),
                    ("借用管理", f"远程办公资格按同年度差旅与远程办公制度审批，获批不代表设备自动发放。{owner}按库存安排借用，项目试运行状态见会议纪要。"),
                ],
            }
            for kind, (label, fmt) in KINDS.items():
                refs = [f"{key}-{other}" for other in {
                    "policy": ["inventory"], "product": ["project", "sla"],
                    "operations": ["sla", "incident"], "inventory": ["policy", "meeting"],
                    "project": ["product", "meeting"], "meeting": ["project", "faq"],
                    "incident": ["sla", "operations"], "faq": ["policy", "product"],
                    "sla": ["operations"], "training": ["policy", "operations", "inventory"],
                }[kind]]
                if delta:
                    refs.append(f"S{company:03d}-2025-{kind}")
                doc = {"document_key": f"{key}-{kind}", "filename": f"{key}_{label}.{fmt}",
                       "format": fmt, "category": kind, "year": year,
                       "title": f"{org} {year}年{label}",
                       "notice": "合成测试文档：企业、项目、制度和数值均为虚构，不作为真实业务依据。",
                       "sections": sections[kind], "references": refs, "facts": facts}
                if kind == "operations":
                    doc["headers"] = ["季度", "监测分钟", "中断分钟", "工单数", "可用率"]
                    doc["rows"] = [[f"{year}Q{q}", 120000 + q * 600, 20 + company * q + delta * 3,
                                    80 + company * 2 + q * 7, None] for q in range(1, 5)]
                elif kind == "inventory":
                    doc["headers"] = ["设备代号", "期初(台)", "采购(台)", "领用(台)", "期末(台)"]
                    doc["rows"] = [[f"TEST-{company:03d}-{n}", 10 + company + n,
                                    5 + delta + n, 3 + n, None] for n in range(1, 7)]
                documents.append(doc)
    return {"batch_id": BATCH, "schema_version": 1,
            "provenance": "deterministic-fictional-scenarios-not-real-company-data",
            "documents": documents}


def upload_order(documents):
    first = [next(d for d in documents if d["format"] == fmt) for fmt in ("txt", "pdf", "xlsx")]
    keys = {d["document_key"] for d in first}
    return first + [d for d in documents if d["document_key"] not in keys]


def save_source(root):
    content = json.dumps(build_source(), ensure_ascii=False, indent=2) + "\n"
    path = root / "source.json"
    if path.exists() and path.read_text() != content:
        raise ValueError("source changed; do not overwrite this batch")
    if not path.exists():
        atomic_write_text(path, content)


def freeze_manifest(root, source):
    documents = []
    for doc in source["documents"]:
        data = (root / "documents" / doc["filename"]).read_bytes()
        if not data:
            raise ValueError("empty document")
        documents.append({"document_key": doc["document_key"], "filename": doc["filename"],
                          "format": doc["format"], "sha256": hashlib.sha256(data).hexdigest(),
                          "size_bytes": len(data)})
    manifest = {"batch_id": source.get("batch_id", BATCH),
                "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "documents": documents}
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("manifest/files changed; refusing replacement")
    if not path.exists():
        atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
