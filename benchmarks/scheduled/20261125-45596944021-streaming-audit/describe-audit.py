"""Render a reviewed streaming inventory from the verified full report and 10M shard."""
import csv,json,sys
from collections import Counter
from pathlib import Path
from unicodedata import east_asian_width

root=Path(sys.argv[1])
def readcsv(name):
    with (root/name).open(newline='') as s:return list(csv.DictReader(s))
allrows=readcsv('all-scheduled-cases.csv')
reg=readcsv('confirmed-regressions.csv')
coverage=readcsv('stream-operator-coverage.csv')
sources=json.loads((root/'sources.json').read_text()); run=sources['actions_run']
verdicts={'regression':'回退','improved':'提升','no-confirmed-regression':'未确认回退','inconclusive':'不确定','invalid-evidence':'无有效对照'}
def v(row):return verdicts.get(row['audited_classification'],row['audited_classification'])
def timing(row,key):return f"{float(row[key]):,.2f}" if row[key] else '无有效计时'
def shift(row):return f"{float(row['change_percent']):+.2f}%" if row['change_percent'] else ''
def table(head, rows):
    listings=[head,*rows]
    width=lambda s:sum(2 if east_asian_width(c) in 'WF' else 1 for c in s)
    widths=[max(map(width,col)) for col in zip(*listings)]
    line=lambda xs:'| '+' | '.join(cell+' '*(w-width(cell)) for cell,w in zip(xs,widths))+' |'
    return [line(head),'|'+'|'.join('-'*(n+2) for n in widths)+'|',*[line(row) for row in rows]]
def default(backend):
    return {r['scenario']:r for r in allrows if r['family']=='engines' and r['backend']==backend and r['primary_fixture_rows']=='10000000' and r['variant']=='default'}
four=default('calc-flow-stream');one=default('calc-flow-stream-1t')
assert four.keys()==one.keys() and four
classes=Counter(x['audited_classification'] for x in allrows)
text=['# 全量流执行基准复核','','本次运行候选为 `'+run['head_sha']+'`。公开安装包的发布基线为 `'+sources['public_release_baseline']+'`。计划中的原生 Rust 分片以候选的第一父提交 `'+sources['native_comparison_baseline']+'` 为基线。运行的最终状态为 `'+run['conclusion']+'`。[GitHub Actions 完整运行]('+run['html_url']+')','',
      f"完整报告有 {len(allrows):,} 条结果。对 {sources['summary']['independently_verified_paired_cases']:,} 个配对用例核对了逐次正确性与两轮判定。共有 {len(reg)} 项确认回退，{classes['improved']} 项确认提升，{classes['inconclusive']} 项结论不确定，{classes['no-confirmed-regression']} 项未确认回退。单次指标、外部引擎和信息性比较保留在全表。[全量 CSV](all-scheduled-cases.csv) · [全部回退](confirmed-regressions.csv)",'','## 每侧或主输入 1,000 万行的默认流式用例','','公开时钟从已就绪数据入队计至结果 Arrow 表。表中时长单位为毫秒。静态 Join 的维表最多 64 行，ASOF 两侧各 1,000 万行。每个线程配置独立与同入口的 11.14 对照。变化百分比来自合并样本 p50，两轮配对区间决定回退或改善判定。','']
rows=[]
for case in sorted(four):
    row=[case]
    for r in (four[case],one[case]):
        row+=[timing(r,'baseline_p50_ms')+' → '+timing(r,'candidate_p50_ms'),shift(r)+' '+v(r)]
    rows.append(row)
if sources["summary"].get("other_report_errors"):
    text+=["本轮有未能形成有效证据的用例或分片。所有报告错误与两轮确认回退分别保留。[报告错误](report-errors.json)",""]
text+=table(['算子','4T 发布 → 候选 p50','4T 配对判定','1T 发布 → 候选 p50','1T 配对判定'],rows)
text+=['','完整 CSV 另列两臂 p95、两轮区间、输出基数、批大小与 checkpoint 变体。SMA20 和双均线采用行数窗口。window_sum 是 10 秒事件时间分桶窗口（tumbling window），group_by 使用跨越整个输入的分桶求每键总和。逐行时间戳滑窗另有原生与安装包的千万行 20 秒窗口对照。[时间戳滑窗实验](https://github.com/wegamekinglc/calc-flow/tree/ba0a913e9c751598acf094586723168587f59be3/benchmarks/evidence/20261125-rolling-window-followup/timestamp-20s)','']
for r in coverage:
    if r['scenario']=='interval_join' and r['backend']=='calc-flow-stream':
        text+=['区间 Join 的目录上限为 '+f"{int(r['catalog_cap_rows']):,}"+' 行，应用其最大实际规模的结果。该四线程用例的判定为 `'+r['largest_default_verdict']+'`。[算子覆盖清单](stream-operator-coverage.csv)','']
top=sorted([r for r in reg if r['surface']=='python-public-managed-stream' and r['primary_fixture_rows']=='10000000'],key=lambda r:float(r['candidate_p50_ms'])-float(r['baseline_p50_ms']),reverse=True)[:8]
if top:
    text+=['## 千万行回退的绝对耗时差','','对所有配置按 p50 绝对差降序排列，列出增加最多的用例。','']
    text+=table(['用例','增加毫秒','相对变化'],[[r['case_id'],f"{float(r['candidate_p50_ms'])-float(r['baseline_p50_ms']):,.2f}",shift(r)] for r in top])
    text+=['']
text+=['## Rust 核心与小批延迟','','原生 Rust 分片中的父提交比较用于查找近期变化。直接对照 11.14 的原生 Rust 单水位与千万行整作业另有固定适配器和逐行校验。[ASOF 单水位与输出分段](https://github.com/wegamekinglc/calc-flow/tree/d26860e433df4d33bbcb92ba23cfc5849bc671a7/benchmarks/evidence/20261125-asof-watermark-phases/output-charge-followup) · [原生四线程千万行对照](https://github.com/wegamekinglc/calc-flow/tree/f1728a86cf24c83b25c1d6ff4188a4b58b3fef0a/benchmarks/evidence/20261125-asof-watermark-phases/dense-projection-native-4t)','']
warm=[r for r in allrows if r['family']=='warm' and r['history_rows']=='10000000']
if warm:
    text+=table(['热态用例','追加行数','发布 p50','候选 p50','配对判定'],[[r['scenario'],r['timed_append_rows'],timing(r,'baseline_p50_ms'),timing(r,'candidate_p50_ms'),v(r)] for r in warm])
    text+=['']
text+=['原始完整报告、千万行分片、绑定的提交和构建摘要保存在同一目录。复核脚本以原始 Actions 附件的 SHA-256 校验来源。[运行与附件标识](sources.json) · [发布版源码与 Polars 对照](https://github.com/wegamekinglc/calc-flow/tree/8b4fb0b5b809b04a5b3241f8b0e48fe57597b3ae/benchmarks/evidence/20261125-stream-engine-release-review)\n']
(root/'README.md').write_text('\n'.join(text))
