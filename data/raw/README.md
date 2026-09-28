# data/raw — 原始数据与来源说明

## xlam_sample.jsonl（100 条）

**这是按 xLAM 记录格式手工合成的样例数据（schema-faithful synthetic sample），
不是 Salesforce/xLAM 数据集本身的样本。** 用途：在离线环境端到端验证
lifter 管线（xLAM 格式 → TaskIR）。

记录格式（xLAM function-calling 约定）：

```json
{"task_id": "xlam-sample-0001",
 "question": "Find flights from SFO to Tokyo on ... and tell me the price in JPY.",
 "steps": ["Let me start. ```json\n[{\"name\": \"FlightSearch.search\", \"arguments\": {...}}]\n```", ...],
 "answer": "Here you go. ..."}
```

## 换用真实 xLAM 数据

```python
from datasets import load_dataset
ds = load_dataset("Salesforce/xLAM", "xlam_func_language_tools", split="train")
# 写出为 jsonl 后：
#   python scripts/run_xlam_pipeline.py --input <你的文件>.jsonl
```

lifter（`src/lifter/xlam.py`）对 `steps` 做容错解析：接受
```json 代码块、纯 JSON list、或已结构化的 dict 列表。

## 生成方式

```bash
python scripts/gen_xlam_sample.py -n 100 --seed 1234
```
