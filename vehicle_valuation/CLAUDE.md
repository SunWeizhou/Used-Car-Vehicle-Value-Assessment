# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **used vehicle valuation system** (二手车残值评估系统) that assesses vehicle condition and value using multi-dimensional health profiling based on maintenance records. The system uses statistical modeling, LLM-based repair classification, and reliability engineering principles.

**Core Approach**: Micro-labeling (individual repair records) → Macro-aggregation (vehicle-level profiles) → Multi-dimensional scoring → Weighted final valuation

## Common Development Commands

### Running the Main Pipeline
```bash
# Run complete vehicle valuation pipeline
python main.py
```

### LLM-Based Repair Classification
```bash
# Process all maintenance records with LLM (concurrent processing)
python test_llm_hours.py

# Retry failed LLM classifications
python retry_failed_records.py      # Multi-threaded retry
python retry_slow_steady.py         # Conservative single-threaded retry
python retry_ultimate.py            # Batch processing with long delays
```

### Data Analysis
```bash
# Analyze repair hours distribution
python inspect_repair_hours.py
```

## High-Level Architecture

### Data Flow

```
Raw CSV Files (data/)
    ↓
[utils/preprocessing.py] → Cleaned DataFrames (df_base, df_parts, df_time)
    ↓
[utils/llm_structuring.py] → LLM Classification (Event_Type, System, Severity, Reasoning)
    ↓
[models/] → Vehicle Profiling
    ├── [lifecycle.py] → Weibull survival model (life expectancy score)
    ├── [behavior.py] → ECDF-based usage intensity & maintenance regularity scores
    └── [reliability.py] → Failure rate intensity (λ) from LLM severity labels
    ↓
[models/weighting.py] → PCA-based multi-dimensional weighting → Final vehicle score
```

### Key Data Structures

**Input Data** (3 CSV files in `data/`):
- `上汽跃进_燃油_baseinfo.csv`: Base repair records (VIN, ID, mileage, date, fault description)
- `上汽跃进_燃油_parts_info.csv`: Parts replacement records (linked by RECORD_ID)
- `上汽跃进_燃油_time_info.csv`: Labor hours records (linked by RECORD_ID)

**Intermediate Data**:
- `data/llm_parsed_results.csv`: LLM-classified repair records with structured labels

**Core DataFrame Columns**:
- `df_base`: ID, VIN, REPAIR_MILEAGE, SETTLE_DATE, FAULT_DESC
- `df_parts`: RECORD_ID, PARTS_NAME
- `df_time`: RECORD_ID, REPAIR_NAME, REPAIR_HOURS

### Model Architecture

**1. Lifecycle Model** ([models/lifecycle.py](models/lifecycle.py))
- **Theory**: Weibull distribution for survival analysis
- **Key Function**: `prepare_weibull_data()` - Prepares censored survival data (event=1 if vehicle not seen for >730 days)
- **Method**: Maximum Likelihood Estimation (MLE) to fit shape parameter (k) and scale parameter (λ)
- **Output**: Lifecycle score (0-100) based on vehicle position in survival curve

**2. Behavior Model** ([models/behavior.py](models/behavior.py))
- **Theory**: Empirical Cumulative Distribution Function (ECDF)
- **Metrics**:
  - Usage intensity: Daily mileage = (max_mileage - min_mileage) / span_days
  - Maintenance regularity: Maintenance density = maint_count / max_mileage × 10000
- **Scoring**: 100 × (1 - ECDF(daily_mileage)) for usage, 100 × ECDF(maint_density) for maintenance
- **Key Fix**: Uses mileage increment (not cumulative) to avoid bias from late-stage vehicles

**3. Reliability Model** ([models/reliability.py](models/reliability.py))
- **Theory**: "Micro-labeling then macro-aggregation" approach
- **Process**:
  1. Map LLM severity labels to weights: L0=0, L1=1, L2=5, L3=20
  2. Aggregate by VIN: total_fault_score = Σ(weights)
  3. Calculate failure rate intensity: Λ = total_fault_score / max_mileage
  4. Compute population baseline: Λ_pop = mean(Λ)
- **Scoring**: Score = 100 × exp(-0.693 × Λ_i / Λ_pop)
  - Score = 50 when Λ_i = Λ_pop
  - Score = 100 when Λ_i = 0
  - Score = 25 when Λ_i = 2×Λ_pop

**4. Weighting Model** ([models/weighting.py](models/weighting.py))
- **Theory**: Principal Component Analysis (PCA) for objective weighting
- **Method**: Computes eigenvalues from correlation matrix, derives weights from first principal component
- **Final Score**: Weighted sum of 4 dimensions (Lifecycle, Usage [inverted], Maintenance, Reliability)

### LLM Integration

**Dual-Verification Classification System** ([utils/llm_structuring.py](utils/llm_structuring.py)):

The LLM uses Chinese national standards (GB/T) for qualitative assessment combined with labor hours data for quantitative validation:

- **GB/T Standards** (Qualitative):
  - L3 (Major): Accident vehicles/overhaul (GB/T 30323/5624)
  - L2 (General): Component repair/part replacement
  - L1 (Minor): Consumable appearance/minor repair
  - L0 (Maintenance): Level 1/2 maintenance (GB/T 18344)

- **Labor Hour Correction** (Quantitative):
  - < P50 hours: Downgrade to L1 even if core components mentioned
  - P50-P85 hours: Typical L2 repair
  - > P85 hours: Strong L3 support
  - > 100 hours: Flag as potential data error unless content confirms major work

**Concurrent Processing Configuration**:
- **Workers**: 5 threads (reduced from 20 to avoid API rate limits)
- **Retry**: Exponential backoff (1s, 2s, 4s delays)
- **Timeout**: 30 seconds per API call
- **Delay**: 0.2s between requests
- **Checkpoint**: Saves every 50 records to `data/llm_parsed_results.csv`
- **Resume**: Automatically skips processed IDs (detected via existing results file)

**API Setup**:
```bash
# Set DeepSeek API key in environment or .env file
export DEEPSEEK_API_KEY="sk-your-key-here"
```

### Severity Level Mapping

| Level | Type | Weight | Examples |
|-------|------|--------|----------|
| L0 | Maintenance | 0 | Oil change, filter replacement |
| L1 | Minor | 1 | Light bulbs, wipers, fuses |
| L2 | General | 5 | Brake pads, battery, tires |
| L3 | Major | 20 | Engine overhaul, transmission, accident repair |

## Important Implementation Details

### Data Cleaning Pipeline
- Removes duplicates (~3,949 records from baseinfo)
- Filters mileage anomalies: 10 km - 2,000,000 km range
- Converts ID columns to string format for consistent joining
- Handles date parsing and validates time ranges

### Right-Censoring in Survival Analysis
Vehicles not seen for >730 days are marked as "failed" (event=1), others are "censored" (event=0). Censored samples still contribute to likelihood function via survival function S(t).

### Mileage Calculation Fix
The behavior model was fixed to use **mileage increment** (max_mileage - min_mileage) instead of cumulative mileage. This prevents bias where late-stage vehicles (entering with high mileage) appear to have artificially low daily mileage.

### Thread Safety
The concurrent LLM processing uses `Lock()` for shared counter updates and implements signal handlers (SIGINT, SIGTERM) for graceful interruption with data preservation.

### Error Handling
- API failures: Marked as `Severity='ERROR'` with error message in `Reasoning`
- Timeout: 30 seconds with automatic retry
- Connection errors: Retried up to 3 times with exponential backoff
- Data loss prevention: Incremental saves every 50 records

## File Organization

```
vehicle_valuation/
├── main.py                      # Main entry point
├── data/                        # Input CSV files and LLM results
│   ├── 上汽跃进_燃油_baseinfo.csv
│   ├── 上汽跃进_燃油_parts_info.csv
│   ├── 上汽跃进_燃油_time_info.csv
│   └── llm_parsed_results.csv   # Generated by LLM processing
├── models/
│   ├── lifecycle.py             # Weibull survival model
│   ├── behavior.py              # ECDF usage/maintenance scoring
│   ├── reliability.py           # Failure rate intensity model
│   └── weighting.py             # PCA-based multi-dimensional weighting
├── utils/
│   ├── preprocessing.py         # Data loading and cleaning
│   ├── llm_structuring.py       # LLM-based repair classification
│   └── math_tools.py            # Mathematical utilities
└── test_llm_hours.py            # LLM processing entry point
```

## Environment Setup

### Required Dependencies
```bash
pip install pandas numpy scipy statsmodels openai
```

### Configuration Files
- `.env`: Contains `DEEPSEEK_API_KEY` (do not commit)
- All data files use mixed encoding: baseinfo.csv uses GBK, others use UTF-8

## Performance Considerations

### LLM Processing Performance
- **Single-threaded**: ~80 records/hour (~19 days for 36,150 records)
- **Concurrent (5 workers)**: ~40 records/second (~15 minutes for 36,150 records)
- **Speedup**: ~1,800x improvement

### Memory Usage
Typical memory footprint:
- baseinfo: ~8 MB
- parts_info: ~15 MB
- time_info: ~3 MB
- Total: ~26 MB

### API Rate Limiting
When experiencing high failure rates (>30%), reduce `max_workers` parameter in `process_sample_batch_concurrent()` or use the conservative retry scripts (`retry_slow_steady.py`).


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Jan 15, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #987 | 9:58 PM | ✅ | Comprehensive project documentation created in CLAUDE.md | ~378 |
| #892 | 11:59 AM | 🟣 | DeepSeek API key configuration added to environment | ~204 |
| #889 | 11:56 AM | 🟣 | LLM工时辅助判定演示工具创建 | ~140 |
| #848 | 11:02 AM | 🟣 | 车辆档案整合功能实现 | ~174 |
| #842 | 11:00 AM | 🔄 | 可靠性模型集成LLM标注数据并改进错误处理 | ~136 |
| #833 | 10:52 AM | ✅ | Project changes staged with 12 unpushed commits | ~205 |
| #823 | 10:49 AM | 🟣 | 故障率强度模型实现 | ~206 |
| #807 | 1:11 AM | 🟣 | Weibull生命周期建模集成到主程序 | ~120 |
| #757 | 12:20 AM | 🟣 | Main entry point implemented for vehicle valuation system | ~251 |
| #747 | 12:16 AM | 🟣 | Created main application entry point file | ~178 |
</claude-mem-context>