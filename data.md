# Cortex Dataset Description

## Overview

- **Source file**: `cortex.parquet`
- **Rows**: 3,924
- **Columns**: 12
- **Domain**: Real estate financial ledger data for a property company ("PropCo")
- **Time range**: 2024–2025 (15 months, 5 quarters)

For architecture, agents, and UI behavior, see `SPEC.md` in the repo root.

## Columns

| Column | Type | Nulls | Unique | Description |
|---|---|---|---|---|
| `entity_name` | str | 0 | 1 | Legal entity name. Single value: **PropCo** |
| `property_name` | str | 581 | 5 | Building identifier (e.g. Building 17, 120, 140, 160, 180). Null for entity-level expenses |
| `tenant_name` | str | 759 | 18 | Tenant identifier (e.g. Tenant 4, 6, 8, 10, 12, 15). Null for property/entity-level entries |
| `ledger_type` | str | 0 | 2 | **revenue** (3,135 rows) or **expenses** (789 rows) |
| `ledger_group` | str | 0 | 5 | High-level accounting group (rental_income, general_expenses, management_fees, sales_discounts, taxes_and_insurances) |
| `ledger_category` | str | 0 | 29 | Detailed accounting category (e.g. revenue_rent_taxed, bank_charges, directors_fee, financial_expenses) |
| `ledger_code` | int32 | 0 | — | Numeric ledger code. Range: 4370–8008, median: 8000 |
| `ledger_description` | str | 0 | 28 | Bilingual (Dutch/English) description (e.g. "Bankkosten \| Bank charges", "Opbrengst Huren belast Revenue Rent taxed") |
| `month` | str | 0 | 15 | Month in `YYYY-MMM` format (e.g. 2024-M05, 2025-M01) |
| `quarter` | str | 0 | 5 | Quarter in `YYYY-QN` format (e.g. 2024-Q1 through 2025-Q1) |
| `year` | str | 0 | 2 | Year: 2024 (3,181 rows) or 2025 (743 rows) |
| `profit` | float64 | 0 | — | Monetary value. Range: -154,415.07 to 154,415.07. Mean: 390.76. Median: ~0.0 |

## Key Observations

### Entity Structure

- Single entity **PropCo** with **5 properties** (Buildings 17, 120, 140, 160, 180)
- **18 tenants** across those properties
- Hierarchical: Entity → Property → Tenant (nulls indicate higher-level aggregation)

### Financial Structure

- **Ledger hierarchy**: `ledger_type` → `ledger_group` → `ledger_category` → `ledger_code`/`ledger_description`
- **Revenue dominates**: 80% of rows (3,135) are revenue entries
- **Top category**: `revenue_rent_taxed` accounts for 2,031 rows (52%)
- **Top group**: `rental_income` accounts for 2,932 rows (75%)
- Descriptions are bilingual (Dutch | English)

### Time Dimension

- **15 months** of data across **5 quarters** (2024-Q1 through 2025-Q1)
- 2024 has the bulk of data (81%)
- Most active month: 2024-M06 (804 rows)

### Profit Distribution

- Highly variable (std: 13,146 vs mean: 391)
- Median near zero, suggesting many small/zero entries
- Symmetric extremes: min -154,415 and max +154,415
- 25th percentile: -32.0, 75th percentile: 378.97

### Null Patterns

- `property_name`: 581 nulls (15%) — entity-level entries without a specific property
- `tenant_name`: 759 nulls (19%) — property-level or entity-level entries without a specific tenant
