# SR&ED Document Quick Reference Guide

## File Location
```
i3International_SRED_Cart_Counting_Project_Filled.docx
```

## Quick Edit Guide

### If You Need to Update Specific Sections:

**1. Project Dates or Team Info**
- Go to: "PROJECT INFORMATION" (first table)
- Edit the table rows directly
- Row 2: Start Date
- Row 3: End Date
- Row 8: Project Team

**2. Technological Advancements (TAs)**
- Section: "SECTION 246: TECHNOLOGICAL ADVANCEMENTS SOUGHT"
- Four headings: TA#1 through TA#4
- Each TA has subsections:
  - "Design and Development" — What was created
  - "New Capability vs. Previous Technology" — Advantages
  - "Technological Advancement for i3 International" — Business impact
  - "Knowledge Gained Through Systematic Investigation" — Key findings

**3. Technological Obstacles (TOs)**
- Section: "SECTION 242: TECHNOLOGICAL OBSTACLES"
- Four headings: TO#1 through TO#4
- Each TO has subsections:
  - "Key Shortcomings of Current Technology"
  - "Unknown Elements / Technical Roadblocks"

**4. Work Activities (WAs)**
- Section: "SECTION 244: ACTIVITIES PERFORMED"
- Four headings: Work Activity #1 through #4
- Each WA has 4-5 "Key Steps" with:
  - "Systematic Investigation"
  - "Key Finding"
  - "Implementation" (code location)
  - "Main Challenge"

**5. Mathematical Equations**
- Section: "KEY MATHEMATICAL FORMULATIONS"
- Seven subsections covering all major formulas
- Use "Courier New" font for equations
- Center-align multi-line equations

**6. Configuration & Implementation**
- Section: "IMPLEMENTATION DETAILS & CONFIGURATION"
- Contains 8 subsections:
  1. Module organization
  2. Configuration parameters (table)
  3. Horizon definitions (table)
  4. Model selection strategy
  5. Validation results (table)
  6. Consumption model validation (table)
  7. Test coverage
  8. Deployment architecture

---

## Key Statistics (for reference)

| Metric | Value |
|--------|-------|
| File Size | 48 KB |
| Paragraphs | 197 |
| Headings | 71 |
| Tables | 4 |
| Sections | 7 major |
| Technological Advancements | 4 |
| Technological Obstacles | 4 |
| Work Activities | 4 |
| Mathematical Topics | 7 |

---

## Key Numbers to Remember

### Configuration Parameters
- `CONFORMAL_ALPHA = 0.1` (90% coverage)
- `MIN_CALIBRATION_SAMPLES = 50`
- `RED_THRESHOLD = 15` carts
- `AMBER_THRESHOLD = 25` carts
- `TARGET_STOCK = 60` carts
- `LEAD_TIME_MIN = 15` minutes
- `ALERT_BAND_MIN_EVENTS = 15`

### Validation Results
- **20-minute forecast MAE:** 3.16 carts
- **ROC-AUC:** 0.937
- **PR-AUC:** 0.80
- **Brier Score:** 0.083
- **Improvement over baseline:** 17%
- **K-Means warning threshold:** 0.185
- **K-Means critical threshold:** 0.571

### Code Statistics
- **Features generated:** 127+
- **Lag values:** 1, 2, 3, 5, 10 records
- **Rolling windows:** 3, 6 records
- **Unit tests (cart_forecasting):** 49
- **Unit tests (cart_platform):** 193

### Consumption Scenarios
- **LOW:** 0.25 carts per entering customer
- **MEDIUM:** 0.50 carts per entering customer (chosen)
- **HIGH:** 0.80 carts per entering customer
- **Reserve availability:** 60 carts/day (LIMITED scenario)

---

## Main Python Files Referenced

| File | Purpose |
|------|---------|
| `cart_forecasting/src/alert_engine.py` | Alert state machine with cooldown & hysteresis |
| `cart_forecasting/src/replenishment_optimizer.py` | Quantity & urgency calculation |
| `cart_forecasting/src/audit/inventory_model.py` | Demand-driven inventory model |
| `cart_forecasting/src/audit/audited_simulator.py` | Scenario evaluation & backtesting |
| `cart_forecasting/src/features.py` | Feature engineering pipeline |
| `cart_forecasting/src/forecasting.py` | Conformal calibration |
| `cart_platform/training/calibration.py` | Per-entity calibrator selection |
| `cart_platform/training/alert_bands.py` | K-Means alert band discovery |
| `cart_platform/training/promotion.py` | Promotion gates & validation |
| `cart_platform/server/scoring_api.py` | Real-time scoring endpoint |
| `cart_platform/registry/database.py` | Entity registry & config |
| `cart_platform/registry/outbox.py` | Transactional outbox pattern |

---

## Making Updates

### To Update Content:
1. Open in Microsoft Word or Google Docs
2. Find the relevant section using Ctrl+F
3. Edit the content while preserving formatting
4. Save as .docx

### To Update Tables:
1. Click inside the table
2. Right-click to add/delete rows
3. Preserve the header row formatting
4. Use consistent font and alignment

### To Add New Equations:
1. Go to "KEY MATHEMATICAL FORMULATIONS" section
2. Add a new heading with "Heading 3" style
3. Use bullet points with formula notation
4. Use Courier New font for code/equations

### To Regenerate from Scratch:
The Python script is self-contained. Edit the values and re-run to generate an updated document.

---

## Common SR&ED Review Issues (to watch for)

✅ **Good:** Technical depth with specific numbers, equations, and measurable outcomes

❌ **Avoid:** Vague descriptions like "improved performance" without metrics

✅ **Good:** Clear mapping between TAs (what you built) and TOs (problems solved)

❌ **Avoid:** Missing the "systematic investigation" aspect

✅ **Good:** Specific code file references and test coverage numbers

❌ **Avoid:** Generic statements without implementation evidence

✅ **Good:** Validation results from real data (not simulated)

❌ **Avoid:** Unsubstantiated claims without backing data

---

## Version History

| Date | Status | Notes |
|------|--------|-------|
| 2026-11-11 | ✅ Complete | Initial document generation from codebase analysis |
| - | - | Ready for SR&ED submission |

---

## Contact & Support

For questions about the content, refer to:
- Project lead: ZacFawkes
- Repository: Cartcounting_model_forecast
- Related docs: README.md, MULTI_SITE_SYSTEM_DESIGN.md

