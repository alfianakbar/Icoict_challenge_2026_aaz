# Explainable Machine Learning for Travel Demand and Context-Aware Activity Recommendation

This repository contains the source code, experimental notebooks, figures, and supporting materials for the ICOICT 2026 TRAVLR Challenge paper:

**Explainable Machine Learning for Travel Demand and Context-Aware Activity Recommendation Using Integrated Accommodation and Activity Data**
**Alfian Akbar Gozali, Telkom University**

## Overview

Travel platforms rarely operate on a single clean dataset. Demand signals, accommodation metadata, and destination activity information are often stored in separate systems with inconsistent identifiers, missing values, and uneven coverage.

This project investigates whether heterogeneous travel-platform data can be integrated into an explainable decision-support pipeline for:

1. accommodation demand analysis,
2. high-demand classification,
3. transaction-volume regression,
4. feature attribution analysis, and
5. context-aware activity shortlisting.

The study uses the TRAVLR challenge dataset from ICOICT 2026 and evaluates the practical value and limitations of multi-source integration under realistic data conditions.

## Research Questions

This project addresses five research questions:

- **RQ1:** Can accommodation transactions, accommodation metadata, and activity data be integrated with sufficient coverage?
- **RQ2:** How does the integrated feature space perform for high-demand accommodation classification?
- **RQ3:** How does the integrated feature space perform for transaction-volume regression?
- **RQ4:** Which features contribute most to the demand prediction model?
- **RQ5:** Can integrated data support context-aware activity shortlisting under offline proxy evaluation?

## Dataset

The dataset is provided through the TRAVLR Challenge Kaggle repository.

Dataset source:

```python
import kagglehub

path = kagglehub.dataset_download("noviananggis/icoict-challenge-2026")
print("Path to dataset files:", path)
