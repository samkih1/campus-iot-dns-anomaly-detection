# Lightweight Hybrid Machine Learning Framework for DNS Anomaly Detection in Campus IoT Environments

## Overview

This repository provides the Python implementation, sanitised feature-level datasets, and experimental results supporting the research article:

**Design and Implementation of a Lightweight Hybrid Machine Learning Framework for DNS Anomaly Detection in Campus IoT Environments**

The study investigates a lightweight, sequential hybrid detection framework combining Random Forest (RF) and Isolation Forest (IF) for identifying anomalous DNS behaviour in simulated campus Internet of Things (IoT) environments.

## Research Objectives

The implementation focuses on:

1. Extracting behavioural features from DNS traffic.
2. Training a supervised Random Forest classifier for known DNS attack categories.
3. Training an Isolation Forest model on benign DNS behaviour.
4. Integrating the models through sequential hybrid decision logic.
5. Evaluating detection performance on an independently generated simulated test dataset.

## Dataset

The repository contains two sanitised feature-level datasets:

| Dataset | Records | Purpose |
|---|---:|---|
| campus_dns_features.csv | 4,400 | Model development |
| campus_dns_test_features.csv | 1,400 | Independent evaluation |

The development dataset contains:

- 3,400 benign observations
- 500 DNS tunnelling observations
- 500 NXDOMAIN flooding observations

The independent test dataset contains:

- 1,000 benign observations
- 200 DNS tunnelling observations
- 200 NXDOMAIN flooding observations

The datasets contain seven numerical DNS behavioural features, a traffic scenario field, and a class label.

Raw packet captures, original DNS queries, and identifying network addresses are not included.

## DNS Behavioural Features

The framework uses seven features:

1. subdomain_length
2. query_entropy
3. inter_query_interval
4. interval_std_10q
5. queries_last_5s
6. nxdomain_ratio_1s
7. repeated_query_ratio_1s

## Hybrid Detection Architecture

The framework uses sequential Random Forest and Isolation Forest decision logic.

1. Random Forest classifies each observation as benign, DNS tunnelling, or NXDOMAIN flooding.
2. Observations classified as known attacks retain their Random Forest classifications.
3. Observations classified as benign undergo Isolation Forest anomaly screening.
4. Isolation Forest scores below the calibrated threshold are flagged as Anomalous_Unknown.
5. Remaining observations retain the benign classification.

The calibrated Isolation Forest threshold used in the reported evaluation is:

**-0.083149**


## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Experiments

Train the Random Forest:

```bash
python3 src/train_random_forest.py
```

Train the Isolation Forest:

```bash
python3 src/train_isolation_forest.py
```

Evaluate the Random Forest:

```bash
python3 src/evaluate_random_forest.py
```

Evaluate the Isolation Forest:

```bash
python3 src/evaluate_isolation_forest_calibrated.py
```

Evaluate the hybrid framework:

```bash
python3 src/evaluate_hybrid_model.py
```

## Experimental Results

| Metric | Random Forest | Isolation Forest | Hybrid RF-IF |
|---|---:|---:|---:|
| Accuracy | 100.00% | 91.57% | 92.14% |
| Precision | 100.00% | 78.09% | 78.43% |
| Recall | 100.00% | 98.00% | 100.00% |
| F1-score | 100.00% | 86.92% | 87.91% |
| False-positive rate | 0.00% | 11.00% | 11.00% |

Random Forest metrics are based on three-class classification. Isolation Forest and hybrid metrics are based on binary attack detection.

## Reproducibility Scope

The supplied feature-level datasets support reproduction of the core model training and evaluation results.

Network simulation, raw dataset construction, feature extraction, and certain supplementary experiments require additional laboratory resources not included in this repository.

## Research Limitations

The results are based on simulated campus DNS traffic and do not establish equivalent performance on operational campus networks.

The hybrid framework achieved 100% recall on the tested attack observations but produced an 11% false-positive rate. The Normal traffic scenario accounted for 98 of the 110 false positives.

Further validation using operational DNS traffic and genuinely unseen attack families is required.

## Research Article

**Design and Implementation of a Lightweight Hybrid Machine Learning Framework for DNS Anomaly Detection in Campus IoT Environments**

Publication details will be added when available.

