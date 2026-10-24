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
git clone https://github.com/samkih1/campus-iot-dns-anomaly-detection.git
cd campus-iot-dns-anomaly-detection
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

Random Forest performance is reported using three-class classification, while Isolation Forest and hybrid performance are evaluated using binary attack-detection metrics. These differences should be considered when interpreting the results.

Random Forest achieved 100% accuracy on the simulated independent test dataset. The hybrid framework incorporated an additional Isolation Forest screening stage for observations initially classified as benign, achieving 100% recall on the tested attack observations but introducing an 11% false-positive rate.

The results do not demonstrate an overall improvement over Random Forest on this dataset. Instead, they demonstrate the operation of the sequential hybrid decision logic and highlight the trade-off between anomaly screening and false-positive detection.
## Reproducibility Scope

The repository provides sanitised feature-level datasets and Python scripts supporting reproduction of the core Random Forest, Isolation Forest, and sequential hybrid evaluation experiments.

The supplied development dataset contains 4,400 observations, while the independently generated simulated test dataset contains 1,400 observations. These datasets support evaluation of the models using the seven DNS behavioural features implemented in the framework.

The repository also contains supplementary scripts associated with the broader experimental workflow. Reproducing the complete network simulation, packet capture, raw traffic processing, and certain supplementary experiments requires additional laboratory resources and datasets that are not included in this repository.

## Research Limitations

The evaluation was conducted using simulated campus DNS traffic. Although the independent test dataset was generated separately from the development dataset, the results do not establish equivalent performance in operational campus networks.

The hybrid framework achieved 100% recall on the tested attack observations but produced an 11% false-positive rate. Of the 110 false positives, 98 occurred in the Normal traffic scenario. This indicates a need for further refinement of anomaly screening to reduce incorrect classification of benign DNS behaviour.

The `Anomalous_Unknown` output represents observations flagged by Isolation Forest as anomalous after being classified as benign by Random Forest. However, the evaluation did not separately establish detection performance against genuinely unseen attack families.

Further validation using operational DNS traffic, previously unseen attack families, and resource-constrained edge platforms is required to assess the framework's generalisability and deployment feasibility.

## Research Article

Title: Design and Implementation of a Lightweight Hybrid Machine Learning Framework for DNS Anomaly Detection in Campus IoT Environments

Author: Samwel Kinuthia Kariuki

Affiliation: Zetech University, Kenya

Publication status: Manuscript in preparation.

This repository contains the implementation, sanitised feature-level datasets, and experimental results supporting the research article.

Journal publication details, DOI, and the recommended citation will be added when available.

