# Results

MAGIC (OpenML 1120): 19020 samples, 10 features; normal (gamma) 12332, anomaly (hadron) 6688 (35.2% anomalous)

3 seeds (mean ± std). Each seed changes the data split, the client partition and the initialisation. Test labels are never used for model selection or the threshold; both come from a separate validation set. ε is the total budget after all rounds at δ=1e-5 (worst-case client, sample-level DP, RDP accountant).

| Partition | Method | ROC AUC | AUPRC | F1 | ε |
|---|---|---|---|---|---|
| none | centralized | 0.861 ± 0.007 | 0.811 ± 0.008 | 0.721 ± 0.014 | – |
| iid | local-only | 0.837 ± 0.006 | 0.782 ± 0.007 | 0.686 ± 0.011 | – |
| iid | fedavg | 0.826 ± 0.013 | 0.769 ± 0.013 | 0.675 ± 0.015 | – |
| iid | fedavg+dp nm=0.5 | 0.801 ± 0.028 | 0.739 ± 0.023 | 0.645 ± 0.025 | 43.29 |
| iid | fedavg+dp nm=1.0 | 0.774 ± 0.010 | 0.716 ± 0.005 | 0.612 ± 0.001 | 7.13 |
| iid | fedavg+dp nm=2.0 | 0.771 ± 0.006 | 0.713 ± 0.004 | 0.613 ± 0.001 | 2.43 |
| noniid | fedavg mu=10 | 0.777 ± 0.012 | 0.706 ± 0.012 | 0.614 ± 0.003 | – |
| noniid | fedavg mu=1 | 0.776 ± 0.009 | 0.695 ± 0.018 | 0.620 ± 0.006 | – |
| noniid | fedavg mu=0 | 0.775 ± 0.009 | 0.695 ± 0.017 | 0.614 ± 0.013 | – |
| noniid | fedavg+dp mu=0 | 0.764 ± 0.013 | 0.701 ± 0.013 | 0.596 ± 0.015 | 7.13 |
| noniid | fedavg+dp mu=1 | 0.763 ± 0.014 | 0.700 ± 0.014 | 0.594 ± 0.019 | 7.13 |
| noniid | fedavg+dp mu=10 | 0.757 ± 0.022 | 0.694 ± 0.024 | 0.599 ± 0.023 | 7.13 |
| noniid | local-only | 0.733 ± 0.003 | 0.651 ± 0.015 | 0.560 ± 0.006 | – |
