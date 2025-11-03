# Experiment 1 Utilities

## Export epoch metrics to CSV

Convert each `run-simple-*.log` into a tabular summary (columns: epoch, loss, PSNR, optional validation metrics):

```bash
python3 Super_resolution/code/export_log_metrics.py \
  --logs-root Super_resolution/logs/experiment_1 \
  --output-root Super_resolution/experiments/experiment_1_constant_depth_3/csv_logs
```

The command writes one `epoch_metrics.csv` per run inside `csv_logs/<run_id>/`.

## Plot summary figures

```bash
python3 Super_resolution/code/analyse_experiment_metrics.py \
  --csv-root Super_resolution/experiments/experiment_1_constant_depth_3/csv_logs \
  --output-dir Super_resolution/experiments/experiment_1_constant_depth_3
```

This renders the loss/PSNR trend, training speed, and training load PNGs beside
the CSV exports.
