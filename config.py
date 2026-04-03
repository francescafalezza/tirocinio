audio:
target_sr: 44100
  lowcut: 40
  highcut: 16000
  filter_order: 5

estimation:
  method: "schroeder"
  min_db: -5
  max_db: -35

output:
  save_plots: true
  plots_dir: "results/plots"