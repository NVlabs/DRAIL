## Running Jobs
### SLURM Helper
The SLURM helper is only activated when you are running `--sess-id X` and
specify a value for `--st`. When `habitat_baselines` is in the command a
`sh` file will be created with the command and then `sbatch` will be run.
`--nodes` does not affect non-sbatch runs. 

- `--cd -1` does not set the CUDA environment variable (which is the default). This is helpful on
  machines where you shouldn't mess with this setting. 

## Setting up W&B
- First log into W&B: `wandb login ...`


## Getting Data From W&B
1. To get data from a particular run (where you know the name of the run) use
  `get_run_data`. You can specify a list of runs you want to get data for. 
2. To get data for data sources in a report you can use `get_report_data`. When accessing reports, you need look up the report by **description, not name**. So if you want to get a report called "my-report" from the code, the description of the report on W&B should be "ID:my-report". 

## config.yaml
The settings that need to go into `config.yaml` are:
- `cmds_loc`
- `wb_proj_name`

## Plotting 
To plot, use `auto_plot.py`. This will automatically fetch and plot runs from
reports on W&B. It has support for plotting horizontal lines, specifying the
color, axes, and which key to plot. **The report on W&B has to follow [this
naming convention from point
2](https://github.com/ASzot/rl-toolkit/tree/master/rlf/exp_mgr#getting-data-from-wb)**.
Typically this is run as `python rl-toolkit/rlf/exp_mgr/auto_plot.py --plot-cfg
my_plot_cfgs/plot.yaml`. 

Documentation of the plot settings YAML file. Most properties in the
`plot_sections` element can also be set in the base element as a default. 
```
---
plot_sections:
    - save_name: [string, filename to save to (not including extension or directory)]
      should_save: [bool, If False, then the plot is not saved, and is apart of the next rendered plot, this is how you can compose data from multiple data sources.]

      # Data source settings
      report_name: [string, either the TB search directory or the W&B report description ID]
      line_report_name: [string, W&B ID or TB directory of where to get data for rendering lines, if nothing, then same as report name]

      # Render settings
      plot_title: [string, overhead title of plot]
      y_bounds: [string, like "0.0,100.0" clips values to this range]
      x_disp_bounds: [string, like "0,1e8" plot display bounds]
      y_disp_bounds: [string, like "60,102" plot display bounds ]
      legend: [bool, If true, will render the legend within the plot]
      nlegend_cols: [int, # of columns in the legend, useful for large legends]

      make_steps_similar: [bool, if True, will forcefully align the steps in the runs for a particular method]
      plot_key: "eval_metrics/ep_success"

      line_match_pat: null
      line_sections:
          - 'mpg'
          - 'mpp2'
      plot_sections:
          - "D-s eval"
          - "D eval"
          - "RGB-s eval"
          - "RGB eval"
          - "RGBD-s eval"
          - "RGBD eval"
      force_reload: False
global_renames: [dict str -> str]
    "eval_metrics/ep_success": "Success (%)"
    _step: "Step"
    "eval_metrics_ep_success": "Success (%)"

    "D-s eval": "D+ps"
    "D eval": "D"
    "RGB-s eval": "RGB+ps"
    "RGB eval": "RGB"
    "RGBD-s eval": "RGBD+ps"
    "RGBD eval": "RGBD"

    "RGBD_s": 'RGBD+ps'
    "input_ablation_RGBD_g": 'RGBD'
    "input_ablation_RGB_s": 'RGB+ps'
    "input_ablation_D_s": 'D+ps'
    "input_ablation_s": 'ps'
    "input_ablation_RGB_g": "RGB"
    "input_ablation_D_g": "D"
    "mpp": "SPA"
    "mpg": "SPA-Priv"
    "mpp2": "SPA"
line_plot_key: "eval_metrics/ep_success"
line_val_key: "eval_metrics/ep_success"
smooth_factor: 0.8
scale_factor: 100
line_op: 'max'
config_yaml: "./config.yaml"
save_loc: "./data/plot"
fig_dims: [6,4]
legend_font_size: 'medium'
colors:
  "RGBD_s": 0
  "input_ablation_RGB_s": 1
  "input_ablation_D_s": 2
  "input_ablation_s": 3
  "input_ablation_RGBD_g": 4
  "input_ablation_RGB_g": 5
  "input_ablation_D_g": 6
  "mpg": 7
  "mpp": 8
  "mpp2": 8

  "D-s eval": 0
  "D eval": 1
  "RGB-s eval": 2
  "RGB eval": 3
  "RGBD-s eval": 4
  "RGBD eval": 5
```


## Plotting Utilities
This can help make good looking plots. 
* Line plots use `uncert_plot` from `rlf/exp_mgr/plotter.py`. 
=======
There is also a utility for creating a separate PDF file containing just the
legend. This is run as `python rl-toolkit/rlf/exp_mgr/auto_plot.py --plot-cfg
my_plot_cfgs/my_legend.yaml` --legend. An example of a legend YAML file is
below. The marker attributes refer to the characteristics of the line next to
the name. 

```
---
plot_sections:
    ablate: "ours,gaifo,gaifo-s"
save_loc: "./data/plot/figs/final"
marker_size: 12
marker_width: 0.0
marker_darkness: 0.1
line_width: 3.0
name_map:
    ours: "Ours"
    gaifo: "GAIfO"
    gail-s: "GAIfO-s"
colors:
    ours: 0 
    gaifo: 1
    gail-s: 2
```
