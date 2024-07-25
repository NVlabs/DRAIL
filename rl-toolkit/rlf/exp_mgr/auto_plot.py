import sys
sys.path.insert(0, './')
import argparse
import yaml
from rlf.rl.utils import human_format_int
from rlf.exp_mgr.wb_data_mgr import get_report_data
from rlf.exp_mgr.plotter import uncert_plot, high_res_save, MARKER_ORDER
import matplotlib.pyplot as plt
import os.path as osp
import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-cfg', type=str, required=True)
    parser.add_argument('--legend', action='store_true')
    return parser


def export_legend(ax, line_width, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False,
            loc='lower center', ncol=10, handlelength=2)
    for line in legend.get_lines():
        line.set_linewidth(line_width)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def plot_legend(plot_cfg_path):
    with open(plot_cfg_path) as f:
        plot_settings = yaml.load(f)
        colors = sns.color_palette()
        group_colors = {name: colors[idx] for name, idx in
                plot_settings['colors'].items()}

        for section_name, section in plot_settings['plot_sections'].items():
            fig, ax = plt.subplots(figsize=(5, 4))
            names = section.split(',')
            darkness = plot_settings['marker_darkness']
            for name in names:
                add_kwargs = {}
                if name in plot_settings['linestyles']:
                    linestyle = plot_settings['linestyles'][name]
                    if isinstance(linestyle, list):
                        add_kwargs['linestyle'] = linestyle[0]
                        add_kwargs['dashes'] = linestyle[1]
                    else:
                        add_kwargs['linestyle'] = linestyle

                disp_name = plot_settings['name_map'][name]
                midx = plot_settings['colors'][name] % len(MARKER_ORDER)
                marker = MARKER_ORDER[midx]
                if marker == 'x':
                    marker_width = 2.0
                else:
                    marker_width = plot_settings['marker_width']

                marker_alpha = plot_settings.get('alphas', {}).get(name, 1.0)
                use_color =(*group_colors[name], marker_alpha)
                ax.plot([0], [1], marker=marker, label=disp_name,
                        color=use_color,
                        markersize=plot_settings['marker_size'],
                        markeredgewidth=marker_width,
                        #markeredgecolor=(darkness, darkness, darkness, 1),
                        markeredgecolor=use_color,
                        **add_kwargs)
            export_legend(ax, plot_settings['line_width'],
                    osp.join(plot_settings['save_loc'], section_name + '_legend.pdf'))
            plt.clf()


def get_tb_data(search_name, plot_key, plot_section, force_reload, match_pat,
        other_plot_keys, config):
    import tensorflow as tf

    method_dfs = defaultdict(list)
    run_to_name = {}

    if match_pat is None:
        match_pat = '*/*'

    search_path = osp.join(search_name, match_pat, plot_key, '*.tfevents.*')
    matches = glob.glob(search_path)
    if len(matches) == 0:
        raise ValueError(f"Could not get any matching files for {search_path}")

    for f in matches:
        run = f.split('/')[-3]
        method_parts = run.split('_')
        seed = method_parts[-2]
        method_name = '_'.join(method_parts[:-2])

        if method_name not in plot_section:
            continue

        values = []
        run_names = []
        method_names = []
        steps = []
        for summary in tf.train.summary_iterator(f):
            if len(summary.summary.value) > 0:
                val = summary.summary.value[0].simple_value
                values.append(val)
                steps.append(summary.step)
                run_names.append(run)
                method_names.append(method_name)

        #if method_name not in method_steps or len(method_steps[method_name]) < len(steps):
        #    method_steps[method_name] = steps

        #steps = method_steps[method_name][:len(steps)]
        run_to_name[run] = method_name

        method_dfs[run].append(pd.DataFrame.from_dict({
            'method': method_names,
            'run': run_names,
            plot_key: values,
            '_step': steps
            }))

    if len(method_dfs) == 0:
        raise ValueError(f"Could not find any matching runs in {search_path}")

    combined_method_dfs = {}
    overall_step = []
    method_steps = {}
    for k, dfs in method_dfs.items():
        overall_df = pd.concat(dfs)
        overall_df = overall_df.sort_values('_step')
        overall_df = overall_df.drop_duplicates(subset=['_step'], keep='last')
        method_name = run_to_name[k]

        n_steps = len(overall_df['_step'])

        if method_name not in method_steps or \
                len(method_steps[method_name]) < n_steps:
            method_steps[method_name] = overall_df['_step'].tolist()

        combined_method_dfs[k] = overall_df

    combined_df = None
    for k, df in combined_method_dfs.items():
        # Update with the max # of steps.
        method_name = run_to_name[k]
        df['_step'] = method_steps[method_name][:len(df)]
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df])

    return combined_df

def make_steps_similar(method_dfs, run_to_name):
    combined_method_dfs = {}
    overall_step = []
    method_steps = {}
    for k, dfs in method_dfs:
        #overall_df = pd.concat(dfs)
        overall_df = dfs
        overall_df = overall_df.sort_values('_step')
        overall_df = overall_df.drop_duplicates(subset=['_step'], keep='last')
        method_name = run_to_name[k]

        n_steps = len(overall_df['_step'])

        if method_name not in method_steps or \
                len(method_steps[method_name]) < n_steps:
            method_steps[method_name] = overall_df['_step'].tolist()

        combined_method_dfs[k] = overall_df

    combined_df = None
    for k, df in combined_method_dfs.items():
        # Update with the max # of steps.
        method_name = run_to_name[k]
        df['_step'] = method_steps[method_name][:len(df)]
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df])

    return combined_df


def get_data(search_name, plot_key, plot_section, force_reload, match_pat,
        other_plot_keys, config, is_tb, other_fetch_fields):
    if is_tb:
        return get_tb_data(search_name, plot_key, plot_section,
                force_reload, match_pat, other_plot_keys, config)
    else:
        return get_report_data(search_name, plot_key, plot_section,
                force_reload, match_pat, other_plot_keys, config,
                other_fetch_fields)

def combine_common(plot_df, group_key, x_name):
    all_dfs = []
    for method_name, method_df in plot_df.groupby([group_key]):
        grouped_runs = method_df.groupby(['prefix'])
        combine = defaultdict(dict)
        for run_name, run_df in grouped_runs:
            name = run_name.split('-')[-1]
            last_step = run_df[x_name].max()
            combine[name][last_step] = run_df
        for name, step_df in combine.items():
            key_order = sorted(step_df.keys())
            last_k = None
            for k in key_order:
                if last_k is None:
                    all_dfs.append(step_df[k])
                else:
                    all_dfs.append(step_df[k][step_df[k][x_name] > last_k])
                last_k = k
    return pd.concat(all_dfs)



def plot_from_file(plot_cfg_path):
    with open(plot_cfg_path) as f:
        plot_settings = yaml.load(f)

        colors = sns.color_palette()
        group_colors = {name: colors[idx] for name, idx in
                plot_settings['colors'].items()}

        def get_setting(local, k, local_override=True, defval=None):
            if local_override:
                if k in local:
                    return local[k]
                elif k in plot_settings:
                    return plot_settings[k]
                else:
                    return defval
            else:
                if k in plot_settings:
                    return plot_settings[k]
                else:
                    return local[k]

        fig = None
        for plot_section in plot_settings['plot_sections']:
            plot_key = plot_section.get('plot_key', plot_settings['plot_key'])
            match_pat = plot_section.get('name_match_pat',
                    plot_settings.get('name_match_pat', None))
            print(f"Getting data for {plot_section['report_name']}")
            should_combine = get_setting(plot_section, 'should_combine',
                    defval=False)
            other_plot_keys = plot_settings.get('other_plot_keys', [])
            other_fetch_fields = []
            fetch_std = get_setting(plot_section, 'fetch_std', defval=False)
            if should_combine:
                other_fetch_fields.append('prefix')

            if fetch_std:
                other_plot_keys.append(plot_key + '_std')

            plot_df = get_data(plot_section['report_name'],
                    plot_key,
                    plot_section['plot_sections'],
                    get_setting(plot_section,'force_reload', defval=False),
                    match_pat,
                    other_plot_keys,
                    plot_settings['config_yaml'],
                    plot_section.get('is_tb', False), other_fetch_fields)
            # W&B will sometimes return NaN rows at the start and end of
            # training.
            plot_df = plot_df.dropna()

            if 'line_sections' in plot_section:
                line_plot_key = get_setting(plot_section, 'line_plot_key')
                take_operation = get_setting(plot_section, 'line_op')
                line_val_key = get_setting(plot_section, 'line_val_key')
                if line_plot_key != line_val_key:
                    fetch_keys = [line_plot_key, line_val_key]
                else:
                    fetch_keys = [line_plot_key]
                if fetch_std:
                    fetch_keys.append(line_plot_key+'_std')
                if len(fetch_keys) == 1:
                    fetch_keys = fetch_keys[0]

                line_is_tb = plot_section.get('is_tb', False)
                if 'line_is_tb' in plot_section:
                    line_is_tb = plot_section['line_is_tb']
                line_report_name = plot_section['report_name']
                if 'line_report_name' in plot_section:
                    line_report_name = plot_section['line_report_name']
                line_match_pat = match_pat
                if 'line_match_pat' in plot_section:
                    line_match_pat = plot_section['line_match_pat']
                line_df = get_data(line_report_name,
                        fetch_keys,
                        plot_section['line_sections'],
                        get_setting(plot_section,'force_reload', defval=False),
                        line_match_pat, [],
                        plot_settings['config_yaml'],
                        line_is_tb, other_fetch_fields)
                line_df = line_df.dropna()
                uniq_step = plot_df['_step'].unique()
                use_line_df = None
                for group_name, df in line_df.groupby('run'):
                    if take_operation == 'min':
                        use_idx = np.argmin(df[line_val_key])
                    elif take_operation == 'max':
                        use_idx = np.argmax(df[line_val_key])
                    elif take_operation == 'final':
                        use_idx = -1
                    else:
                        raise ValueError(f"Unrecognized line reduce {take_operation}")
                    df = df.iloc[np.array([use_idx]).repeat(len(uniq_step))]
                    if line_plot_key != line_val_key:
                        del df[line_val_key]

                    df.index = np.arange(len(uniq_step))
                    df['_step'] = uniq_step
                    if use_line_df is None:
                        use_line_df = df
                    else:
                        use_line_df = pd.concat([use_line_df, df])
                rename_dict = {line_plot_key: plot_key}
                if fetch_std:
                    rename_dict[line_plot_key+'_std'] = plot_key+'_std'
                use_line_df = use_line_df.rename(columns=rename_dict)
                plot_df = pd.concat([plot_df, use_line_df])

            use_fig_dims = plot_section.get('fig_dims', plot_settings.get('fig_dims', (5,4)))
            if fig is None:
                fig, ax = plt.subplots(figsize=use_fig_dims)
            def get_nums_from_str(s):
                return [float(x) for x in s.split(',')]

            local_renames = {}
            if 'renames' in plot_section:
                local_renames = plot_section['renames']

            title = plot_section['plot_title']
            if 'scale_factor' in plot_settings:
                plot_df[plot_key] *= plot_settings['scale_factor']
                if fetch_std:
                    plot_df[plot_key+'_std'] *= plot_settings['scale_factor']
            use_legend_font_size = plot_section.get('legend_font_size',
                    plot_settings.get('legend_font_size', 'x-large'))

            if should_combine:
                plot_df = combine_common(plot_df, 'method', '_step')
            run_grouped = plot_df.groupby('run')
            name_map = {x['run']: x['method'] for i, x in plot_df.iterrows()}
            if get_setting(plot_section, 'make_steps_similar', defval=False):
                plot_df = make_steps_similar(run_grouped,name_map)

            uncert_plot(plot_df, ax, '_step', plot_key, 'run', 'method',
                    get_setting(plot_section, 'smooth_factor'),
                    y_bounds=get_nums_from_str(plot_section['y_bounds']),
                    x_disp_bounds=get_nums_from_str(
                        get_setting(plot_section,'x_disp_bounds')),
                    y_disp_bounds=get_nums_from_str(plot_section['y_disp_bounds']),
                    xtick_fn=human_format_int,
                    legend=plot_section['legend'],
                    title=title,
                    group_colors=group_colors,
                    method_idxs=plot_settings['colors'],
                    tight=True,
                    legend_font_size=use_legend_font_size,
                    num_marker_points=plot_settings.get('num_marker_points', {}),
                    line_styles=plot_settings.get('linestyles', {}),
                    nlegend_cols=plot_section.get("nlegend_cols", 1),
                    fetch_std=fetch_std,
                    rename_map={
                        **plot_settings['global_renames'],
                        **local_renames,
                        })
            if plot_section.get("should_save", True):
                save_loc = plot_settings['save_loc']
                if not osp.exists(save_loc):
                    os.makedirs(save_loc)
                save_path = osp.join(save_loc, plot_section['save_name'] + '.pdf')
                high_res_save(save_path)
                plt.clf()
                fig = None

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.legend:
        plot_legend(args.plot_cfg)
    else:
        plot_from_file(args.plot_cfg)
