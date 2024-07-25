import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import numpy as np

#MARKER_ORDER = ['^', 'o', 'v', 'D', 's',]
MARKER_ORDER = ['^', '<', 'v', 'd', 's', 'x', 'o', '>']

# Taken from the answer here https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth_arr(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def smooth_data(df, smooth_vals, value, gp_keys=['method', 'run']):
    gp_df = df.groupby(gp_keys)

    ret_dfs = []
    if not isinstance(smooth_vals, dict):
        smooth_vals = {'default': float(smooth_vals)}
    for sub_df in [gp_df.get_group(k) for k in gp_df.indices]:
        df_method_name = sub_df['method'].iloc[0]
        if isinstance(df_method_name, pd.Series):
            df_method_name = df_method_name.tolist()[0]

        if df_method_name in smooth_vals:
            smooth = smooth_vals[sub_df['method'][0]]
        else:
            smooth = smooth_vals['default']
        use_df = sub_df.copy()
        use_df = use_df.dropna()
        use_df[value] = smooth_arr(use_df[value].tolist(), smooth)

        ret_dfs.append(use_df)

    data = pd.concat(ret_dfs, ignore_index=True)
    return data


def make_steps_match(plot_df, group_key, x_name):
    all_dfs = []
    for method_name, method_df in plot_df.groupby([group_key]):
        grouped_runs = method_df.groupby(['run'])
        max_len = -1
        max_step_idxs = None
        for run_name, run_df in grouped_runs:
            if len(run_df) > max_len:
                max_len = len(run_df)
                max_step_idxs = run_df[x_name]
        for run_name, run_df in grouped_runs:
            run_df[x_name] = max_step_idxs[:len(run_df)]
            all_dfs.append(run_df)
    return pd.concat(all_dfs)


def uncert_plot(plot_df, ax, x_name, y_name, avg_key, group_key, smooth_factor,
                y_bounds=None, y_disp_bounds=None, x_disp_bounds=None,
                group_colors=None, xtick_fn=None, ytick_fn=None, legend=False,
               rename_map={}, title=None, axes_font_size=14, title_font_size=18,
               legend_font_size='x-large', method_idxs={}, num_marker_points={},
               line_styles={}, tight=False, nlegend_cols=1, fetch_std=False):
    """
    - num_marker_points dict: Key maps method name to the number of markers drawn on the line, NOT the
      number of points that are plotted! By default this is 8.
    """
    plot_df = plot_df.copy()
    if tight:
        plt.tight_layout(pad=2.2)
    if group_colors is None:
        methods = plot_df[group_key].unique()
        colors = sns.color_palette()
        group_colors = {method: color for method, color in zip(methods, colors)}

    avg_y_df = plot_df.groupby([group_key, x_name]).mean()
    std_y_df = plot_df.groupby([group_key, x_name]).std()
    method_runs = plot_df.groupby('method')['run'].unique()
    if fetch_std:
        y_std = y_name+'_std'
        new_df = []
        for k, sub_df in plot_df.groupby([group_key]):
            where_matches = avg_y_df.index.get_level_values(0) == k

            use_df = avg_y_df[where_matches]
            if np.isnan(sub_df.iloc[0][y_std]):
                use_df['std'] = std_y_df[where_matches][y_name]
            else:
                use_df['std'] = avg_y_df[where_matches][y_std]
            new_df.append(use_df)
        avg_y_df = pd.concat(new_df)
    else:
        avg_y_df['std'] = std_y_df[y_name]

    lines = []
    names = []

    # Update the legend info with any previously plotted lines
    if ax.get_legend() is not None:
        all_lines = ax.get_lines()

        for i, (l, n) in enumerate(zip(ax.get_legend().get_lines(), ax.get_legend().get_texts())):
            names.append(n.get_text())
            lines.append((all_lines[i*2+1],all_lines[i*2]))

    for name, sub_df in avg_y_df.groupby(level=0):
        names.append(name)
        #sub_df = smooth_data(sub_df, smooth_factor, y_name, [group_key, avg_key])
        x_vals = sub_df.index.get_level_values(x_name).to_numpy()
        y_vals = sub_df[y_name].to_numpy()

        if x_disp_bounds is not None:
            use_y_vals = sub_df[sub_df.index.get_level_values(x_name) < x_disp_bounds[1]][y_name].to_numpy()
        else:
            use_y_vals = y_vals
        print(f"{name}: n_seeds: {len(method_runs[name])} (from WB run IDs {list(method_runs[name])})", max(use_y_vals), use_y_vals[-1])
        y_std = sub_df['std'].fillna(0).to_numpy()
        if isinstance(smooth_factor, dict):
            use_smooth_factor = (smooth_factor[name]
                    if name in smooth_factor else smooth_factor['default'])
        else:
            use_smooth_factor = smooth_factor
        if use_smooth_factor != 0.0:
            y_vals = np.array(smooth_arr(y_vals, use_smooth_factor))
            y_std = np.array(smooth_arr(y_std, use_smooth_factor))

        add_kwargs = {}
        if name in line_styles:
            add_kwargs['linestyle'] = line_styles[name]
        l = ax.plot(x_vals, y_vals, **add_kwargs)
        sel_vals = [int(x) for x in np.linspace(0, len(x_vals)-1,
            num=num_marker_points.get(name, 8))]
        midx = method_idxs[name] % len(MARKER_ORDER)
        ladd = ax.plot(x_vals[sel_vals], y_vals[sel_vals], MARKER_ORDER[midx],
                label=rename_map.get(name, name), color=group_colors[name],
                markersize=8)

        lines.append((ladd[0], l[0]))

        plt.setp(l, linewidth=2, color=group_colors[name])
        min_y_fill = y_vals - y_std
        max_y_fill = y_vals + y_std

        if y_bounds is not None:
            min_y_fill = np.clip(min_y_fill, y_bounds[0], y_bounds[1])
            max_y_fill = np.clip(max_y_fill, y_bounds[0], y_bounds[1])

        ax.fill_between(x_vals,
                        min_y_fill,
                        max_y_fill,
                        alpha=0.2, color=group_colors[name])
    if y_disp_bounds is not None:
        ax.set_ylim(*y_disp_bounds)
    if x_disp_bounds is not None:
        ax.set_xlim(*x_disp_bounds)

    if xtick_fn is not None:
        plt.xticks(ax.get_xticks(), [xtick_fn(t) for t in ax.get_xticks()])
    if ytick_fn is not None:
        plt.yticks(ax.get_yticks(), [ytick_fn(t) for t in ax.get_yticks()])

    if legend:
        labs = [(i, l[0].get_label()) for i, l in enumerate(lines)]
        labs = sorted(labs, key=lambda x: method_idxs[names[x[0]]])
        plt.legend([lines[i] for i, _ in labs], [x[1] for x in labs],
                fontsize=legend_font_size, ncol=nlegend_cols)

    ax.grid(b=True, which='major', color='lightgray', linestyle='--')

    ax.set_xlabel(rename_map.get(x_name, x_name), fontsize=axes_font_size)
    ax.set_ylabel(rename_map.get(y_name, y_name), fontsize=axes_font_size)
    if title is not None and title != "":
        ax.set_title(title, fontsize=title_font_size)

def high_res_save(save_path):
    file_format = save_path.split('.')[-1]
    plt.savefig(save_path, format=file_format, dpi=1000, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

def gen_fake_data(x_scale, data_key, n_runs=5):
    def create_sigmoid():
        noise = np.random.normal(0, 0.01, 100)
        x = np.linspace(0.0, 8.0, 100)
        y = 1/(1 + np.exp(-x))
        y += noise
        return x,y
    df = None
    for i in range(n_runs):
        x, y = create_sigmoid()
        sub_df = pd.DataFrame({'_step': [int(x_i * x_scale) for x_i in x], data_key: y})
        sub_df['run'] = f"run_{i}"
        if df is None:
            df = sub_df
        else:
            df = pd.concat([df, sub_df])
    df['method'] = 'fake'
    return df

