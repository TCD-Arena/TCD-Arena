import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes,
)  # Import the inset_axes function
from matplotlib.projections import polar  # Import the polar projection module directly


# --- CONFIG ---
groups = [7,2,3,4,9,2,6]

# --- NAMES (With LaTeX fixes) ---
group_names = [       
    "Observational\nNoise",    
    "Causal\nSufficiency",
    "Faithfulness",         
    "Nonlinearity",
    "Innovation\nNoise",     
    "Stationarity",
    "Data\nQuality"           

]

# Background Tints (Pastel versions of Orange, Purple, Blue, Green, Red)
# Using 'xx' at end of hex or using these defaults
bg_tints = ['#588B8B', "#D89BDB", '#FFD5C2', '#F28F3B', '#C8553D', "#2D3047", "#93B7BE"]


# We use these to remove performance scores from the table
performance_scores = [
    "AUROC",
    "F1", 
    "ACC",
    "SHD",
    "Acc",
    "runtime",
    "path"
]


# Make a dictionary from this an translate them to proper names:
def subscript(text):
    # Return a LaTeX string for subscripted text, e.g., V_{ino,auto}
    return r"\textbf{V}_{\mathrm{" + text + r"}}"





rename_labels = {
    # ino
    "inno_auto": rf"${subscript('inno,auto')}$",
    "inno_common": rf"${subscript('inno,com')}$",
    "inno_multiplicative": rf"${subscript('inno,mul')}$",
    "inno_shock": rf"${subscript('inno,shock')}$",
    "inno_real": rf"${subscript('inno,real')}$",
    "inno_time": rf"${subscript('inno,time')}$",
    "inno_weib": rf"${subscript('inno,weib')}$",
    "inno_uni": rf"${subscript('inno,uni')}$",
    "inno_var": rf"${subscript('inno,var')}$",

    # missing / missing mechanisms (big/small variants)
    "mcar": rf"${subscript('mcar')}$",
    "mnar": rf"${subscript('mnar')}$",
    "mar": rf"${subscript('mar')}$",

    # coefficients / effect-size variants
    "coef_n": rf"${subscript('coef')}$",

    # nl
    "nl_mono": rf"${subscript('mono')}$",
    "nl_comp": rf"${subscript('comp')}$",
    "nl_trend": rf"${subscript('trend')}$",
    "nl_rbf": rf"${subscript('rbf')}$",

    # obs
    "obs_auto": rf"${subscript('obs,auto')}$",
    "obs_com": rf"${subscript('obs,com')}$",
    "obs_real": rf"${subscript('obs,real')}$",
    "obs_mul": rf"${subscript('obs,mul')}$",
    "obs_shock": rf"${subscript('obs,shock')}$",
    "obs_add": rf"${subscript('obs,add')}$",
    "obs_time": rf"${subscript('obs,time')}$",

    # remaining / structural / data representation
    "missing_info": rf"${subscript('empty')}$",
    "empty": rf"${subscript('empty')}$",
    "length": rf"${subscript('length')}$",
    "lagged_confounder": rf"${subscript('conf,l')}$",
    "conf_l": rf"${subscript('conf,l')}$",
    "conf_i": rf"${subscript('conf,i')}$",
    "interpolate_nl": rf"${subscript('miss')}$",
    "faith_lagged": rf"${subscript('faith,l')}$",
    "faith_inst": rf"${subscript('faith,i')}$",
    "faith_z": rf"${subscript('faith,z')}$",
    "unequal_var_t_n": rf"${subscript('var')}$",
    "nonstat_n": rf"${subscript('stat')}$",
    "stat": rf"${subscript('stat')}$",
    "standardization": rf"${subscript('scale')}$",
    "scale": rf"${subscript('scale')}$",
    
    #double
    "double_common": rf"${subscript('double,com')}$",
    "conf_double": rf"${subscript('conf,double')}$",
    
    #Individual
    "inno_common_small": rf"${subscript('inno,com,small')}$",
    "inno_common_big": rf"${subscript('inno,com,big')}$",
    "inno_common_large": rf"${subscript('inno,com,large')}$",
    "nl_rbf_small": rf"${subscript('inno,rbf,small')}$",

    }


def get_small_big_sets(paths):
    """
    Check if the small and big sets are available for each method.
    """
    fully_available = []
    ready_for_plot = []
    bases = ["_".join(x.split("_")[:-1]) for x in paths]
    # count the number of entries for each individual base:
    base_counts = {}
    for base in bases:
        if base not in base_counts:
            base_counts[base] = 0
        base_counts[base] += 1

    for x in base_counts.keys():
        if base_counts[x] > 1:
            fully_available.append(x)
       
    for item in fully_available:
        ready_for_plot.append([item + "_small.csv", item + "_big.csv"])

    return ready_for_plot


def return_violation_property_and_order(to_process):
    violation_property_dict = {}
    for x in to_process:
        x = str(x)
        if "obs_" in x: 
            violation_property_dict[x] = ("generator.obs_n.snr", False)
        elif "inno_" in x: 
            if "inno_var" in x: 
                violation_property_dict[x] = ("generator.inno_n.non_equal_variance_range", False)
            elif ("weib" not in x) and ("uni" not in x):
                violation_property_dict[x] = ("generator.inno_n.non_additive_noise_proba", True)
            else:
                violation_property_dict[x] = ("generator.inno_n.non_gaussian_additive", True)
        elif "length" in x: 
            violation_property_dict[x] = ("generator.time_series_n", False)
        elif "interpolate" in x:
            violation_property_dict[x] = ("generator.interpolate"	, True)
        elif "nl_trend" in x:
            violation_property_dict[x] = ("spline_samples", False)
        elif "nl_mono" in x:
            violation_property_dict[x] = ("power_dist", False)
        elif "splines" in x:
            violation_property_dict[x] = ("spline_samples", False)
        elif "nl_" in x: 
            violation_property_dict[x] = ("nonlinear_proba", True)
        elif "coef_n" in x: 
            violation_property_dict[x] = ("nonstationary_change", True)
        elif "stat" in x: 
            violation_property_dict[x] = ("generator.change_points", True)
        elif "conf_l" in x: 
            violation_property_dict[x] = ("generator.lagged.alternative_link_proba", True)
        elif "empty" in x: 
            violation_property_dict[x] = ("generator.change_points", True)
        elif "scale" in x: 
            violation_property_dict[x] = ("generator.standardization_factor", True)
        elif "mar_" in x: 
            violation_property_dict[x] = ("generator.interpolate", True)
        elif "mnar_" in x: 
            violation_property_dict[x] = ("generator.interpolate", True)
        elif "mcar_" in x: 
            violation_property_dict[x] = ("generator.interpolate", True)        
        elif "standard" in x: 
            violation_property_dict[x] = ("generator.standardization_factor", True)
        elif "nonstat_n" in x: #! 
            violation_property_dict[x] = ("generator.change_points", True)
        elif "missing_info" in x:  #!
            violation_property_dict[x] = ("generator.change_points", True) # This doesnt order properly. Ignore the graph.
        elif "faith_lagged" in x:
            violation_property_dict[x] = ("link_mask_path", False)
        elif "conf_i" in x:
            violation_property_dict[x] = ("generator.exog.link_proba", True)
        elif "faith_z" in x:
            violation_property_dict[x] = ("generator.lagged.param_range", False)
        elif "faith_inst" in x:
            violation_property_dict[x] = ("instant_link_mask_path", False)
        elif "lagged_confounder" in x: 
            violation_property_dict[x] = ("generator.struc.alternative_link_proba", True)
        elif "instant_confounder" in x:
            violation_property_dict[x] = ("generator.exog.link_proba", True)
        else:
            print("NOT ASSIGNED: ", x)
    return violation_property_dict


def matplot_styling(tight_layout=True):

    # Plotting preliminaries
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 12,  # 18
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
        }
    )
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amssymb}\usepackage{wasysym}")
    if tight_layout:
        plt.tight_layout()
    


cmap = matplotlib.colormaps["Paired"]
colors1 = [cmap(x) for x in range(0, 12)]

color_map = {
    "GVAR": "#E7969C",
    "Varlingam": "#AF1B1D",
    "CausalPretraining": "#DBC20F",
    "CrossCorrelation": colors1[3],
    "PCMCI+": colors1[1],
    "PCMCI": colors1[0],
    "Dynotears": colors1[6],
    "Nts-Notears": colors1[7],
    "SVAR-RFCI": colors1[8],
    "F-PCMCI": "#090770",
}

marker_map = {
    "GVAR": "o",
    "Varlingam": "s",
    "CausalPretraining": "D",
    "CrossCorrelation": "^",
    "PCMCI+": "v",
    "PCMCI": "<",
    "Dynotears": ">",
    "Nts-Notears": "P",
    "SVAR-RFCI": "X",
    "F-PCMCI": "H",
}

renames = {
    "cp": "CausalPretraining",
    "pcmciplus": "PCMCI+",
    "varlingam": "Varlingam",
    "direct_crosscorr": "CrossCorrelation",
    "dynotears": "Dynotears",
    "pcmci": "PCMCI",
    "var": "GVAR",
    "physical": "Physical",
    "crosscorr": "CC-Peak",
    "combo": "CC-Peak+Phys",
    "ntsnotears": "Nts-Notears",
    "svarrfci": "SVAR-RFCI",
    "fpcmci": "F-PCMCI",
}

ordering = [
    "CrossCorrelation",
    "CausalPretraining",
    "GVAR",
    "Varlingam",
    "PCMCI",
    "PCMCI+",
    "SVAR-RFCI",
    "F-PCMCI",
    "Dynotears",
    "Nts-Notears",
]

exp_ordering = [
    "obs_add",
    "obs_mul",
    "obs_time",
    "obs_auto",
    "obs_com",
    "obs_shock",
    "obs_real",
    "conf_i",
    "conf_l",
    "faith_inst",
    "faith_lagged",
    "faith_z",
    "nl_mono",
    "nl_trend",
    "nl_rbf",
    "nl_comp",
    "inno_multiplicative",
    "inno_time",
    "inno_auto",
    "inno_common",
    "inno_shock",
    "inno_real",
    "inno_uni",
    "inno_weib",
    "inno_var",
    "coef_n",
    "stat",
    "length",
    "mar",
    "mnar",
    "mcar",
    "scale",
    "empty",
]


def new_base_radar(
    ax,
    out,
    ylim=[0.5, 1],
    outer_grid_color="#222222",
    tick_colors="#222222",
    gridline_color="#AAAAAA",
    label_font_size=8,
    label_pad=12,
    bg_color="#FAFAFA",
    axis_rotation=2.17,
    invert_axis=False,
):
    # Each attribute we'll plot in the radar chart.
    labels = out.columns


    labels = [rename_labels[x] if x in rename_labels.keys() else x for x in labels]
    # Number of variables we're plotting.
    num_vars = len(labels)
    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / axis_rotation)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles)[:-1], labels)

    # # Go through labels and adjust alignment based on where
    # # it is in the circle.
    # for label, angle in zip(ax.get_xticklabels(), angles):
    #     if angle in (0, np.pi):
    #         label.set_horizontalalignment("center")
    #     elif 0 < angle < np.pi:
    #         label.set_horizontalalignment("left")
    #     else:
    #         label.set_horizontalalignment("right")

    # Enforce Y lim on every axis.
    if invert_axis:
        ax.set_ylim(float(ylim[1]), float(ylim[0]))
    else:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    # You can also set gridlines manually like this:
    # ax.set_rgrids([ylim[0], 0.7, ylim[1]])

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes.
    ax.set_rlabel_position(30)
    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors=tick_colors)
    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=label_font_size, pad=label_pad)

    for label in ax.get_xticklabels():
        label.set_clip_on(False)

    # Change the color of the circular gridlines.
    ax.grid(color=gridline_color, linestyle="--", linewidth=0.25)
    # Change the color of the outermost gridline (the spine).
    ax.spines["polar"].set_color(outer_grid_color)
    # Change the background color inside the circle itself.
    ax.set_facecolor(bg_color)

    # Add title.
    return ax, angles


def make_circle(
    sections=[[0.0], [90.0], [180.0], [240.0], [300]],
    donut_labels=[
        "Innovation Noise",
        "Observational Noise",
        "Nonlinearity",
        "Graph Structure",
        "Data Rerepresentation",
    ],
    figsize=(10, 10),
    radar_max_plot_radius=1.0,
    donut_thickness=0.2,
    gap_between_radar_and_donut=0.1,
    donut_label_fontsize=9,
    donut_label_color="black",
    donut_label_weight="bold",
    donut_alpha=0.8,
    donut_edgecolor="w",
    donut_linewidth=1,
    donut_zorder=3,
    cmap=plt.colormaps["Pastel1"],
    hardcode_label_angle=[0,0,0,0,0],
    offset_circle =  0.11635528346628864,
    x_postions_label = [0,0.5,0.6,0.7,0.8],
    y_positions_label = [0,0.5,0.6,0.7,0.8]
):
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=figsize)

    donut_inner_radius = radar_max_plot_radius + gap_between_radar_and_donut
    donut_outer_radius = donut_inner_radius + donut_thickness

    vals = np.array(sections)
    valsnorm = vals / np.sum(vals) * 2 * np.pi
    valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)
    outer_colors = cmap(np.arange(vals.shape[0]) % cmap.N)
    ax.bar(
        x=valsleft[:, 0] - offset_circle ,
        width=valsnorm.sum(axis=1),
        bottom=donut_inner_radius,
        height=donut_thickness,
        color=outer_colors,
        edgecolor=donut_edgecolor,
        linewidth=donut_linewidth,
        align="edge",
        alpha=donut_alpha,
        zorder=donut_zorder,
    )

    for i in range(vals.shape[0]):
        ax.text(
            x_postions_label[i],
            y_positions_label[i],
            donut_labels[i],
            ha="center",
            va="center",
            color=donut_label_color,
            fontsize=donut_label_fontsize,
            rotation=hardcode_label_angle[i],
            weight=donut_label_weight,
            zorder=1,
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_rlim(0, donut_outer_radius)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_axis_off()

    return fig, ax