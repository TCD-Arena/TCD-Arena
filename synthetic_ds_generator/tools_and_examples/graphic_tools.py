import matplotlib
import matplotlib.pyplot as plt


# Some design choices
cmap = plt.get_cmap("coolwarm")
colors = [cmap(i / 4) for i in range(5)]


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



# Make a dictionary from this an translate them to proper names:
def subscript(text):
    # Return a LaTeX string for subscripted text with upright (non-italic) font
    # e.g. produces a bold V with an upright subscript like V_{ino,auto}
    return r"$\mathbf{V}_{\mathrm{" + text + "}}$"


rename_labels = {
    # ino
    "inno_auto": subscript('inno,auto'),
    "inno_common": subscript('inno,com'),
    "inno_mul": subscript('inno,mul'),
    "inno_shock": subscript('inno,shock'),
    "inno_time": subscript('inno,time'),
    "inno_weibull": subscript('inno,weib'),
    "inno_uniform": subscript('inno,uni'),
    "inno_var": subscript('inno,var'),
    "inno_real": subscript('inno,real'),

    # nl
    "nl_power": subscript('nl,mono'),
    "nl_splines": subscript('nl,trend'),
    "nl_symbolic": subscript('nl,comp'),
    "nl_rbf": subscript('nl,rbf'),
    # obs
    "obs_auto": subscript('obs,auto'),
    "obs_common": subscript('obs,com'),
    "obs_mul": subscript('obs,mul'),
    "obs_shock": subscript('obs,shock'),
    "obs_add": subscript('obs,add'),
    "obs_time": subscript('obs,time'),
    "obs_real": subscript('obs,real'),
    # remaining
    "missing_info": subscript('empty'),
    "length": subscript('length'),
    "lagged_confounder": subscript('conf,l'),
    "instant_confounder": subscript('conf,i'),
    "mcar": subscript('mcar'),
    "mar": subscript('mar'),
    "mnar": subscript('mnar'),
    "faith_lagged": subscript('faith,l'),
    "faith_inst": subscript('faith,i'),
    "faith_zero": subscript('faith,z'),
    "nonstat_n": subscript('stat'),
    "standardization": subscript('scale'),
}
