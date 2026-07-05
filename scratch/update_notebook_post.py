import json

notebook_path = "docs/source/getting_started/multi_objective_optim.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the BiEGO adaptive nadir points code cell
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "from smt_optim.acquisition_strategies.biego import BiEGO" in "".join(cell["source"]):
        
        # Replace the cell content with the step-by-step plotting code
        new_source = '''from smt_optim.acquisition_strategies.biego import BiEGO

nt_init = 5

opt_config_biego = DriverConfig(
    max_iter=15, 
    nt_init=nt_init, 
    seed=42
)

# min_max_calls is reduced to quickly enter the bi-objective phase
driver_biego = Driver(
    problem=prob_definition, 
    config=opt_config_biego, 
    strategy=BiEGO,
    strategy_kwargs={"min_max_calls": 2, "n_multi_start": 50},
)

state_biego = driver_biego.optimize()

# Extracting the history of adaptive nadir points (r)
r_history = driver_biego.strategy.r_history

# Visualizing the trajectory with an interactive animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from smt_optim.utils.multi_obj import get_pareto_mask

y_all = state_biego.dataset.export_as_dict()["obj"]
y_init = y_all[:nt_init]
y_infills = y_all[nt_init:]

fig, ax = plt.subplots(figsize=(6, 4))

x1 = np.linspace(0, 1, 100)
line_true_pf, = ax.plot(x1, 1 - np.sqrt(x1), "k-", label="Ref PF", linewidth=2)
scatter_pareto, = ax.plot([], [], "o", color="darkorange", label="Pareto Optimal", alpha=0.9)
scatter_dom, = ax.plot([], [], "bo", label="Dominated", alpha=0.4)
scatter_new, = ax.plot([], [], "r*", markersize=12, label="New Infill")

nadir_marker, = ax.plot([], [], "mX", markersize=10, label="Adaptive Nadir (r)")
nadir_hline = ax.axhline(0, color="m", linestyle="--", alpha=0.5)
nadir_vline = ax.axvline(0, color="m", linestyle="--", alpha=0.5)

ax.set_xlabel("$f_1$")
ax.set_ylabel("$f_2$")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right")

def init():
    scatter_pareto.set_data([], [])
    scatter_dom.set_data([], [])
    scatter_new.set_data([], [])
    nadir_marker.set_data([], [])
    nadir_hline.set_visible(False)
    nadir_vline.set_visible(False)
    return scatter_pareto, scatter_dom, scatter_new, nadir_marker, nadir_hline, nadir_vline

def update(frame):
    # Only show bi-objective phase (skip first 4 if min_max_calls=2)
    i = frame + 4
    if i >= len(y_infills):
        return
        
    all_past_pts = np.vstack((y_init, y_infills[:i]))
    if len(all_past_pts) > 0:
        p_mask = get_pareto_mask(all_past_pts)
        pareto_pts = all_past_pts[p_mask]
        dom_pts = all_past_pts[~p_mask]
        
        scatter_pareto.set_data(pareto_pts[:, 0], pareto_pts[:, 1])
        scatter_dom.set_data(dom_pts[:, 0], dom_pts[:, 1])
        
    scatter_new.set_data([y_infills[i, 0]], [y_infills[i, 1]])
    
    if i < len(r_history) and r_history[i] is not None:
        r_unscaled = r_history[i]
        nadir_marker.set_data([r_unscaled[0]], [r_unscaled[1]])
        nadir_hline.set_ydata([r_unscaled[1], r_unscaled[1]])
        nadir_vline.set_xdata([r_unscaled[0], r_unscaled[0]])
        nadir_hline.set_visible(True)
        nadir_vline.set_visible(True)
        
    ax.set_title(f"BiEGO Bi-objective Phase: Infill {i+1}")
    ax.relim()
    ax.autoscale_view()
    return scatter_pareto, scatter_dom, scatter_new, nadir_marker, nadir_hline, nadir_vline

# Frames equal to number of bi-objective infills
n_frames = max(0, len(y_infills) - 4)
ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, repeat=False)
plt.close(fig) # Prevent static plot display

HTML(ani.to_jshtml())
'''
        cell["source"] = [line + "\n" for line in new_source.split("\n")]
        # Also clear outputs so it generates them freshly if run
        cell["outputs"] = []

# Check if profiling section is already added
has_profiling = any("## Advanced Post-Processing and Data Profiles" in "".join(c.get("source", [])) for c in nb["cells"])

if not has_profiling:
    # Add the profiling section at the end
    profiling_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Advanced Post-Processing and Data Profiles\n",
            "\n",
            "The SMT-Optim framework includes advanced post-processing tools inspired by the work of Oihan Cordelier and Antoine Maugras. These tools allow you to plot **Data Profiles**, **Performance Profiles**, **Convergence Profiles**, and **Accuracy Profiles** over several analytical instances.\n",
            "\n",
            "To utilize these tools, you can import them from `smt_optim.utils.profiles`. Here is a quick example of how you might aggregate multiple runs across different frameworks (like `MOSEGO` and `BiEGO`) to generate these profiles:"
        ]
    }

    profiling_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from smt_optim.utils.profiles import profile, convergence_profile\n",
            "\n",
            "# Note: The profiling functions generally expect dictionary data structures \n",
            "# aggregating multiple instances over several algorithms. \n",
            "# \n",
            "# data = {\n",
            "#     'MOSEGO': { 'ZDT1': np.array([...]), 'ZDT2': np.array([...]) },\n",
            "#     'BiEGO':  { 'ZDT1': np.array([...]), 'ZDT2': np.array([...]) }\n",
            "# }\n",
            "#\n",
            "# perf_profile = profile(data, tau=0.1, type='perf')\n",
            "# data_profile = profile(data, tau=0.1, type='data', dim={'ZDT1': 2, 'ZDT2': 2})\n",
            "#\n",
            "# These profiles can then be plotted using matplotlib to visually compare \n",
            "# the robustness and efficiency of the Bayesian optimization frameworks."
        ]
    }

    nb["cells"].append(profiling_markdown)
    nb["cells"].append(profiling_code)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
