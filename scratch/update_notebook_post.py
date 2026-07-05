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
    strategy_kwargs={"min_max_calls": 2}
)

state_biego = driver_biego.optimize()

# Extracting the history of adaptive nadir points (r)
r_history = driver_biego.strategy.r_history

# Visualizing the trajectory step by step
import matplotlib.pyplot as plt

y_all = state_biego.dataset.export_as_dict()["obj"]
y_init = y_all[:nt_init]
y_infills = y_all[nt_init:]

r_idx = 0
# We plot the last 4 infills of the bi-objective phase to show the tracking
for i in range(len(y_infills)):
    if i < 4: continue # skip single objective phases
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # True PF
    x1 = np.linspace(0, 1, 100)
    ax.plot(x1, 1 - np.sqrt(x1), "k-", label="Ref PF", linewidth=2)
    
    # Initial DoE & Previous Infills
    ax.plot(y_init[:, 0], y_init[:, 1], "go", label="Initial DoE", alpha=0.7)
    if i > 0:
        ax.plot(y_infills[:i, 0], y_infills[:i, 1], "bo", label="Previous Infills", alpha=0.7)
        
    # New Infill
    ax.plot(y_infills[i, 0], y_infills[i, 1], "r*", markersize=12, label="New Infill")
    
    # Adaptive Nadir
    if r_idx < len(r_history):
        r = r_history[r_idx]
        qoi_factor = state_biego.qoi_factor[0][:2]
        qoi_step = state_biego.qoi_step[0][:2]
        r_unscaled = r * qoi_factor + qoi_step
        ax.plot(r_unscaled[0], r_unscaled[1], "mX", markersize=10, label="Adaptive Nadir (r)")
        ax.axhline(r_unscaled[1], color="m", linestyle="--", alpha=0.5)
        ax.axvline(r_unscaled[0], color="m", linestyle="--", alpha=0.5)
        r_idx += 1
        
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_title(f"BiEGO Bi-objective Phase: Infill {i+1}")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()
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
