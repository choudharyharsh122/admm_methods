### To run please use this command in the respecive directory

> python admm_run.py


### Please check the config files in admm folders for details about parameters.

### Once the tests are run the complete run history files are saved in respective folders of alpha values:  0.1, 0.01

### To visualize results please run the cells in notebook admm_viz.ipynb and admm_colin_viz.ipynb, a few examples:

*create admm object*

> admm = ADMM(alpha=0.01, dim=64, base_dir="run_data_admm_gurobi")

*all scalar valued objects are accessed via*
> admm.trial(0).series.objective

> admm.trial(0).series.infeasibility

*all vector valued objects are accessed via*
> admm.trial(0).iters.control

> admm.trial(0).iters.control_cont

*metadata accessed via*
> admm.trial(0).meta

*For deterministic solver there is only a single trial* 

:):
