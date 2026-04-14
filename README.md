### To run please use this command in the respecive directory

> python admm_run_random_seeds.py --mesh-list 32,64 --backend gurobi --break-iter 30,30 --alpha 0.01,0.1,1 --rho 5,25,125 --gamma 1.8 --zeta 0.9 --source_strength 1 --vol_frac 0.4

### Backend can be set to gurobi, if mergesplit, then this code currently runs for a single seed (hardcoded right now in the file admm_run_random_seeds.py)

### The mesh list is supplied comma separated, the break-iter argument is the number of iterations we want to run a particular instance of admm for (the length of this list is same as mesh list)

### gamma (initial funnel width factor), beta, zeta are funnel parameters

### rho here is initial penalty which is set according to the alpha values, we can also start with same rho for each alpha but it is seen that this then leads to a lot of rejected steps before penalty becomes large enough for a step to be accepted. In the interest of runtime these are set like this. (For CONLIN this maybe a bad idea)

### Once the tests are run the complete run history files are saved in respective folders of alpha values:  0.1, 0.01

### To visualize results please run the cells in notebook admm_viz.ipynb and admm_colin_viz.ipynb
