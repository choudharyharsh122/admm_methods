### To run please use this command

python admm_run_random_seeds.py --mesh-list 64 --backend mergesplit --break-iter 30 --alpha 1e-3 --rho 0.1 --gamma 1.8 --zeta 0.9 --source_strength 1 --vol_frac 0.4

### Backend can be set to gurobi, if mergesplit, then this code currently runs for a single seed (hardcoded right now in the file admm_run_random_seeds.py)
