from Utils import *
from Problem import Prob
from Plot_result import *
from functions.Greedy import *
from functions.Gurobi import *
from functions.LKH_solve import *
from functions.AMP_R_Greedy import *
from functions.AMP_R_Mix import *
from functions.AMP_R_LKH import *


if __name__ == '__main__':
    ## Select Problem
    # Prob(Problem_type, n_depot, n_customer, seed)
    # Problem_type
    # 0 : MDVRP-(u), 1 : HMDVRP-(u), 2 : MDVRP-(c), 3 : HMDVRP-(c), 4 : MDVRP-(c)-LV, 5 : HMDVRP-(c)-LV
    # n_depot : number of depot
    # n_customer : number of customer
    # n_vehicle : number of vehicle
    # seed : random instance

    Problem_type = 0
    n_depot = 2
    n_customer = 5
    n_vehicle = 10 # only for -LV cases
    seed = 0

    ## AMP_R_Greedy : With greedy algorithm to solve SV-TSP or SV-CVRP (0 ~ 5)
    ## AMP_R_LKH : With LKH algorithm to solve SV-TSP (0 ~ 1) only uncapacitated
    ## AMP_R_Mix : With LKH algorithm to solve SV-TSP (0 ~ 1) only uncapacitated
    ## Greedy : Totally greedy for HMDVRP (0 ~ 5)
    ## GRB : Integer programming with gurobi (0 ~ 5)
    ## LKH : Solving HMDVRP with LKH (0 ~ 1) only uncapacitated

    alg = "LKH"

    p = Prob(Problem_type, n_depot, n_customer, n_vehicle, seed)

    tic = time.time()
    if alg == "AMP_R_Greedy":
        results = function_AMP_R_greedy(p)
    elif alg == "AMP_R_LKH":
        results = function_AMP_R_lkh(p)
    elif alg == "AMP_R_Mix":
        results = function_AMP_R_mix(p)
    elif alg == "Greedy":
        results = function_greedy(p)
    elif alg == "GRB":
        results = function_gurobi(p)
    elif alg == "LKH":
        results = function_lkh(p)
    toc = time.time()

    plot_result(p, results, toc - tic)
    plt.show()