from Params import *

def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def function_gurobi(p):
    C = p.C
    D = p.D
    U = p.U
    K = p.K
    d = p.d
    q = p.q
    Q = p.Q
    num_depot = p.depot_n
    positions = p.positions

    mdl = gp.Model("VRP")
    Pair = [(i, j) for i in U for j in U if i != j]
    pairs = [(i, j, k, l) for i in U for j in U for k, l in K]

    x = mdl.addVars(pairs, vtype=GRB.BINARY, name="x")
    mdl.modelSense = GRB.MINIMIZE
    u = mdl.addVars(C, vtype=GRB.CONTINUOUS, name="u")

    # Objective Function
    mdl.setObjective(quicksum(d[i, j] / p.Vel[k, l] * x[i, j, k, l] for i, j in Pair for k, l in K), GRB.MINIMIZE)

    # Constraint
    mdl.addConstr(quicksum(x[i, i, k, l] for i in U for k, l in K) == 0)
    mdl.addConstrs((quicksum(x[i, j, k, l] for i in U) - quicksum(x[j, i, k, l] for i in U)) == 0 for j in U for k, l in K)
    mdl.addConstrs(quicksum(x[i, j, k, l] for k, l in K for i in U if i != j) == 1 for j in C)
    mdl.addConstrs(quicksum(x[k, j, k, l] for j in C) <= 1 for k, l in K)
    mdl.addConstrs(quicksum(x[i, j, k, l] for j in C for i in D if not i == k) == 0 for k, l in K)
    mdl.addConstrs(quicksum(x[i, j, k, l] * q[j] for i in U for j in C if j != i) <= Q[k, l] for k, l in K)
    mdl.addConstrs((u[i] - u[j] + len(C) * x[i, j, k, l] <= len(C) - 1) for i in C for j in C if i != j for k, l in K)

    # Solve
    mdl.Params.MIPGap = 0.00001  # Optimal 과 tolance
    mdl.Params.TimeLimit = p.gurobi_time  # seconds
    mdl.Params.LogToConsole = 0
    mdl.optimize()

    # 각 차량의 경로를 저장할 딕셔너리
    routes = {}
    caps = {}
    vehs = {}
    route_len = np.zeros(p.depot_n)
    for dep, num_v in enumerate(p.vehicle_n):
        cap = [[]]
        route = [[]]
        veh = []
        cnt = 0
        for v in range(num_v):
            tp = [(i, j) for i, j, k, l in x.keys() if k == dep and v == l and x[i, j, k, l].X > 0.5]
            if tp:
                tp2 = [i for i, j in tp]
                ind = tp[0][0]
                for j in range(len(tp2)):
                    route_len[dep] += euclidean_distance(positions[tp[j][0]], positions[tp[j][1]]) / p.Vel[dep, v]
                    if ind in tp2:
                        ind = tp[tp2.index(ind)][1]
                        if ind < num_depot:
                            route.append([])
                            cap.append([])
                            cnt += 1
                        else:
                            route[cnt].append(ind - num_depot)
                            cap[cnt].append(p.q[ind])
                veh.append((v, p.Depots[dep].Vehicles[v].type_))
        if not route[-1]:
            route.remove(route[-1])
            cap.remove(cap[-1])
        routes[dep] = route
        caps[dep] = cap
        vehs[dep] = veh

    result = Results(routes, caps, route_len, vehs, "GRB")
    return result