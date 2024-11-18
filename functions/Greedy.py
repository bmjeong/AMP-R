from Params import *

def cal_dist(pck_task, d, id):
    dist = 0
    if len(pck_task) > 0:
        dist += d[id, int(pck_task[0])]
        if len(pck_task) > 1:
            for i in range(len(pck_task) - 1):
                dist += d[int(pck_task[i]), int(pck_task[i + 1])]
        dist += d[id, int(pck_task[-1])]
    return dist

def two_opt(route_S, d, id):
    route_S_ = []
    for route in route_S:
        if route:
            best_distance = cal_dist(route, d, id)
            # print(best_distance)
            n = len(route)
            improved = True
            while improved:
                improved = False
                route_ = copy.deepcopy(route)
                for i in range(n - 1):
                    for j in range(i + 2, n + 1):
                        new_tour = route_.copy()
                        new_tour[i:j] = np.flip(route_[i:j])  # Reverse the sub-tour
                        new_distance = cal_dist(new_tour, d, id)
                        if new_distance < best_distance:
                            route = new_tour
                            best_distance = new_distance
                            improved = True
            route_S_.append(route)
        else:
            route_S_.append([])
    return route_S_

def function_greedy(p):
    path = [[[] for j in range(p.vehicle_n[i])] for i in range(p.depot_n)]
    rem_v = copy.copy(p.K)
    rem_c = copy.copy(p.C)
    while rem_c:
        cost = {}
        for cus in rem_c:
            for dep, veh in rem_v:
                if path[dep][veh]:
                    qq = 0
                    for c in path[dep][veh]:
                        qq += p.q[c]
                    if p.q[cus] + qq < p.Q[dep, veh]:
                        p0 = path[dep][veh][0]
                        cost[(cus, dep, veh, 0)] = (p.d[dep, cus] + p.d[cus, p0] - p.d[dep, p0]) / p.Vel[dep, veh]
                        pe = path[dep][veh][-1]
                        cost[(cus, dep, veh, len(path[dep][veh]))] = ((p.d[dep, cus] + p.d[cus, pe] - p.d[dep, pe]) /
                                                                      p.Vel[dep, veh])
                        for k in range(len(path[dep][veh]) - 1):
                            pk = path[dep][veh][k]
                            pk1 = path[dep][veh][k + 1]
                            cost[(cus, dep, veh, len(path[dep][veh]))] = \
                                ((p.d[pk, cus] + p.d[cus, pk1] - p.d[pk, pk1]) / p.Vel[dep, veh])
                else:
                    cost[(cus, dep, veh, 0)] = p.d[dep, cus] / p.Vel[dep, veh] * 2
        (a, b, c, d) = min(cost, key=cost.get)
        path[b][c].insert(d, a)
        rem_c.remove(a)

    path_ = []
    for i, pp in enumerate(path):
        pp = two_opt(pp, p.d, i)
        path_.append(pp)

    routes = {}
    caps = {}
    vehs = {}
    route_len = np.zeros(p.depot_n)
    for dep, num_v in enumerate(p.vehicle_n):
        cap = []
        route = []
        veh = []
        v_cnt = 0
        for v in range(num_v):
            if path_[dep][v]:
                route.append([])
                cap.append([])
                route_len[dep] += p.d[dep, path_[dep][v][0]] / p.Vel[dep, v]
                for k, cus in enumerate(path_[dep][v]):
                    route[v_cnt].append(cus - p.depot_n)
                    cap[v_cnt].append(p.q[cus])
                    if k > 0:
                        route_len[dep] += p.d[path_[dep][v][k-1], cus] / p.Vel[dep, v]
                route_len[dep] += p.d[dep, path_[dep][v][-1]] / p.Vel[dep, v]
                veh.append((v, p.Depots[dep].Vehicles[v].type_))
                v_cnt += 1
        routes[dep] = route
        caps[dep] = cap
        vehs[dep] = veh

    result = Results(routes, caps, route_len, vehs, "Greedy")
    return result