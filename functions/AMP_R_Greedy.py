from Params import *
from functions.BP_util import *

def function_AMP_R_greedy(p, flg_buf=True):
    num_depot = p.depot_n
    num_customer = p.customer_n

    log_msg = np.zeros((p.N_iter, num_depot, num_customer))

    depots = []
    for i in range(num_depot):
        depots.append(BP_depot(i, p, flg_buf))

    boards = []
    for i in range(num_customer):
        boards.append(Board(i, num_depot, num_customer))

    for idx_iter in range(p.N_iter):
        aa = list(range(num_depot))

        for id_a in aa:
            depots[id_a].msg_cal(boards)
            log_msg[idx_iter, id_a, depots[id_a].customer] = depots[id_a].msg.t2a[0, :]

        for id_a in aa:
            for id_t in depots[id_a].customer:
                id_a_4_t = np.where(id_a == np.asarray(boards[id_t].depot))[0][0]
                id_t_4_a = np.where(id_t == np.asarray(depots[id_a].customer))[0][0]
                boards[id_t].msg[0, id_a_4_t] = depots[id_a].msg.a2t[0, id_t_4_a]
                boards[id_t].msg[1, id_a_4_t] = depots[id_a].decision[0, id_t_4_a]
                boards[id_t].msg[2, id_a_4_t] = depots[id_a].decision[1, id_t_4_a]
                boards[id_t].msg[3, id_a_4_t] = depots[id_a].decision[2, id_t_4_a]
                boards[id_t].msg[4, id_a_4_t] = depots[id_a].decision[3, id_t_4_a]

    for id_a in range(num_depot):
        depots[id_a].refinement(boards)

    routes = {}
    caps = {}
    vehs = {}
    route_len = np.zeros(p.depot_n)
    for dep, num_v in enumerate(p.vehicle_n):
        cap = []
        route = []
        veh = []
        cnt = 0
        veh_cap = []
        for v in range(num_v):
            veh_cap.append(p.Q[dep, v])
        for v in range(num_v):
            if max(veh_cap) > min(veh_cap):
                route.append([])
                cap.append([])

            if depots[dep].route_S[v]:
                if max(veh_cap) == min(veh_cap):
                    route.append([])
                    cap.append([])
                veh.append((v, p.Depots[dep].Vehicles[v].type_))

                route_len[dep] += p.d[dep, depots[dep].route_S[v][0] + p.depot_n] / p.Vel[dep, v]
                for k, cus in enumerate(depots[dep].route_S[v]):
                    route[cnt].append(cus)
                    cap[cnt].append(p.q[cus + p.depot_n])
                    if k > 0:
                        route_len[dep] += p.d[depots[dep].route_S[v][k - 1] + p.depot_n, cus + p.depot_n] / p.Vel[
                            dep, v]
                route_len[dep] += p.d[dep, depots[dep].route_S[v][-1] + p.depot_n] / p.Vel[dep, v]

                if max(veh_cap) == min(veh_cap):
                    cnt += 1
            else:
                if max(veh_cap) > min(veh_cap):
                    veh.append((v, -1))

            if max(veh_cap) > min(veh_cap):
                cnt += 1

        routes[dep] = route
        caps[dep] = cap
        vehs[dep] = veh

    result = Results(routes, caps, route_len, vehs, "AMP_R_Greedy")

    return result
