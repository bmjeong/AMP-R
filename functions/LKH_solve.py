from Params import *

def gen_prb_str(num_customer, num_depot, d, D, C, p, num):
    problem_str = "NAME : mVRP\n"
    problem_str += "TYPE : ATSP\n"
    problem_str += "DIMENSION : " + str((num_customer + 2) * num_depot) + "\n"
    problem_str += "EDGE_WEIGHT_TYPE : EXPLICIT\n"
    problem_str += "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
    problem_str += "EDGE_WEIGHT_SECTION\n"
    dist_mat = np.zeros(((num_customer + 2) * num_depot, (num_customer + 2) * num_depot))

    M = 10000
    for i in range(num_depot):
        for j in range(num_depot):
            ## Agent / Task 간 연결 부분은 일단 다 M 으로 채우는 것
            for k in range(num_customer + 2):
                for l in range(num_customer + 2):
                    if k >= num_customer or l >= num_customer:
                        dist_mat[i + k * num_depot, j + l * num_depot] = M * M
                        dist_mat[i + l * num_depot, j + k * num_depot] = M * M

            if i == j:
                ## agent_start to Task 로 가는 edge agent_i 에서 Task_j_i 로 가는 것만 cost 가 있고 나머지는 M
                for k in range(num_customer):
                    dist_mat[i + num_customer * num_depot, j + k * num_depot] = d[D[i], C[k]] / p.Vel[j, 0] + M
                ## agent_start 에서 agent_finish 로 가능 건 본인 node 만 가능
                dist_mat[i + num_customer * num_depot, j + (num_customer + 1) * num_depot] = 0
            else:
                ## agent_finish 에서 agent_start 로 가능 건 본인 node 아닌쪽만 가능
                dist_mat[i + (num_customer + 1) * num_depot, j + num_customer * num_depot] = 0

            ## agent 가 2개 뿐일때는 diagonal 은 0 이고, 아닌쪽은 2--> 1 1-->2 로 항상 연결되므로 M으로 처리할 부분이 없음
            ## 아래는 동일 Task 간 edge 연결에 대한 것
            ## agent 가 많으면 1 --> 2, 2 --> 3 3--> 1 로 가는거만 허용 반대로 연결되는 건 불허
            if num_depot > 2:
                if i + 1 == j or j + num_depot == i + 1:
                    pass
                else:
                    for k in range(num_customer):
                        dist_mat[i + k * num_depot, j + k * num_depot] = M * M

            if i + 1 == j or i + 1 == j + num_depot:
                # 서로 다른 태스크 간 연결 시에
                for k in range(num_customer - 1):
                    for l in range(num_customer - k - 1):
                        dist_mat[i + k * num_depot, j + (l + k + 1) * num_depot] = d[C[k], C[l + k + 1]] / p.Vel[
                            j, 0] + M
                        dist_mat[i + (l + k + 1) * num_depot, j + k * num_depot] = d[C[k], C[l + k + 1]] / p.Vel[
                            j, 0] + M
                # customer 에서 depot_finish 쪽으로 연결되는 것
                for k in range(num_customer):
                    dist_mat[i + k * num_depot, j + (num_customer + 1) * num_depot] = d[j, C[k]] / p.Vel[j, 0]
                    # print(j, p.Vel[j, 0])
                    # print(i, j, k, d[D[j], C[k]],  d[D[j], C[k]]/p.Vel[j, 0])

            else:
                for k in range(num_customer - 1):
                    for l in range(num_customer - k - 1):
                        dist_mat[i + k * num_depot, j + (l + k + 1) * num_depot] = M * M
                        dist_mat[i + (l + k + 1) * num_depot, j + k * num_depot] = M * M

    for i in range((num_customer + 2) * num_depot):
        for j in range((num_customer + 2) * num_depot):
            if dist_mat[i, j] < M * M:
                problem_str += '{:.5e}'.format(dist_mat[i, j] * 100) + " "
            else:
                problem_str += str(num) + " "

        problem_str += "\n"

    return problem_str

def function_lkh(p):
    C = p.C
    D = p.D
    K = p.K
    d = p.d
    num_customer = p.customer_n
    num_depot = p.depot_n
    positions = p.positions

    customer_demand = []
    for i in range(p.customer_n):
        customer_demand.append(p.q[p.depot_n + i])

    solver_path = p.solver_path  ### 위치가 바뀌면 여기를 수정해야 함. 컴퓨터 별로 다를 수 있음

    if num_customer == 10:
        run_num = 20
        trial_num = 8000
    elif num_customer == 20:
        run_num = 30
        trial_num = 10000
    elif num_customer == 40:
        run_num = 70
        trial_num = 20000
    elif num_customer == 60:
        run_num = 120
        trial_num = 35000
    else:
        run_num = 150
        trial_num = 40000

    try:
        problem_str = gen_prb_str(num_customer, num_depot, d, D, C, p, 1.8e7)
        problem = lkh.LKHProblem.parse(problem_str)
        out = lkh.solve(solver_path, problem=problem, max_trials=trial_num, runs=run_num)
    except:
        try:
            print("2nd try")
            problem_str = gen_prb_str(num_customer, num_depot, d, D, C, p, 1.6e7)
            problem = lkh.LKHProblem.parse(problem_str)
            out = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=50)
        except:
            print("3rd try")
            problem_str = gen_prb_str(num_customer, num_depot, d, D, C, p, 1.5e7)
            problem = lkh.LKHProblem.parse(problem_str)
            out = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=50)
    # print(out)
    ind_set = []
    for i in range(num_depot):
        ind = num_customer * num_depot + i + 1
        ind_set.append([out[0].index(ind), out[0].index(ind + num_depot)])
    path = []
    for i in range(num_depot):
        if ind_set[i][0] < ind_set[i][1]:
            path.append(out[0][ind_set[i][0]:ind_set[i][1] + 1])
        else:
            tp = out[0][ind_set[i][0]:]
            for j in range(ind_set[i][1] + 1):
                tp.append(out[0][j])
            path.append(tp)

    routes = {}
    caps = {}
    vehs = {}
    route_len = np.zeros(p.depot_n)
    cnt = 0
    for p_i in path:
        path_ = []
        cap_ = []
        for t in p_i:
            if t <= (num_depot * num_customer):
                if t % num_depot == 0:
                    path_.append(int(t / num_depot) - 1)
                    cap_.append(customer_demand[int(t / num_depot) - 1])
        if path_:
            routes[cnt] = [path_]
            caps[cnt] = [cap_]
            vehs[cnt] = [(0, p.Depots[cnt].Vehicles[0].type_)]
        else:
            routes[cnt] = []
            caps[cnt] = []
            vehs[cnt] = []


        cnt += 1


    for i in range(len(routes)):
        if routes[i]:
            route_len[i] += d[i, routes[i][0][0] + num_depot] / p.Vel[i, 0]
            for j in range(len(routes[i][0]) - 1):
                route_len[i] += d[routes[i][0][j] + num_depot, routes[i][0][j+1] + num_depot] / p.Vel[i, 0]
            route_len[i] += d[i, routes[i][0][-1] + num_depot] / p.Vel[i, 0]

    result = Results(routes, caps, route_len, vehs, "LKH")
    return result