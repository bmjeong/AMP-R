from Utils import *

class Message_a:
    def __init__(self, n_task):
        self.a2t = np.ones((2, n_task)) / 2
        self.t2a = np.ones((2, n_task)) / 2

class Message_buf:
    def __init__(self, n_task, buff_sz):
        self.id = 0
        self.a2t = np.ones((2, n_task, buff_sz)) / 2
        self.t2a = np.ones((2, n_task, buff_sz)) / 2

class Board:  # task 별로 board 하나
    def __init__(self, id, num_depot, num_customer):
        # 1 : msg0, 2 : 할당 여부, 3 : 할당 위치
        self.id = id
        self.depot = []
        for i in range(num_depot):
            self.depot.append(i)
        self.depot = np.asarray(self.depot)
        self.num_depot = len(self.depot)
        self.num_customer = num_customer
        self.msg = np.ones(self.num_depot) * np.array(([[0.5], [0], [0], [0], [0]]))

class BP_depot:
    def __init__(self, id, p, flg_buf=True):
        self.solver_path = p.solver_path
        self.id = id
        if flg_buf:
            self.buff_size = p.buff_size
        else:
            self.buff_size = 1
        self.customer = []
        self.customer_demand = []
        for i in range(p.customer_n):
            self.customer.append(i)
            self.customer_demand.append(p.q[p.depot_n + i])
        self.num_customer = p.customer_n
        self.num_depot = p.depot_n
        self.num_vehicle = p.Depots[id].vehicle_n

        self.vehicle_cap = []
        self.vehicle_vel = []
        for i in range(self.num_vehicle):
            self.vehicle_cap.append(p.Q[self.id, i])
            self.vehicle_vel.append(p.Vel[self.id, i])

        self.d = p.d
        self.base_dist = p.base_dist

        self.msg = Message_a(self.num_customer)
        self.buff = Message_buf(self.num_customer, p.buff_size)

        self.decision = np.zeros((4, self.num_customer))  # 첫번째에는 할당 여부, 두번째에는 할당 위치를 나타냄

        self.value_a2t = np.ones((self.num_customer, self.num_vehicle))
        self.value_t2a = np.ones((self.num_customer, self.num_vehicle))
        self.value_t2t = np.ones((self.num_customer, self.num_customer, self.num_vehicle))
        self.value_total = np.ones((1 + self.num_customer, 1 + self.num_customer, self.num_vehicle))

        for i in range(self.num_customer):
            for j in range(self.num_vehicle):
                self.value_a2t[i, j] = np.exp(-self.d[self.id, self.num_depot + i] / self.vehicle_vel[j] / self.base_dist + 0.01 * np.random.random(1) - 0.005)

                self.value_total[0, i + 1, j] = self.value_a2t[i, j]
                self.value_t2a[i, j] = np.exp(-self.d[self.id, self.num_depot + i] / self.vehicle_vel[j] / self.base_dist + 0.01 * np.random.random(1) - 0.005)
                self.value_total[i + 1, 0, j] = self.value_t2a[i, j]
                for k in range(self.num_customer):
                    if not i == k:
                        self.value_t2t[i, k, j] = np.exp(-self.d[self.num_depot + i, self.num_depot + k] / self.vehicle_vel[j] / self.base_dist + 0.01 * np.random.random(1) - 0.005)
                    else:
                        self.value_t2t[i, k, j] = 1
                    self.value_total[i + 1, k + 1, j] = self.value_t2t[i, k, j]

        self.route_S = []

        self.mem_cus_cand = []
        self.mem_route = []

    def cal_val_insert(self, route, task, l, v):
        if route:
            if l == 0:
                l1 = 0
                l2 = route[l] + 1
            elif l == len(route):
                l1 = route[l - 1] + 1
                l2 = 0
            else:
                l1 = route[l - 1] + 1
                l2 = route[l] + 1
            return self.value_total[l1, task + 1, v] * self.value_total[task + 1, l2, v] / (self.value_total[l1, l2, v] + 1e-64)
        else:
            return self.value_total[0, task + 1, v] * self.value_total[0, task + 1, v]

    def cal_val_delete(self, route, l, v):
        if route:
            if len(route) == 1:
                return 1 / (self.value_total[0, route[l] + 1, v] * self.value_total[0, route[l] + 1, v] + 1e-64)
            else:
                if l == 0:
                    l1 = 0
                    l2 = route[l + 1] + 1
                elif l == len(route) - 1:
                    l1 = route[l - 1] + 1
                    l2 = 0
                else:
                    l1 = route[l - 1] + 1
                    l2 = route[l + 1] + 1
                return self.value_total[l1, l2, v] / (
                            self.value_total[l1, route[l] + 1, v] * self.value_total[route[l] + 1, l2, v] + 1e-64)
        else:
            print("Route is empty, so it's impossible to calculate cal_val_delete")
            return None

    def cal_Possible_insert(self, route_S, rem_task):
        possible_insert = []
        for i, route in enumerate(route_S):
            if route:
                dem = 0
                for j in route:
                    dem += self.customer_demand[j]
                for k in rem_task:
                    if dem + self.customer_demand[k] <= self.vehicle_cap[i]:
                        for j in range(len(route) + 1):
                            possible_insert.append((i, j, k))
            else:
                for k in rem_task:
                    if self.customer_demand[k] <= self.vehicle_cap[i]:
                        possible_insert.append((i, 0, k))
        return possible_insert

    def cal_val_mat(self, route_S, rem_task, msg_t2a):
        possible_insert = self.cal_Possible_insert(route_S, rem_task)
        if possible_insert:
            val_mat = np.zeros(len(possible_insert))
            for i in range(len(possible_insert)):
                v, l, t = possible_insert[i]
                val_mat[i] = self.cal_val_insert(route_S[v], t, l, v) * msg_t2a[1, t] / (msg_t2a[0, t] + 1e-64)
            ind = np.argmax(val_mat)
            if val_mat[ind] > 1:
                return possible_insert[ind], val_mat[ind]
            else:
                return False, False
        else:
            return False, False

    def cal_val_mat_cr(self, route_S, rem_task, msg):
        possible_insert = self.cal_Possible_insert(route_S, rem_task)
        if possible_insert:
            val_mat = np.zeros(len(possible_insert))
            for i in range(len(possible_insert)):
                v, l, t = possible_insert[i]
                val_mat[i] = self.cal_val_insert(route_S[v], t, l, v)
            ind = np.argmax(val_mat)
            return possible_insert[ind], val_mat[ind]
        else:
            return False, False

    def cal_val_mat_nomsg(self, route_S, rem_task):
        possible_insert = self.cal_Possible_insert(route_S, rem_task)
        if possible_insert:
            val_mat = np.zeros(len(possible_insert))
            for i in range(len(possible_insert)):
                v, l, t = possible_insert[i]
                val_mat[i] = self.cal_val_insert(route_S[v], t, l, v)
            ind = np.argmax(val_mat)
            return possible_insert[ind], val_mat[ind]
        else:
            return False, False

    def remove_task(self, route, task):
        for r in route:
            if r:
                if task in r:
                    r.remove(task)
        return route

    def cal_dist(self, pck_task):
        dist = 0
        if len(pck_task) > 0:
            dist += self.d[self.id, self.num_depot + int(pck_task[0])]
            if len(pck_task) > 1:
                for i in range(len(pck_task) - 1):
                    dist += self.d[self.num_depot + int(pck_task[i]), self.num_depot + int(pck_task[i + 1])]
            dist += self.d[self.id, self.num_depot + int(pck_task[-1])]
        return dist

    ### LKH 로 마지막에 Conflict Resolution 후에 사용
    def two_opt_(self, route_S):
        alloc = []
        for rr in route_S:
            for cus2 in rr:
                alloc.append(cus2)
        if len(alloc) > 2:
            problem_str = "NAME : mVRP\n"
            problem_str += "TYPE : CVRP\n"
            problem_str += "DIMENSION : " + str(len(alloc) + 1) + "\n"
            problem_str += "CAPACITY : " + str(min(self.vehicle_cap)) + "\n"
            problem_str += "EDGE_WEIGHT_TYPE : EXPLICIT\n"
            problem_str += "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"

            dist_mat = np.zeros((len(alloc) + 1, len(alloc) + 1))
            for i in range(len(alloc)):
                for j in range(i + 1, len(alloc) + 1):
                    if i == 0:
                        dd = self.d[self.id, self.num_depot + alloc[j - 1]]
                    else:
                        dd = self.d[self.num_depot + alloc[i - 1], self.num_depot + alloc[j - 1]]
                    dist_mat[i, j] = dd
                    dist_mat[j, i] = dd

            problem_str += "EDGE_WEIGHT_SECTION\n"
            for i in range(len(alloc) + 1):
                for j in range(len(alloc) + 1):
                    problem_str += str(int(dist_mat[i, j] * 1000)) + " "
                problem_str += "\n"
            problem_str += "DEMAND_SECTION\n"
            for i in range(len(alloc) + 1):
                if i == 0:
                    dem = 0
                else:
                    dem = self.customer_demand[alloc[i - 1]]
                problem_str += str(i + 1) + " " + str(dem) + "\n"
            problem_str += "DEPOT_SECTION\n"
            problem_str += "1\n"
            problem_str += "-1\n"

            solver_path = self.solver_path
            problem = lkh.LKHProblem.parse(problem_str)
            # print(problem)

            try:
                if self.num_customer == 60:
                    out = lkh.solve(solver_path, problem=problem, max_trials=1500, runs=10)
                elif self.num_customer == 80:
                    out = lkh.solve(solver_path, problem=problem, max_trials=3000, runs=20)
                else:
                    out = lkh.solve(solver_path, problem=problem, max_trials=1000, runs=5)
                route_S_ = []
                for i, path in enumerate(out):
                    tp_route = []
                    for cus in path:
                        if cus > 1:
                            tp_route.append(alloc[cus - 2])
                    route_S_.append(tp_route)

                for i in range(len(route_S_), self.num_vehicle + 1):
                    route_S_.append([])
                return route_S_
            except:
                # print("except")
                return route_S
        else:
            return route_S

    def insert_depot(self, alloc):
        route_S = [[]]
        cap = 0
        cnt = 0
        for task in alloc:
            if cap + self.customer_demand[int(task)] <= self.vehicle_cap[cnt]:
                route_S[cnt].append(task)
                cap += self.customer_demand[int(task)]
            else:
                cnt += 1
                route_S.append([])
                if cnt == self.num_vehicle:
                    return False
                route_S[cnt].append(task)
                cap = self.customer_demand[int(task)]

        while True:
            if len(route_S) < self.num_vehicle:
                route_S.append([])
            else:
                break

        return route_S

    def gen_init_path(self, msg_t2a):
        # print("AA")
        route_S = [[] for i in range(self.num_vehicle)]
        rem_task = list(range(self.num_customer))
        alloc = []
        while True:

            loc_task, val = self.cal_val_mat(route_S, rem_task, msg_t2a)
            if loc_task == False:
                break
            else:
                route_S[loc_task[0]].insert(loc_task[1], loc_task[2])
                rem_task.remove(loc_task[2])
                alloc.append(loc_task[2])
                if not rem_task:
                    break

        return route_S, rem_task, alloc

    def gen_init_path_lkh(self, msg_t2a):
        cus_cand = []
        for i in range(self.num_customer):
            if msg_t2a[1, i] / (msg_t2a[0, i] + 1e-64) > 1:
                cus_cand.append(i)
        if True:
            M = 100
            MM = 10000
            dist_mat = np.zeros((len(cus_cand) * 2 + 1, len(cus_cand) * 2 + 1))

            for i in range(len(cus_cand)):
                dist_mat[0, 2 * (i + 1) - 1] = - np.log(self.value_total[0, cus_cand[i] + 1, 0]) - np.log(
                    msg_t2a[1, cus_cand[i]]) + M
                dist_mat[0, 2 * (i + 1)] = - np.log(msg_t2a[0, cus_cand[i]]) + M
                dist_mat[2 * (i + 1) - 1, 0] = 0
                dist_mat[2 * (i + 1), 0] = - np.log(self.value_total[0, cus_cand[i] + 1, 0])

            for i in range(len(cus_cand)):
                for j in range(len(cus_cand)):
                    if not i == j:
                        dist_mat[2 * (i + 1) - 1, 2 * (j + 1) - 1] = MM
                        dist_mat[2 * (i + 1) - 1, 2 * (j + 1)] = - np.log(msg_t2a[0, cus_cand[j]]) + M
                        dist_mat[2 * (i + 1), 2 * (j + 1) - 1] = (- np.log(self.value_total[cus_cand[i] + 1, cus_cand[j] + 1, 0])
                                                                  - np.log(msg_t2a[1, cus_cand[j]]) + M)
                        dist_mat[2 * (i + 1), 2 * (j + 1)] = (- np.log(self.value_total[cus_cand[i] + 1, 0, 0])
                                                              - np.log(msg_t2a[0, cus_cand[j]]) + M)

            problem_str = "NAME : mVRP\n"
            problem_str += "TYPE : ATSP\n"
            problem_str += "DIMENSION : " + str(len(cus_cand) * 2 + 1) + "\n"
            problem_str += "EDGE_WEIGHT_TYPE : EXPLICIT\n"
            problem_str += "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
            problem_str += "EDGE_WEIGHT_SECTION\n"
            for i in range(len(cus_cand) * 2 + 1):
                for j in range(len(cus_cand) * 2 + 1):
                    problem_str += str(int(dist_mat[i, j] * 1000)) + " "
                problem_str += "\n"
            problem_str += "EOF"

            solver_path = self.solver_path
            problem = lkh.LKHProblem.parse(problem_str)

            if self.num_customer == 10:
                num_trial = 30
                num_run = 2
            elif self.num_customer == 20:
                num_trial = 50
                num_run = 2
            elif self.num_customer == 40:
                num_trial = 200
                num_run = 4
            elif self.num_customer == 60:
                num_trial = 300
                num_run = 6
            elif self.num_customer == 80:
                num_trial = 500
                num_run = 8
            else:
                num_trial = 100
                num_run = 10

            out = lkh.solve(solver_path, problem=problem, max_trials=num_trial, runs=num_run)
            route_S = [[] for i in range(self.num_vehicle)]
            rem_task = list(range(self.num_customer))
            alloc = []

            for i, path in enumerate(out):
                for j in range(int((len(path) - 1) / 2)):
                    if path[2 * j + 1] < path[2 * (j + 1)]:
                        cus = cus_cand[int(path[2 * j + 1] / 2) - 1]
                        route_S[i].append(cus)
                        rem_task.remove(cus)
                        alloc.append(cus)

            del_cus = []
            for v, route in enumerate(route_S):
                for l, cus in enumerate(route):
                    if msg_t2a[1, cus] / (msg_t2a[0, cus] * self.cal_val_delete(route, l, v) + 1e-64) < 1:
                        del_cus.append((v, cus))
            for v, cus in del_cus:
                route_S[v].remove(cus)
                rem_task.append(cus)
                alloc.remove(cus)
        return route_S, rem_task, alloc

    def msg_cal(self, boards, lkh_=False, new_=False):
        self.decision = np.zeros((4, self.num_customer))
        msg_t2a = copy.copy(self.msg.t2a)
        ## Find Si
        cus_cand = []
        for i in range(self.num_customer):
            if msg_t2a[1, i] > msg_t2a[0, i]:
                cus_cand.append(i)
        if len(cus_cand) > 2 and lkh_ == True:
            route_S, rem_task, alloc = self.gen_init_path_lkh(msg_t2a)
        else:
            route_S, rem_task, alloc = self.gen_init_path(msg_t2a)

        self.route_S = copy.copy(route_S)

        msg_0 = np.ones(self.num_customer)
        msg_1 = np.ones(self.num_customer)
        for id_t4a in range(self.num_customer):
            ## Cal msgc2d
            if id_t4a in alloc:
                # msg_0 = exp(w_R0) * msg1(Add) / msg0(Add)
                # msg_1 = exp(w_P)
                # R_0 = S_i - c_j + Add, R_1 = S_i - c_j, P = S_i 이다.
                # 따라서 일단 w_R0, w_P 에서 w_R_1를 빼주면, 각각
                # msg_0 = val_insert(R_1, add) * msg1(Add) / msg0(Add)
                # msg_1 = val_insert(R_1, c_j)
                # 그런데, val_insert(R_1, c_j) = 1 / val_delete(S_i, c_j) 이므로
                # msg_0 = val_delete(S_i, c_j) * val_insert(R_1, add) * msg1(Add) / msg0(Add)
                # msg_1 = 1
                route_S_c = copy.deepcopy(route_S)
                rem_task_c = copy.copy(rem_task)
                alloc_c = copy.copy(alloc)
                route_S_c = self.remove_task(route_S_c, id_t4a)

                loc_task, val = self.cal_val_mat_nomsg(route_S_c, [id_t4a])
                msg_1[id_t4a] *= val

                while True:
                    loc_task, val = self.cal_val_mat(route_S_c, rem_task_c, msg_t2a)
                    if loc_task == False:
                        break
                    else:
                        route_S_c[loc_task[0]].insert(loc_task[1], loc_task[2])
                        msg_0[id_t4a] *= val
                        rem_task_c.remove(loc_task[2])
                        alloc_c.append(loc_task[2])
                        if not rem_task:
                            break

            else:
                # msg_0 = exp(w_R0) * msg1(Del) / msg0(Del)
                # msg_1 = exp(w_P)
                # R_0 = S_i, R_1 = S_i - Del, P = S_i - Del + c_j 이다.
                # 따라서 일단 w_R0, w_P 에서 w_R_1를 빼주면, 각각
                # msg_0 = val_insert(S_i - Del, Del) * msg1(Add) / msg0(Add)
                # msg_1 = val_insert(S_i - Del, c_j)
                # 그런데, val_insert(S_i - Del, Del) = 1 / val_delete(S_i, Del) 이므로
                # msg_0 = msg1(Del) / msg0(Del)
                # msg_1 = val_delete(S_i, Del) * val_insert(S_i - Del, c_j)
                route_S_c = copy.deepcopy(route_S)
                rem_task_c = copy.copy(rem_task)
                alloc_c = copy.copy(alloc)

                possible_insert = self.cal_Possible_insert(route_S, [id_t4a])
                if possible_insert:
                    loc_task, val = self.cal_val_mat_nomsg(route_S, [id_t4a])
                    msg_1[id_t4a] *= val
                else:
                    veh_cap = copy.deepcopy(self.vehicle_cap)
                    for v, route in enumerate(route_S_c):
                        if route:
                            for i in range(len(route)):
                                veh_cap[v] -= self.customer_demand[route[i]]

                    val_mat = []
                    ind_mat = []
                    for v, route_tp in enumerate(route_S_c):
                        for l, cus_tp in enumerate(route_tp):
                            if veh_cap[v] + self.customer_demand[cus_tp] > self.customer_demand[id_t4a]:
                                val_mat.append(self.cal_val_delete(route_tp, l, v))
                                ind_mat.append((v, l))
                    if val_mat:
                        ind = np.argmax(val_mat)
                        vv, ll = ind_mat[ind]
                        r0 = route_S_c[vv][ll]
                        rem_task_c.append(r0)
                        alloc_c.remove(r0)
                        val_del = np.max(val_mat)
                        msg_0[id_t4a] *= msg_t2a[1, r0] / msg_t2a[0, r0]
                        msg_1[id_t4a] *= val_del
                        route_S_c = self.remove_task(route_S_c, route_S_c[vv][ll])

                        loc_task, val = self.cal_val_mat_nomsg(route_S_c, [id_t4a])
                        msg_1[id_t4a] *= val
                    else:
                        val_mat = []
                        ind_mat = []
                        for v, route_tp in enumerate(route_S_c):
                            for l1 in range(len(route_tp) - 1):
                                for l2 in range(l1 + 1, len(route_tp)):
                                    if veh_cap[v] + self.customer_demand[route_tp[l1]] + self.customer_demand[
                                        route_tp[l2]] > self.customer_demand[id_t4a]:
                                        route_tp_tp = copy.deepcopy(route_tp)
                                        val_del1 = self.cal_val_delete(route_tp_tp, l1, v)
                                        route_tp_tp.remove(route_tp[l1])
                                        val_del2 = self.cal_val_delete(route_tp_tp, route_tp_tp.index(route_tp[l2]), v)
                                        val_mat.append(val_del1 * val_del2)
                                        ind_mat.append((v, l1, l2))
                        ind = np.argmax(val_mat)
                        vv, ll1, ll2 = ind_mat[ind]
                        r1 = route_S_c[vv][ll1]
                        r2 = route_S_c[vv][ll2]
                        rem_task_c.append(r1)
                        rem_task_c.append(r2)
                        alloc_c.remove(r1)
                        alloc_c.remove(r2)
                        val_del1 = np.max(val_mat)
                        msg_0[id_t4a] *= msg_t2a[1, r1] / msg_t2a[0, r1]
                        msg_0[id_t4a] *= msg_t2a[1, r2] / msg_t2a[0, r2]
                        msg_1[id_t4a] *= val_del1
                        r1 = route_S_c[vv][ll1]
                        r2 = route_S_c[vv][ll2]
                        route_S_c = self.remove_task(route_S_c, r1)
                        route_S_c = self.remove_task(route_S_c, r2)

                        loc_task, val = self.cal_val_mat_nomsg(route_S_c, [id_t4a])
                        msg_1[id_t4a] *= val

        msg = []
        msg.append(msg_0)
        msg.append(msg_1)
        msg = np.asarray(msg)

        ### Damping 효과
        # 여기까지 a2t 메시지 계산한 거
        self.buff.a2t[:, :, self.buff.id] = msg / (msg_0 + msg_1 + 1e-64)
        self.msg.a2t = np.mean(self.buff.a2t, 2)

        for id_t_4_a, id_t in enumerate(self.customer):
            if boards[id_t].num_depot == 1:
                self.buff.t2a[:, id_t_4_a, self.buff.id] = [1e-64, 1]  # agent 가 하나라면 무조건 xij = 1
            else:
                msg_a2t_0 = []
                for i in range(self.num_depot):
                    if not i == self.id:
                        msg_a2t_0.append((1 - boards[id_t].msg[0, i]) / boards[id_t].msg[0, i])
                msg_0 = np.asarray(msg_a2t_0)
                mmsg_0 = max(msg_0)
                tp = np.asarray([mmsg_0, 1])
                self.buff.t2a[:, id_t_4_a, self.buff.id] = tp / (mmsg_0 + 1 + 1e-64)

        self.msg.t2a = np.mean(self.buff.t2a, 2)
        self.buff.id += 1
        if self.buff.id > self.buff_size - 1:
            self.buff.id = 0

        rem_cap = []
        for j, route in enumerate(route_S):
            dem = 0
            for task in route:
                dem += self.customer_demand[task]
            rem_cap.append(self.vehicle_cap[j] - dem)
            for i, task in enumerate(route):
                self.decision[0, task] = 1
                self.decision[1, task] = i + 1

        for i in range(self.num_customer):
            self.decision[2, i] = sum(rem_cap)
            if self.decision[0, i] == 0:
                possible_insert = self.cal_Possible_insert(self.route_S, [i])
                if possible_insert:
                    val_mat = np.zeros(len(possible_insert))
                    for k in range(len(possible_insert)):
                        v, l, t = possible_insert[k]
                        val_mat[k] = self.cal_val_insert(self.route_S[v], t, l, v)
                    self.decision[3, i] = min(val_mat)
                else:
                    self.decision[3, i] = 1
            else:
                self.decision[3, i] = 1

    def refinement(self, boards):
        msg_mat = np.zeros((self.num_depot, self.num_customer))
        msg_mat_c = np.zeros((self.num_depot, self.num_customer))
        dec_mat = np.zeros((self.num_depot, self.num_customer))
        rem_cap = np.zeros(self.num_depot)
        for i in range(boards[0].num_depot):
            rem_cap[i] = boards[0].msg[3, i]

        # board_msg 0 : msg0 value, 1 : 할당 여부, 2: 할당 위치, 3 : remained cap, 4 : val_insert
        for j in range(boards[0].num_customer):
            if sum(boards[j].msg[1, :]) > 1:
                ind = np.argmin(boards[j].msg[0, :] * (1 - boards[j].msg[1, :]))
                for k in range(self.num_depot):
                    if k == ind:
                        dec_mat[ind, j] = 1
                    else:
                        if boards[j].msg[1, k] == 1:
                            rem_cap[k] += self.customer_demand[j]
            elif sum(boards[j].msg[1, :]) == 1:
                dec_mat[boards[j].msg[1, :] == 1, j] = 1

        for i in range(self.num_depot):
            for j in range(self.num_customer):
                if sum(dec_mat[:, j]) == 0:
                    msg_mat[i, j] = (boards[j].msg[0, i])
                    # * (1 - boards[j].msg[4, i]))
                else:
                    msg_mat[i, j] = 1
                msg_mat_c[i, j] = boards[j].msg[0, i]

        while True:
            ind = np.unravel_index(np.argmin(msg_mat, axis=None), msg_mat.shape)
            if msg_mat[ind[0], ind[1]] < 1:
                if rem_cap[ind[0]] >= self.customer_demand[ind[1]] * self.num_depot:
                    dec_mat[ind[0], ind[1]] = 1
                    rem_cap[ind[0]] -= self.customer_demand[ind[1]]
                    msg_mat[:, ind[1]] = 1
                else:
                    msg_mat[ind[0], ind[1]] = 1
            else:
                break

        route_S_c = copy.copy(self.route_S)
        alloc = []
        for i in range(len(route_S_c)):
            for cus in route_S_c[i]:
                alloc.append(cus)

        rem_task = []
        msg = []
        for i in range(self.num_customer):
            if dec_mat[self.id, i] == 1 and not i in alloc:
                rem_task.append(i)
            elif dec_mat[self.id, i] == 0 and i in alloc:
                for j in range(len(route_S_c)):
                    if i in route_S_c[j]:
                        route_S_c[j].remove(i)
                        break

            msg.append(msg_mat_c[self.id, i])

        while True:
            loc_task, val = self.cal_val_mat_cr(route_S_c, rem_task, msg)
            if loc_task == False:
                break
            else:
                route_S_c[loc_task[0]].insert(loc_task[1], loc_task[2])
                rem_task.remove(loc_task[2])
                if not rem_task:
                    break

        rem_cap_ = np.zeros(self.num_vehicle)
        for i in range(self.num_vehicle):
            rem_cap_[i] = self.vehicle_cap[i]
        for i in range(len(route_S_c)):
            for cus in route_S_c[i]:
                rem_cap_[i] -= self.customer_demand[cus]

        if rem_task:
            rem_cap_ = np.zeros(self.num_vehicle)
            for i in range(self.num_vehicle):
                rem_cap_[i] = self.vehicle_cap[i]

            for i in range(len(route_S_c)):
                for cus in route_S_c[i]:
                    rem_cap_[i] -= self.customer_demand[cus]
            for task in rem_task:
                ind_ = np.argmax(rem_cap_)
                route_S_c[ind_].append(task)

        route_S_ = self.two_opt_(route_S_c)
        self.route_S = copy.deepcopy(route_S_)

