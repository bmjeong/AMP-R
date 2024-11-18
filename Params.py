from Utils import *

class Params:
    def __init__(self, Depots, Customers, N_iter=100, buff_size=10):
        self.gurobi_time = 1200 # sec

        self.Depots = Depots
        self.Customers = Customers

        self.map_size = np.array([[0, 100], [0, 100]])
        self.depot_n = len(Depots)

        self.vehicle_n = []
        self.vel = []

        cnt = 0
        for depot in Depots:
            self.vehicle_n.append(len(depot.Vehicles))
            self.vel.append([])
            for v in depot.Vehicles:
                self.vel[cnt].append(v.vel)
            cnt += 1

        self.customer_n = len(Customers)

        self.N_iter = N_iter
        self.buff_size = buff_size

        self.x_range = (0, 100)
        self.y_range = (0, 100)

        # 고객 노드 집합, num_depot 부터 num_depot + num_customer 까지
        self.C = list(range(self.depot_n, self.customer_n + self.depot_n))
        self.D = list(range(self.depot_n))
        self.U = self.D + self.C  # 모든 노드 집합, 0 ~ num_depot 은 차량의 시작점
        # 차량 집합 K 생성
        # self.K = list(range(sum(self.vehicle_n)))  # 차량 집합, 0부터 num_vehicle까지
        self.K = []
        for i, n_v in enumerate(self.vehicle_n):
            for j in range(n_v):
                self.K.append((i, j))

        # self.K_depot = [[] for i in range(self.depot_n)]
        # for k in self.K:
        #     self.K_depot[self.depot4v[k]].append(k)
        # print(self.K_depot)
        # 각 고객 노드의 무게 입력
        self.q = {i: Customers[i - self.depot_n].demand for i in self.C}

        # 차량의 용량 설정
        self.Q = {(i, j): Depots[i].Vehicles[j].cap for i, j in self.K}

        # 차량의 속도 저장
        self.Vel = {(i, j): Depots[i].Vehicles[j].vel for i, j in self.K}

        self.positions = {}
        for i in range(self.depot_n):
            self.positions[i] = Depots[i].position * 100 ## 꼭 np.array 여야 함

        for i in range(self.depot_n, self.customer_n + self.depot_n):
            self.positions[i] = Customers[i - self.depot_n].position * 100

        self.base_dist = np.sqrt((self.x_range[1] - self.x_range[0]) ** 2 + (self.y_range[1] - self.y_range[0]) ** 2)

        # 거리 정보
        self.d = {(i, j): LA.norm(self.positions[i] - self.positions[j]) for i in self.U for j in self.U if i != j}

class Results:
    def __init__(self, routes, caps, route_len, vehs, alg):
        self.alg = alg
        self.routes = routes
        self.caps = caps
        self.route_len = route_len
        self.vehs = vehs