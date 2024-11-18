from Utils import *

def plot_result(p, results, cal_time):

    routes = results.routes
    vehs = results.vehs
    route_len = results.route_len
    alg = results.alg

    col = ['red', 'blue', 'black']
    plt.figure()
    for i, pos in p.positions.items():
        if i < p.depot_n:
            plt.plot(pos[0], pos[1], 'ko', markersize=10)
            plt.text(pos[0], pos[1], f'{i}', fontsize=12)
        else:
            plt.plot(pos[0], pos[1], 'bo', markersize=10)
            plt.text(pos[0], pos[1], str(i - p.depot_n) + ", cap : " + str(p.q[i]), fontsize=12)

    col_type = []
    sav_leg = ["Car", "Bike", "Truck"]
    for k in range(p.depot_n):
        if routes[k]:
            for i in range(len(vehs[k])):
                # print(i, routes[k][i])
                if routes[k][i]:
                    # print(i, routes[k][i])
                    edge = []
                    edge.append((k, routes[k][i][0] + p.depot_n))
                    for j in range(len(routes[k][i]) - 1):
                        edge.append((routes[k][i][j] + p.depot_n, routes[k][i][j + 1] + p.depot_n))
                    edge.append((k, routes[k][i][-1] + p.depot_n))

                    for n, l in edge:
                        if not p.Depots[k].Vehicles[vehs[k][i][0]].type_ in col_type:
                            plt.arrow(p.positions[n][0], p.positions[n][1],
                                      p.positions[l][0] - p.positions[n][0], p.positions[l][1] - p.positions[n][1],
                                      head_width=0.2, length_includes_head=True,
                                      color=col[p.Depots[k].Vehicles[vehs[k][i][0]].type_],
                                      label = sav_leg[p.Depots[k].Vehicles[vehs[k][i][0]].type_])
                            col_type.append(p.Depots[k].Vehicles[vehs[k][i][0]].type_)
                        else:
                            plt.arrow(p.positions[n][0], p.positions[n][1],
                                      p.positions[l][0] - p.positions[n][0], p.positions[l][1] - p.positions[n][1],
                                      head_width=0.2, length_includes_head=True,
                                      color=col[p.Depots[k].Vehicles[vehs[k][i][0]].type_])

    # leg_type = []
    # for c in col_type:
    #     leg_type.append(sav_leg[c])
    # plt.legend(leg_type)
    plt.legend()

    # All_allocated, No_duplicated, No_over_capacity = chk_solution(p, routes, caps)

    str_xlabel = 'X coordinate\n'
    str_xlabel += 'cal_time : ' + str(cal_time) + 'sec, '
    str_xlabel += 'route_cost : ' + str(sum(route_len)) + ' m'
    plt.xlabel(str_xlabel)

    plt.ylabel('Y coordinate')
    plt.xlim([p.x_range[0] - 10, p.x_range[1] + 10])
    plt.ylim([p.y_range[0] - 10, p.y_range[1] + 10])

    str_title = 'VRP Solution, ' + alg
    # if All_allocated and No_duplicated and No_over_capacity:
    #     pass
    #
    # if not All_allocated:
    #     str_title += ', Not all allocated'
    # if not No_duplicated:
    #     str_title += ', duplicated'
    # if not No_over_capacity:
    #     str_title += ', over_capacity'

    plt.title(str_title)
    plt.grid(True)

# def draw_solution_no_p(p, routes, caps, route_len, vehs, cal_time, alg):
#     col = ['red', 'blue', 'black']
#     plt.figure()
#     for i, pos in p.positions.items():
#         if i < p.depot_n:
#             plt.plot(pos[0], pos[1], 'ko', markersize=10)
#             plt.text(pos[0], pos[1], f'{i}', fontsize=12)
#         else:
#             plt.plot(pos[0], pos[1], 'bo', markersize=10)
#             plt.text(pos[0], pos[1], str(i - p.depot_n) + ", cap : " + str(p.q[i]), fontsize=12)
#
#     col_type = []
#     sav_leg = ["Car", "Bike", "Truck"]
#     for k in range(p.depot_n):
#         if routes[k]:
#             for i in range(len(vehs[k])):
#                 edge = []
#                 edge.append((k, routes[k][i][0] + p.depot_n))
#                 for j in range(len(routes[k][i]) - 1):
#                     edge.append((routes[k][i][j] + p.depot_n, routes[k][i][j + 1] + p.depot_n))
#                 edge.append((k, routes[k][i][-1] + p.depot_n))
#
#                 for n, l in edge:
#                     if not p.Depots[k].Vehicles[vehs[k][i][0]].type_ in col_type:
#                         plt.arrow(p.positions[n][0], p.positions[n][1],
#                                   p.positions[l][0] - p.positions[n][0], p.positions[l][1] - p.positions[n][1],
#                                   head_width=0.2, length_includes_head=True,
#                                   color=col[p.Depots[k].Vehicles[vehs[k][i][0]].type_],
#                                   label = sav_leg[p.Depots[k].Vehicles[vehs[k][i][0]].type_])
#                         col_type.append(p.Depots[k].Vehicles[vehs[k][i][0]].type_)
#                     else:
#                         plt.arrow(p.positions[n][0], p.positions[n][1],
#                                   p.positions[l][0] - p.positions[n][0], p.positions[l][1] - p.positions[n][1],
#                                   head_width=0.2, length_includes_head=True,
#                                   color=col[p.Depots[k].Vehicles[vehs[k][i][0]].type_])
#
#     # leg_type = []
#     # for c in col_type:
#     #     leg_type.append(sav_leg[c])
#     # plt.legend(leg_type)
#     plt.legend()
#
#     # All_allocated, No_duplicated, No_over_capacity = chk_solution(p, routes, caps)
#
#     str_xlabel = 'X coordinate\n'
#     str_xlabel += 'cal_time : ' + str(cal_time) + 'sec, '
#     str_xlabel += 'route_cost : ' + str(sum(route_len)) + ' m'
#     plt.xlabel(str_xlabel)
#
#     plt.ylabel('Y coordinate')
#     plt.xlim([p.x_range[0] - 10, p.x_range[1] + 10])
#     plt.ylim([p.y_range[0] - 10, p.y_range[1] + 10])
#
#     str_title = 'VRP Solution, ' + alg
#     # if All_allocated and No_duplicated and No_over_capacity:
#     #     pass
#     #
#     # if not All_allocated:
#     #     str_title += ', Not all allocated'
#     # if not No_duplicated:
#     #     str_title += ', duplicated'
#     # if not No_over_capacity:
#     #     str_title += ', over_capacity'
#
#     plt.title(str_title)
#     plt.grid(True)
#
# def chk_solution(p, routes, caps):
#     All_allocated = False  # 모든 customer 가 할당되면 true
#     No_duplicated = True  # customer 가 여러 vehicle 에 선택되지 않으면 true
#     No_over_capacity = True  # cap 을 초과하지 않으면 true
#
#     cnt = 0
#     for i in range(p.customer_n):
#         flg = False
#         for j in range(p.depot_n):
#             if routes[j]:
#                 if i in routes[j]:
#                     if flg:
#                         No_duplicated = False
#                     flg = True
#         if flg:
#             cnt += 1
#
#     if cnt == p.customer_n:
#         All_allocated = True
#
#     cap = {}
#     for i in range(len(caps)):
#         cap[i] = [0]
#         cnt = 0
#         for j in range(len(caps[i])):
#             if caps[i][j] == 0:
#                 cap[i].append(0)
#                 cnt += 1
#             else:
#                 cap[i][cnt] += caps[i][j]
#
#     for i in range(p.depot_n):
#         if max(cap[i]) > p.vehicle_capacity:
#             No_over_capacity = False
#
#     for i in range(p.depot_n):
#         if routes[i].count(-1) >= p.max_departure:
#             No_over_capacity = False
#
#     return All_allocated, No_duplicated, No_over_capacity
#
# def print_solution(p, routes, caps, route_len, cal_time):
#     cap = {}
#     for i in range(len(caps)):
#         cap[i] = [0]
#         cnt = 0
#         for j in range(len(caps[i])):
#             if caps[i][j] == 0:
#                 cap[i].append(0)
#                 cnt += 1
#             else:
#                 cap[i][cnt] += caps[i][j]
#                 # cap[i][cnt] += 9
#
#     print("max_cap = ", p.vehicle_capacity, ", max_departure = ", p.max_departure)
#     for i in range(len(routes)):
#         print('Depot ', i, routes[i], 'cap:', cap[i], 'len:', int(route_len[i]*1000)/1000)
#
#     print('cal_time : ' + str(int(cal_time*1000)/1000) + ' sec')
#     print('route_length : ' + str(int(sum(route_len)*1000)/1000) + ' m')
#
#     All_allocated, No_duplicated, No_over_capacity = chk_solution(p, routes, caps)
#
#     str_title = "Problem : "
#     if All_allocated and No_duplicated and No_over_capacity:
#         str_title += "No"
#         pass
#
#     if not All_allocated:
#         str_title += ', Not all allocated'
#     if not No_duplicated:
#         str_title += ', duplicated'
#     if not No_over_capacity:
#         str_title += ', over_capacity'
#     print(str_title)