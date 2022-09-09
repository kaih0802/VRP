import copy
import math
import numpy as np
#from scipy import spatial
import time
from alns import ALNS
from alns import State as alns_state
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations, NoImprovement, MaxRuntime
from alns.weights import SimpleWeights
import matplotlib.pyplot as plt

class VRP_state(alns_state):
    def __init__(self, current, removed, env):
        self.environment = env
        self.solution = current
        self.removed = removed
        self.vehicles = env.vehicles
        self.delivery_info = env.delivery_info
        self.distance = env.distance_matrix
        self.n_vehicles = env.n_vehicles
        self.para_multi = 10
        self.conv_time_to_cost =env.conv_time_to_cost
        delivery_id = []
        for v in range(len(current)):
            delivery_id += current[v]
        delivery_id = [value for value in delivery_id if value != 0]
        self.delivery_id = delivery_id + removed


    def obj_largrange(self):
        # evaluate solution quality, use lagrange relaxation for infeasible solution
        # USAGE COST
        VRP_solution = self.solution
        usage_cost = 0
        for k in range(self.n_vehicles):
            if len(VRP_solution[k]) > 0:
                usage_cost += self.vehicles[k]['cost']
        # TOUR COST and CHECK TIME WINDOWS
        travel_cost = 0
        time_violate_cost = 0
        for k in range(self.n_vehicles):
            travel_time = 0
            tour_time = 0
            for i in range(1, len(VRP_solution[k])):
                if i < len(VRP_solution[k]) - 1:
                    tour_time += self.distance[
                        VRP_solution[k][i - 1],
                        VRP_solution[k][i],
                    ]
                    tour_time = max(
                        tour_time, self.delivery_info[VRP_solution[k][i]]['time_window_min']
                    )
                    if tour_time > self.delivery_info[VRP_solution[k][i]]['time_window_max']:
                        time_violate_cost += (tour_time - self.delivery_info[VRP_solution[k][i]]['time_window_max'])*self.para_multi
                travel_time += self.distance[
                    VRP_solution[k][i - 1],
                    VRP_solution[k][i],
                ]
            travel_cost += self.conv_time_to_cost * tour_time

        # CHECK VOLUME
        cap_violate_cost = 0
        for k in range(self.n_vehicles):
            tot_vol_used = 0
            for i in range(1, len(VRP_solution[k]) - 1):
                tot_vol_used += self.delivery_info[VRP_solution[k][i]]['vol']

            if tot_vol_used > self.vehicles[k]['capacity']:
                cap_violate_cost += (tot_vol_used - self.vehicles[k]['capacity'])*self.para_multi
        total_cost = usage_cost + travel_cost + time_violate_cost + cap_violate_cost

        return total_cost

    def objective(self):
        if len(self.removed) > 0:
            raise Exception('Unsatisfied Customers: ', self.removed)
        else:
            return self.obj_largrange()

class Crowd_state(alns_state):
    def __init__(self, nodes, scenarios, env):
        self.ratio = 0.5
        self.environment = env
        self.solution = nodes
        self.scenarios = scenarios
        self.para_multi = 10
        self.vehicles = env.vehicles
        self.delivery_info = env.delivery_info
        self.distance = env.distance_matrix
        self.n_vehicles = env.n_vehicles
        self.conv_time_to_cost = env.conv_time_to_cost
        self.n_scenarios = len(scenarios)
        self.calculated_solution = np.arange(len(nodes)+1)
        self.calculated_obj_value = -1

    def greedyinsertion(self, deliveries):

        # in each iteration, insert the node with minimum insertion cost
        idx_deliveries = deliveries
        # Initialize a new route
        route = []
        for V in range(self.n_vehicles):
            route.append([])
        distance_matrix = self.distance
        deliveries = self.delivery_info
        vehicles = self.vehicles

        while len(idx_deliveries) != 0:
            # initialize a node to insert (id, insert_cost, route, position)
            best = (0, 10000, 0, 0)
            for node in idx_deliveries:
                for v in range(len(route)):
                    if len(route[v]) <= 2:

                        cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0]
                        if cost < best[1]:
                            best = (node, cost, v, 0)
                    else:
                        for p in range(1, (len(route[v]) - 1)):
                            cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                                   distance_matrix[route[v][p - 1], route[v][p]]
                            # Checking feasibility
                            new_tour = route[v].copy()
                            new_tour.insert(p, node)
                            tour_time = 0
                            cap = vehicles[v]['capacity']

                            # Penalty for violating time window
                            for i in range(1, (len(new_tour) - 1)):
                                cap -= deliveries[new_tour[i]]['vol']
                                tour_time += distance_matrix[new_tour[i - 1], new_tour[i]]
                                tour_time = max(tour_time,
                                                deliveries[new_tour[i]]['time_window_min'])
                                if tour_time > deliveries[new_tour[i]]['time_window_max']:
                                    cost += self.para_multi * (tour_time - deliveries[new_tour[i]]['time_window_max'])
                            # Penalty for violating capacity
                            if cap < 0:
                                cost -= self.para_multi * cap

                            if cost < best[1]:
                                best = (node, cost, v, p)
            # route[vehicle].insert(position, node_id)
            route[best[2]].insert(best[3], best[0])
            # remove node id from pending deliveries
            idx_deliveries.remove(best[0])
            if len(route[best[2]]) == 1:
                route[best[2]] = [0] + route[best[2]] + [0]
        if len(idx_deliveries) == 0:
            return route
        else:
            raise Exception('Error in greedy insertion')

    def objective(self):
        START = time.time()
        calculated = (list(self.solution) == list(self.calculated_solution))
        print(calculated)
        if calculated:
            END = time.time()
            print("REPEAT CALCULATION", int(END - START), " SECONDS",self.calculated_obj_value)
            return self.calculated_obj_value
        else:
            delivery_id = np.nonzero(self.solution)[0]
            delivery_id += 1
            crowd_id = np.nonzero(self.solution==0)[0]
            crowd_id += 1
            result = 0
            for s in self.scenarios:
                real_delivery = [value for value in crowd_id if value in s] + list(delivery_id)
                real_crowd = [value for value in crowd_id if value not in s]
                VRP_route = self.greedyinsertion(real_delivery)
                VRP_cost = VRP_state(VRP_route, [], self.environment)

                crowd_cost = 0
                for id in real_crowd:
                    crowd_cost += self.delivery_info[id]['crowd_cost']
                total_cost = VRP_cost.obj_largrange() + crowd_cost
                result += total_cost
            output = result / self.n_scenarios
            END = time.time()
            self.calculated_solution = self.solution
            self.calculated_obj_value = output
            print("#################obj COST: ", int(END - START), " SECONDS##############")
            return output


class NEW_agent:
    def __init__(self, env):
        self.env = env
        self.name = 'Heur38'
        self.quantile = 0.5
        self.data_improving_quantile = {
            "quantile": [],
            "performace": []
        }
        self.distance = env.distance_matrix
        self.para_repair_noise = 1
        self.destroy_degree = 0.3
        self.weights_destroy = np.ones(8, dtype=float)
        self.weights_repair = np.ones(8, dtype=float)
        self.n_scenarios = 3
        self.decrease = 0.995
        self.conflict_matrix = self.build_conflict_graph(self.env.distance_matrix, self.env.get_delivery())

    def build_conflict_graph(self, distance_matrix, deliveries):
        # build a confict graph, 1 means unable to follow time window constraint
        # -1 means in any case vehicle must wait in the node until the time window
        # 0 means time window can be satisfied
        #distance_matrix = self.env.distance_matrix
        #deliveries = self.env.delivery_info
        tw_min = [0]
        tw_max = [1000]  # set tw_max=1000 for depot, make sure not violate time window when back to depot

        for _, ele in deliveries.items():
            tw_min.append(ele['time_window_min'])
            tw_max.append(ele['time_window_max'])
        num_points = len(tw_min)
        conflict_graph = np.zeros((num_points,num_points))
        for i in range(num_points):
            for j in range(num_points):
                if tw_min[i] + distance_matrix[i, j] > tw_max[j]:
                    conflict_graph[i][j] = 1
                elif tw_max[i] + distance_matrix[i, j] < tw_min[j]:
                    conflict_graph[i][j] = -1
        #conflict_graph[:, 0] = 0
        return conflict_graph

    def sort_vehicles(self):
        # sort vehicles by efficiency (cost / capacity), return a list of vehicle's index
        vehicles = self.env.get_vehicles()
        vehicles_efficiency = []

        for vehicle in vehicles:
            cost_unit = vehicle['cost'] / vehicle['capacity']
            vehicles_efficiency.append(cost_unit)
        sorted_index = np.argsort(vehicles_efficiency)

        return sorted_index

    def greedy_insertion(self, deliverties, distance_matrix, vehicles, index_sorted_vehicles, conflict):
        conv_time = self.env.conv_time_to_cost
        # convert deliveries to a list of dictionaries
        list_deliveries = []
        for ele in deliverties.items():
            list_deliveries.append(ele[1])
        # initialize an empty route
        route = [[] for x in range(len(vehicles))]

        for x in index_sorted_vehicles:
            select_vehicle = vehicles[x]
            remain_cap = select_vehicle['capacity']
            subroute = []

            if len(list_deliveries)>0:
                # choose the customer i with the maximum volume, set subroute 0 -> i -> 0
                subroute = [0, 0]
                item_max = max(list_deliveries, key=lambda x: x['vol'])
                subroute.insert(1,item_max['id'])
                remain_cap -= item_max['vol']
                list_deliveries.remove(item_max)

            while len(list_deliveries) > 0 and remain_cap > 0:
                # cheapest insertion
                # initialize customer (delivery, insert_cost, violate_TW, insert_position)
                cheapest_customer = (0,10000,True,0)
                # find the cheapest
                for delivery in list_deliveries:
                    id = delivery['id']
                    for p in range(1,len(subroute)):

                        if conflict[subroute[p-1]][id] !=1 and conflict[id][subroute[p]] !=1 and remain_cap >= delivery['vol']:

                            tour_time = 0
                            feasible = True
                            after_insertion = subroute.copy()
                            after_insertion.insert(p,id)

                            # Verify if route is still feasible after insertion
                            for i in range(1, (len(after_insertion) - 1)):
                                tour_time += distance_matrix[after_insertion[i - 1],after_insertion[i]]
                                tour_time = max(tour_time,
                                                deliverties[after_insertion[i]]['time_window_min'])
                                if tour_time > deliverties[after_insertion[i]]['time_window_max']:
                                    feasible = False
                                    break

                            # cost = time * conv_time_to_cost, here just use time, time = distance
                            insert_cost = (distance_matrix[subroute[p-1]][id] + distance_matrix[id][subroute[p]] - distance_matrix[p-1][p]) * conv_time
                            if insert_cost < cheapest_customer[1] and feasible:
                                cheapest_customer = (delivery, insert_cost, False,p)

                if cheapest_customer[2] == False:
                    subroute.insert(cheapest_customer[3],cheapest_customer[0]['id'])
                    list_deliveries.remove(cheapest_customer[0])
                else:
                    break

            route[x] = subroute

        # after cheapest insertion procedure, if still some unsatisfied customer, randomly insert them
        if len(list_deliveries)>0:
            print('Cannot find feasible solution in construction phase, random insert')
            for remain_delivery in list_deliveries:
                v = np.random.randint(len(index_sorted_vehicles))
                route[v].insert((len(route[v]) - 1), remain_delivery['id'])

        return route

    def get_delivery_id(self, solution):
        delivery_id = []
        for v in range(len(solution)):
            delivery_id += solution[v]
        delivery_id = [value for value in delivery_id if value != 0]
        return delivery_id

    def delivery_to_list(self,deliverties):
        list_deliveries = []
        for ele in deliverties.items():
            list_deliveries.append(ele[1])
        return  list_deliveries

    '''*************************************Destroy operators*******************************************'''
    def destroy_random(self,state, rnd_state):
        # randomly remove number of nodes from current solution
        Current = copy.deepcopy(state)
        route= Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))
        removed_nodes = []
        for b in range(num_removal):
            v = np.random.randint(len(route))
            while len(route[v])<=2:
                v = np.random.randint(len(route))
            p = np.random.randint(1,(len(route[v])-1))
            node = route[v].pop(p)
            removed_nodes.append(node)

        # if a tour only left depot, empty it
        for n in range(len(route)):
            if len(route[n])<=2:
                route[n] = []

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    def destroy_route(self,state, rnd_state):
        # randomly destroy one or more entire routes
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))
        removed_nodes = []
        while len(removed_nodes) < num_removal:
            v = np.random.randint(len(route))
            while len(route[v]) <= 2:
                v = np.random.randint(len(route))
            subroute = route[v]
            route[v] = []
            nodes_removing = [node for node in subroute if node != 0]
            removed_nodes += nodes_removing

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    def destroy_worst_distance(self,state, rnd_state):
        # in each iteration remove the worst distance node (argmax_i {d_ji + d_ik - d_jk})
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))
        distance_matrix = Current.distance

        removed_nodes = []
        while len(removed_nodes) < num_removal:
            # initialize the node to remove (id, distance)
            node_need_remove = (-1, 0)
            # select the worst distance node
            for v in range(len(route)):
                if len(route[v]) > 2:
                    for i in range(1,(len(route[v])-1)):
                        node_i = route[v][i]
                        previous = route[v][i - 1]
                        next = route[v][i + 1]
                        d_i = distance_matrix[previous][node_i] + distance_matrix[node_i][next] - distance_matrix[previous][next]
                        if d_i > node_need_remove[1]:
                            node_need_remove = (node_i, d_i)
            # remove selected node from current solution
            for v in range(len(route)):
                if node_need_remove[0] in route[v]:
                    route[v].remove(node_need_remove[0])

            removed_nodes.append(node_need_remove[0])
        # if a tour only left depot, empty it
        for n in range(len(route)):
            if len(route[n])<=2:
                route[n] = []

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    def destroy_worst_time(self, state, rnd_state):
        # in each iteration remove the worst time node (argmax_i {|tw_i_min-t_i|})
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        removed_nodes = []
        while len(removed_nodes) < num_removal:
            # initialize the node to remove (id, time)
            node_need_remove = (-1, 0)

            # select the worst time node
            for v in range(len(route)):
                if len(route[v]) > 2:
                    tour_time = 0
                    for i in range(1, (len(route[v]) - 1)):
                        node_i = route[v][i]
                        previous = route[v][i - 1]
                        tour_time += distance_matrix[previous, node_i]
                        T_i = abs(deliveries[node_i]['time_window_min'] - tour_time)
                        tour_time = max(tour_time, deliveries[node_i]['time_window_min'])
                        if T_i > node_need_remove[1]:
                            node_need_remove = (node_i, T_i)
            # remove selected node from current solution
            for v in range(len(route)):
                if node_need_remove[0] in route[v]:
                    route[v].remove(node_need_remove[0])

            removed_nodes.append(node_need_remove[0])
        # if a tour only left depot, empty it
        for n in range(len(route)):
            if len(route[n]) <= 2:
                route[n] = []

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    def destroy_timewindow(self, state, rnd_state):
        # randomly choose one node to remove and select other nodes with similar time window
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))
        deliveries = Current.delivery_info
        delivery_id= Current.delivery_id.copy()


        random_index = np.random.randint(len(delivery_id))
        first_choice = delivery_id.pop(random_index)
        tw_first = deliveries[first_choice]['time_window_max']
        removed_nodes = [first_choice]
        # compute time window difference from first chosen node for other nodes and
        for i in range(len(delivery_id)):
            node_id = delivery_id[i]
            tw_difference = abs(deliveries[node_id]['time_window_max'] - tw_first)
            delivery_id[i] = (node_id, tw_difference)
        # sort nodes by ascending order and keep required number of nodes
        delivery_id.sort(key=lambda tup: tup[1])
        selected_nodes = delivery_id[:(num_removal - 1)]
        selected_nodes_id = [item[0] for item in selected_nodes]

        removed_nodes += selected_nodes_id
        # remove from current solution
        for v in range(len(route)):
            route[v] = [value for value in route[v] if (value not in removed_nodes)]
            if len(route[v]) <= 2:
                route[v] = []

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    def destroy_proximity(self, state, rnd_state):
        # randomly choose one node to remove and select other nodes with less distance from or to the node
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        delivery_id = Current.delivery_id.copy()

        random_index = np.random.randint(len(delivery_id))
        first_choice = delivery_id.pop(random_index)
        removed_nodes = [first_choice]
        # compute demand difference from first chosen node for other nodes and
        for i in range(len(delivery_id)):
            node_id = delivery_id[i]
            from_first_node = distance_matrix[first_choice][node_id]
            to_first_node = distance_matrix[node_id][first_choice]
            delivery_id[i] = (node_id, from_first_node,to_first_node)
        # sort nodes by ascending order and keep required number of nodes
        From = sorted(delivery_id, key=lambda tup: tup[1])
        To = sorted(delivery_id, key=lambda tup: tup[2])
        while len(removed_nodes) < num_removal:
            min_from = From.pop(0)[0]
            if min_from not in removed_nodes:
                removed_nodes.append(min_from)
            min_to = To.pop(0)[0]
            if min_to not in removed_nodes:
                removed_nodes.append(min_to)

        # remove from current solution
        for v in range(len(route)):
            route[v] = [value for value in route[v] if (value not in removed_nodes)]
            if len(route[v]) <=2:
                route[v] = []

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    def destroy_demand(self, state, rnd_state):
        # randomly choose one node to remove and select other nodes with similar demand
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))

        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        delivery_id = Current.delivery_id.copy()


        random_index = np.random.randint(len(delivery_id))
        first_choice = delivery_id.pop(random_index)
        demand_first = deliveries[first_choice]['vol']
        removed_nodes = [first_choice]
        # compute demand difference from first chosen node for other nodes and
        for i in range(len(delivery_id)):
            node_id = delivery_id[i]
            demand_difference = abs(deliveries[node_id]['vol'] - demand_first)
            delivery_id[i] = (node_id, demand_difference)
        # sort nodes by ascending order and keep required number of nodes
        delivery_id.sort(key=lambda tup: tup[1])
        selected_nodes =delivery_id[:(num_removal -1)]
        selected_nodes_id = [item[0] for item in selected_nodes]

        removed_nodes+=selected_nodes_id
        # remove from current solution
        for v in range(len(route)):
            route[v] = [value for value in route[v] if (value not in removed_nodes)]
            if len(route[v]) <=2:
                route[v] = []

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    def destroy_related(self, state, rnd_state):
        # randomly choose one node to remove and select other nodes with similar time window and long distance
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = math.ceil(self.destroy_degree * len(Current.delivery_id))
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        delivery_id = Current.delivery_id.copy()

        random_index = np.random.randint(len(delivery_id))
        first_choice = delivery_id.pop(random_index)
        # find which tour the random chosen node belongs to
        for v in range(len(route)):
            if first_choice in route[v]:
                same_tour = route[v]
        removed_nodes = [first_choice]
        tw_first = deliveries[first_choice]['time_window_max']
        # compute demand difference from first chosen node for other nodes and
        for i in range(len(delivery_id)):
            node_id = delivery_id[i]
            if node_id not in same_tour:
                weight = 0.5
            else:
                weight = 1
            tw_difference = abs(deliveries[node_id]['time_window_max'] - tw_first)
            from_first_node = distance_matrix[first_choice][node_id]
            to_first_node = distance_matrix[node_id][first_choice]
            relation = ((from_first_node + to_first_node) / 2 - tw_difference) * weight
            delivery_id[i] = (node_id, relation)
        # sort nodes by descending order and keep required number of nodes
        delivery_id.sort(key=lambda tup: tup[1], reverse = True)
        selected_nodes = delivery_id[:(num_removal - 1)]
        selected_nodes_id = [item[0] for item in selected_nodes]

        removed_nodes += selected_nodes_id
        # remove from current solution
        for v in range(len(route)):
            route[v] = [value for value in route[v] if (value not in removed_nodes)]
            if len(route[v]) <= 2:
                route[v] = []

        Current.solution = route
        Current.removed = removed_nodes
        return Current

    '''*************************************Repair operators*******************************************'''
    def repair_greedy(self, state, rnd_state):
        # in each iteration, insert the node with minimum insertion cost
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = self.destroy_degree * len(Current.delivery_id)
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        delivery_id = Current.delivery_id.copy()
        vehicles = Current.vehicles
        removed = Current.removed


        while len(removed) != 0:
            # initialize a node to insert (id, insert_cost, route, position)
            best = (0, 10000, 0, 0)
            for node in removed:
                for v in range(len(route)):
                    if len(route[v]) <= 2:
                        cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0]
                        if cost < best[1]:
                            best = (node,cost,v,0)
                    else:
                        for p in range(1,(len(route[v]) - 1)):
                            cost = distance_matrix[route[v][p-1],[node]] + distance_matrix[[node],route[v][p]] - distance_matrix[route[v][p-1],route[v][p]]
                            if cost < best[1]:
                                # check the feasibility of this potential insertion
                                feasible = True
                                new_tour = route[v].copy()
                                new_tour.insert(p,node)
                                tour_time = 0
                                cap = vehicles[v]['capacity']
                                for i in range(1, (len(new_tour) - 1)):
                                    cap -= deliveries[new_tour[i]]['vol']
                                    tour_time += distance_matrix[new_tour[i - 1], new_tour[i]]
                                    tour_time = max(tour_time,
                                                    deliveries[new_tour[i]]['time_window_min'])
                                    if tour_time > deliveries[new_tour[i]]['time_window_max'] or cap < 0:
                                        feasible = False
                                        break
                                if feasible:
                                    best = (node, cost, v, p)
            # check if found feasible insertion, if not switch to same method with noise
            if best[0]!= 0:
                route[best[2]].insert(best[3],best[0])
                removed.remove(best[0])
                if len(route[best[2]])==1:
                    route[best[2]] = [0] + route[best[2]] + [0]
            else:
                #print("cannot find feasible insertion, add noise to generate a solution")
                Current.solution = route
                Current.removed = removed
                new_state = self.repair_greedy_noise(Current,rnd_state)
                return new_state
                # break
        Current.solution = route
        Current.removed = removed
        return Current

    def repair_greedy_noise(self, state, rnd_state):
        # in each iteration, insert the node with minimum insertion cost
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = self.destroy_degree * len(Current.delivery_id)
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        delivery_id = Current.delivery_id.copy()
        vehicles = Current.vehicles
        removed = Current.removed

        while len(removed) != 0:
            # initialize a node to insert (id, insert_cost, route, position)
            best = (0, 10000, 0, 0)
            for node in removed:
                for v in range(len(route)):
                    if len(route[v]) <= 2:
                        cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0] + np.random.normal() * self.para_repair_noise
                        if cost < best[1]:
                            best = (node, cost, v, 0)
                    else:
                        for p in range(1, (len(route[v]) - 1)):
                            cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                                   distance_matrix[route[v][p - 1], route[v][p]] + np.random.normal() * self.para_repair_noise
                            if cost < best[1]:
                                best = (node, cost, v, p)

            route[best[2]].insert(best[3], best[0])
            removed.remove(best[0])
            if len(route[best[2]]) == 1:
                route[best[2]] = [0] + route[best[2]] + [0]

        Current.solution = route
        Current.removed = removed
        return Current

    def repair_regret(self, state, rnd_state):
        # in each iteration, insert the node with minimum insertion cost
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = self.destroy_degree * len(Current.delivery_id)
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        delivery_id = Current.delivery_id.copy()
        vehicles = Current.vehicles
        removed = Current.removed

        while len(removed) != 0:
            # initialize a node to insert (regret, (id,insert_cost, best_route, best_position))
            regret = (0, 0)

            for node in removed:
                # (id, insert_cost, route, position)
                best = (0, 10000, 0, 0)
                second = (0, 10000, 0, 0)
                for v in range(len(route)):
                    if len(route[v]) <= 2:
                        cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0]
                        if cost < second[1]:
                            if cost < best[1]:
                                second = best
                                best = (node, cost, v, 0)
                            else:
                                second = (node, cost, v, 0)
                    else:
                        for p in range(1, (len(route[v]) - 1)):
                            cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                                   distance_matrix[route[v][p - 1], route[v][p]]
                            if cost < second[1]:
                                # check the feasibility of this potential insertion
                                feasible = True
                                new_tour = route[v].copy()

                                new_tour.insert(p, node)
                                tour_time = 0
                                cap = vehicles[v]['capacity']
                                for i in range(1, (len(new_tour) - 1)):
                                    cap -= deliveries[new_tour[i]]['vol']
                                    tour_time += distance_matrix[new_tour[i - 1], new_tour[i]]
                                    tour_time = max(tour_time,
                                                    deliveries[new_tour[i]]['time_window_min'])
                                    if tour_time > deliveries[new_tour[i]]['time_window_max'] or cap < 0:
                                        feasible = False
                                        break
                                if feasible:
                                    if cost < best[1]:
                                        second = best
                                        best = (node, cost, v, p)
                                    else:
                                        second = (node, cost, v, p)
                if (second[1] - best[1]) > regret[0]:
                    new_r_value = second[1] - best[1]
                    regret = (new_r_value, best)
            # check if found feasible insertion, if not switch to same method with noise
            if regret[1] != 0:
                insertion = regret[1]
                route[insertion[2]].insert(insertion[3], insertion[0])
                removed.remove(insertion[0])
                if len(route[insertion[2]]) == 1:
                    route[insertion[2]] = [0] + route[insertion[2]] + [0]
            else:
                #print("cannot find feasible insertion, add noise to generate a solution")
                Current.solution = route
                Current.removed = removed
                new_state = self.repair_regret_noise(Current, rnd_state)
                return new_state
        Current.solution = route
        Current.removed = removed
        return Current

    def repair_regret_noise(self, state, rnd_state):
        # regret with noise
        Current = copy.deepcopy(state)
        route = Current.solution
        num_removal = self.destroy_degree * len(Current.delivery_id)
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        delivery_id = Current.delivery_id.copy()
        vehicles = Current.vehicles
        removed = Current.removed

        while len(removed) != 0:
            # initialize a node to insert (regret, (id,insert_cost, best_route, best_position))
            regret = (0, 0)

            for node in removed:
                # (id, insert_cost, route, position)
                best = (0, 10000, 0, 0)
                second = (0, 10000, 0, 0)
                for v in range(len(route)):
                    if len(route[v]) <= 2:
                        cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0] + np.random.normal() * self.para_repair_noise
                        if cost < second[1]:
                            if cost < best[1]:
                                second = best
                                best = (node, cost, v, 0)
                            else:
                                second = (node, cost, v, 0)
                    else:
                        for p in range(1, (len(route[v]) - 1)):
                            cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                                   distance_matrix[route[v][p - 1], route[v][p]] + np.random.normal() * self.para_repair_noise
                            if cost < second[1]:
                                if cost < best[1]:
                                    second = best
                                    best = (node, cost, v, p)
                                else:
                                    second = (node, cost, v, p)
                if (second[1] - best[1]) > regret[0]:
                    new_r_value = second[1] - best[1]
                    regret = (new_r_value, best)

            insertion = regret[1]
            route[insertion[2]].insert(insertion[3], insertion[0])
            removed.remove(insertion[0])
            if len(route[insertion[2]]) == 1:
                route[insertion[2]] = [0] + route[insertion[2]] + [0]

        Current.solution = route
        Current.removed = removed
        return Current

    def repair_random(self, state, rnd_state):
        # in each iteration, randomly chose node and insert it in the best feasible position
        Current = copy.deepcopy(state)
        route = Current.solution
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        vehicles = Current.vehicles
        removed = Current.removed

        remaining_nodes = []
        while len(removed) != 0:
            # initialize a node to insert (id, insert_cost, route, position)
            best = (0, 10000, 0, 0)
            pop_idx = np.random.randint(len(removed))
            node = removed[pop_idx]
            for v in range(len(route)):
                if len(route[v]) <= 2:
                    cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0]
                    if cost < best[1]:
                        best = (node, cost, v, 0)
                else:
                    for p in range(1, (len(route[v]) - 1)):
                        cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                               distance_matrix[route[v][p - 1], route[v][p]]
                        if cost < best[1]:
                            # check the feasibility of this potential insertion
                            feasible = True
                            new_tour = route[v].copy()
                            new_tour.insert(p, node)
                            tour_time = 0
                            cap = vehicles[v]['capacity']
                            for i in range(1, (len(new_tour) - 1)):
                                cap -= deliveries[new_tour[i]]['vol']
                                tour_time += distance_matrix[new_tour[i - 1], new_tour[i]]
                                tour_time = max(tour_time,
                                                deliveries[new_tour[i]]['time_window_min'])
                                if tour_time > deliveries[new_tour[i]]['time_window_max'] or cap < 0:
                                    feasible = False
                                    break
                            if feasible:
                                best = (node, cost, v, p)
            # check if found feasible insertion, if not switch to same method with noise

            if best[0] != 0:
                route[best[2]].insert(best[3], best[0])
                if len(route[best[2]]) == 1:
                    route[best[2]] = [0] + route[best[2]] + [0]
            else:
                remaining_nodes.append(node)
            removed.remove(node)
        if len(remaining_nodes)>0:
            #print("cannot find feasible insertion, add noise to generate a solution")
            Current.solution = route
            Current.removed = remaining_nodes
            new_state = self.repair_random_noise(Current, rnd_state)
            return new_state
        else:
            Current.solution = route
            Current.removed = remaining_nodes
            return Current

    def repair_random_noise(self, state, rnd_state):
        # in each iteration, randomly chose node and insert it in the best position with noise
        Current = copy.deepcopy(state)
        route = Current.solution
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        #delivery_id = Current.delivery_id.copy()
        vehicles = Current.vehicles
        removed = Current.removed

        while len(removed) != 0:
            # initialize a node to insert (id, insert_cost, route, position)
            best = (0, 10000, 0, 0)
            pop_idx = np.random.randint(len(removed))
            node = removed.pop(pop_idx)
            for v in range(len(route)):
                if len(route[v]) <= 2:
                    cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0] + np.random.normal() * self.para_repair_noise
                    if cost < best[1]:
                        best = (node, cost, v, 0)
                else:
                    for p in range(1, (len(route[v]) - 1)):
                        cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                               distance_matrix[route[v][p - 1], route[v][p]]
                        if cost < best[1]:
                            best = (node, cost, v, p)

            route[best[2]].insert(best[3], best[0])
            if len(route[best[2]]) == 1:
                route[best[2]] = [0] + route[best[2]] + [0]

        Current.solution = route
        Current.removed = removed
        return Current

    def repair_time(self, state, rnd_state):
        # in each iteration, chose node with most tight tw and insert it in the best feasible position
        Current = copy.deepcopy(state)
        route = Current.solution
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        vehicles = Current.vehicles
        removed = Current.removed

        remaining_nodes = []
        for i in range(len(removed)):
            idx = removed[i]
            tw = deliveries[idx]['time_window_max'] - deliveries[idx]['time_window_min']
            removed[i] = (idx, tw)
        removed.sort(key=lambda tup: tup[1])
        while len(removed) != 0:
            # initialize a node to insert (id, insert_cost, route, position)
            best = (0, 10000, 0, 0)
            node = removed[0][0]
            for v in range(len(route)):
                if len(route[v]) <= 2:
                    cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][0]
                    if cost < best[1]:
                        best = (node, cost, v, 0)
                else:
                    for p in range(1, (len(route[v]) - 1)):
                        cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                               distance_matrix[route[v][p - 1], route[v][p]]
                        if cost < best[1]:
                            # check the feasibility of this potential insertion
                            feasible = True
                            new_tour = route[v].copy()
                            new_tour.insert(p, node)
                            tour_time = 0
                            cap = vehicles[v]['capacity']
                            for i in range(1, (len(new_tour) - 1)):
                                cap -= deliveries[new_tour[i]]['vol']
                                tour_time += distance_matrix[new_tour[i - 1], new_tour[i]]
                                tour_time = max(tour_time,
                                                deliveries[new_tour[i]]['time_window_min'])
                                if tour_time > deliveries[new_tour[i]]['time_window_max'] or cap < 0:
                                    feasible = False
                                    break
                            if feasible:
                                best = (node, cost, v, p)
            # check if found feasible insertion, if not switch to same method with noise
            if best[0] != 0:
                route[best[2]].insert(best[3], best[0])
                if len(route[best[2]]) == 1:
                    route[best[2]] = [0] + route[best[2]] + [0]
            else:
                remaining_nodes.append(node)
            removed.pop(0)
        if len(remaining_nodes) > 0:
            #print("cannot find feasible insertion, add noise to generate a solution")
            Current.solution = route

            Current.removed = remaining_nodes
            new_state = self.repair_random_noise(Current, rnd_state)
            return new_state
        else:
            Current.solution = route
            Current.removed = remaining_nodes
            return Current

    def repair_time_noise(self, state, rnd_state):
        Current = copy.deepcopy(state)
        route = Current.solution
        distance_matrix = Current.distance
        deliveries = Current.delivery_info
        #delivery_id = Current.delivery_id.copy()
        vehicles = Current.vehicles
        removed = Current.removed

        for i in range(len(removed)):
            idx = removed[i]
            tw = deliveries[idx]['time_window_max'] - deliveries[idx]['time_window_min'] #+ np.random.normal() * self.para_repair_noise
            removed[i] = (idx, tw)
        removed.sort(key=lambda tup: tup[1])
        while len(removed) != 0:
            # initialize a node to insert (id, insert_cost, route, position)
            best = (0, 10000, 0, 0)
            item = removed.pop(0)
            node = item[0]
            for v in range(len(route)):
                if len(route[v]) <= 2:
                    cost = vehicles[v]['cost'] + distance_matrix[0][node] + distance_matrix[node][
                        0] + np.random.normal() * self.para_repair_noise
                    if cost < best[1]:
                        best = (node, cost, v, 0)
                else:
                    for p in range(1, (len(route[v]) - 1)):
                        cost = distance_matrix[route[v][p - 1], [node]] + distance_matrix[[node], route[v][p]] - \
                               distance_matrix[route[v][p - 1], route[v][p]]
                        if cost < best[1]:
                            best = (node, cost, v, p)

            route[best[2]].insert(best[3], best[0])
            if len(route[best[2]]) == 1:
                route[best[2]] = [0] + route[best[2]] + [0]

        Current.solution = route
        Current.removed = removed
        return Current

    '''*********************************Operators for crowdshipping***********************************'''
    def crowd_destroy_cost(self, state, rnd_state):
        START = time.time()
        current = copy.deepcopy(state)
        delivery_info = current.delivery_info
        solution = current.solution
        # Add some noise to the number of nodes changing
        degree = math.ceil(len(solution) * (current.ratio + np.random.uniform(-0.05,0.05)))
        if degree < 0:
            print("Number of nodes smaller than 1")
            degree = 1
        vehicle_shipping = np.nonzero(solution)[0]
        vehicle_shipping +=1
        if degree > len(vehicle_shipping):
            degree = len(vehicle_shipping)
        while degree >0:
            # Initialize a nodes to be removed from VRP delivery (index, crowd_cost)
            least_cost = (0,1000)
            # Find the node in vehicle_shipping which has minimun crowd_cost
            for node in vehicle_shipping:
                if delivery_info[node]['crowd_cost'] < least_cost[1]:
                    least_cost = (node,delivery_info[node]['crowd_cost'])
            # Remove the node from vehicle_shipping and push it to crowdshipping
            vehicle_shipping = np.delete(vehicle_shipping, np.where(vehicle_shipping == least_cost[0]))
            solution[least_cost[0]-1] = 0
            degree -= 1

        current.solution = solution
        END = time.time()
        print("#################destroy_cost COST: ", int(END-START), " SECONDS##############")
        return current

    def crowd_destroy_skip(self, state, rnd_state):
        print("$$$$$$$$$$skip destory")
        return state

    def crowd_destroy_distance(self, state, rnd_state):
        START = time.time()
        current = copy.deepcopy(state)
        solution = current.solution
        # Add some noise to the number of nodes changing
        degree = math.ceil(len(solution) * (current.ratio + np.random.uniform(-0.05, 0.05)))
        if degree < 0:
            print("Number of nodes smaller than 1")
            degree = 1
        vehicle_shipping = np.nonzero(solution)[0]
        vehicle_shipping += 1
        # Build distance matrix for remaining nodes
        distance_matrix = self.distance
        crowd_shipping = np.nonzero(solution == 0)[0]
        crowd_shipping += 1
        distance_matrix_remained = np.delete(distance_matrix, crowd_shipping, axis=0)
        distance_matrix_remained = np.delete(distance_matrix_remained, crowd_shipping, axis=1)

        if degree > len(vehicle_shipping):
            degree = len(vehicle_shipping)

        while degree > 0:
            # Initialize a nodes to be removed from VRP delivery (index, distance, index in matrix)
            max_distance = (0, 0, 0)
            # Find the node in vehicle_shipping which has minimun crowd_cost
            for n in range(len(vehicle_shipping)):
                node_id = vehicle_shipping[n]

                idx_remained_matrix = n + 1
                distance =  np.partition(distance_matrix_remained[idx_remained_matrix, :],1)[1]
                distance += np.partition(distance_matrix_remained[:, idx_remained_matrix],1)[1]
                if distance > max_distance[1]:
                    max_distance = (node_id, distance, idx_remained_matrix)
            # Remove the node from vehicle_shipping and push it to crowdshipping
            vehicle_shipping = np.delete(vehicle_shipping, np.where(vehicle_shipping == max_distance[0]))
            solution[max_distance[0] - 1] = 0
            # Update the remaining distance matrix
            distance_matrix_remained = np.delete(distance_matrix_remained, max_distance[2], axis=0)
            distance_matrix_remained = np.delete(distance_matrix_remained, max_distance[2], axis=1)
            degree -= 1

        current.solution = solution
        END = time.time()
        print("#################destroy_distance COST: ", int(END - START), " SECONDS##############")
        return current

    def crowd_destroy_tw(self, state, rnd_state):
        START = time.time()
        current = copy.deepcopy(state)
        delivery_info = current.delivery_info
        solution = current.solution
        # Add some noise to the number of nodes changing
        degree = math.ceil(len(solution) * (current.ratio + np.random.uniform(-0.05, 0.05)))
        if degree < 0:
            print("Number of nodes smaller than 1")
            degree = 1
        vehicle_shipping = np.nonzero(solution)[0]
        vehicle_shipping += 1

        conflict_matrix = self.conflict_matrix
        crowd_shipping = np.nonzero(solution == 0)[0]
        crowd_shipping += 1
        conflict_matrix_remained = np.delete(conflict_matrix, crowd_shipping, axis=0)
        conflict_matrix_remained = np.delete(conflict_matrix_remained, crowd_shipping, axis=1)

        if degree > len(vehicle_shipping):
            degree = len(vehicle_shipping)

        while degree > 0:
            # Initialize a nodes to be removed from VRP delivery (index, crowd_cost, index in matrix)
            max_conflict = (0, -1, 0)
            # Find the node in vehicle_shipping which has minimun crowd_cost
            for n in range(len(vehicle_shipping)):
                node_id = vehicle_shipping[n]
                idx_remained_matrix = n+1
                conflict =np.count_nonzero(conflict_matrix_remained[idx_remained_matrix,:] == 1)
                conflict += np.count_nonzero(conflict_matrix_remained[:, idx_remained_matrix] == 1)
                if conflict > max_conflict[1]:
                    max_conflict = (node_id, conflict,idx_remained_matrix)
            # Remove the node from vehicle_shipping and push it to crowdshipping
            vehicle_shipping = np.delete(vehicle_shipping, np.where(vehicle_shipping == max_conflict[0]))
            solution[max_conflict[0] - 1] = 0
            # Update the remaining conflict matrix
            conflict_matrix_remained = np.delete(conflict_matrix_remained, max_conflict[2], axis=0)
            conflict_matrix_remained = np.delete(conflict_matrix_remained, max_conflict[2], axis=1)
            degree -= 1

        current.solution = solution
        END = time.time()
        print("#################destroy_tw COST: ", int(END - START), " SECONDS##############")
        return current

    def crowd_repair_skip(self, state, rnd_state):
        state.ratio *= self.decrease
        print("$$$$$$$$$$skip repair")
        return state

    def crowd_repair_cost(self, state, rnd_state):
        START = time.time()
        solution = state.solution.copy()
        delivery_info = state.delivery_info
        # Add some noise to the number of nodes changing
        degree = math.ceil(len(solution) * (state.ratio + np.random.uniform(-0.05, 0.05)))
        if degree < 0:
            print("Number of nodes smaller than 1")
            degree = 1
        crowd_shipping = np.nonzero(solution==0)[0]
        crowd_shipping += 1
        if degree > len(crowd_shipping):
            degree = len(crowd_shipping)
        while degree > 0:
            # Initialize a nodes to be removed from VRP delivery (index, crowd_cost)
            max_cost = (0, 0)
            # Find the node in crowd_shipping which has minimun crowd_cost
            for node in crowd_shipping:
                if delivery_info[node]['crowd_cost'] > max_cost[1]:
                    max_cost = (node, delivery_info[node]['crowd_cost'])
            # Remove the node from crowd_shipping and push it to vehichle shipping
            crowd_shipping = np.delete(crowd_shipping, np.where(crowd_shipping == max_cost[0]))
            solution[max_cost[0] - 1] = 1
            degree -= 1

        state.solution = solution
        state.ratio *= self.decrease
        END = time.time()
        print("#################repair_cost COST: ", int(END - START), " SECONDS##############")
        return state

    def crowd_repair_distance(self, state, rnd_state):
        START = time.time()
        solution = state.solution.copy()
        distance_matrix = state.distance
        delivery_info = state.delivery_info
        # Add some noise to the number of nodes changing
        degree = math.ceil(len(solution) * (state.ratio + np.random.uniform(-0.05, 0.05)))
        if degree < 0:
            print("Number of nodes smaller than 1")
            degree = 1
        crowd_shipping = np.nonzero(solution == 0)[0]
        crowd_shipping += 1
        idx = np.arange((len(solution) + 1))
        if degree > len(crowd_shipping):
            degree = len(crowd_shipping)
        while degree > 0:
            # Initialize a nodes to be removed from VRP delivery (index, crowd_cost)
            min_dist = (0, 1000)
            # Find the node in crowd_shipping which has minimun crowd_cost
            for node in crowd_shipping:
                #construct a new distance matrix with only Vehicle shippings and "node"
                crowd_others = np.delete(crowd_shipping, np.where(crowd_shipping == node))
                distance_matrix_remained = np.delete(distance_matrix, crowd_others, axis=0)
                distance_matrix_remained = np.delete(distance_matrix_remained, crowd_others, axis=1)
                # find the minimum distance from and to the node
                idx_remained = np.delete(idx,crowd_others)
                idx_node_matrix, = np.where(idx_remained == node)
                distance = np.partition(distance_matrix_remained[idx_node_matrix[0], :], 1)[1]
                distance += np.partition(distance_matrix_remained[:, idx_node_matrix[0]], 1)[1]

                # Select the node with min distance
                if distance < min_dist[1]:
                    min_dist = (node, distance)
            # Remove the node from crowd_shipping and push it to vehicle
            crowd_shipping = np.delete(crowd_shipping, np.where(crowd_shipping == min_dist[0]))
            solution[min_dist[0] - 1] = 1
            degree -= 1

        state.solution = solution
        state.ratio *= self.decrease
        END = time.time()
        print("#################repair_distance COST: ", int(END - START), " SECONDS##############")
        return state

    def crowd_repair_prob(self, state, rnd_state):
        START = time.time()
        solution = state.solution.copy()
        delivery_info = state.delivery_info
        # Add some noise to the number of nodes changing
        degree = math.ceil(len(solution) * (state.ratio + np.random.uniform(-0.05, 0.05)))
        if degree < 0:
            print("Number of nodes smaller than 1")
            degree = 1
        crowd_shipping = np.nonzero(solution==0)[0]
        crowd_shipping += 1
        if degree > len(crowd_shipping):
            degree = len(crowd_shipping)
        while degree > 0:
            # Initialize a nodes to be removed from VRP delivery (index, crowd_cost)
            max_prob = (0, 0)
            # Find the node in crowd_shipping which has the highest fail probability
            for node in crowd_shipping:
                if delivery_info[node]['p_failed'] > max_prob[1]:
                    max_prob = (node, delivery_info[node]['p_failed'])
            # Remove the node from crowd_shipping and push it to crowdshipping
            crowd_shipping = np.delete(crowd_shipping, np.where(crowd_shipping == max_prob[0]))
            solution[max_prob[0] - 1] = 1
            degree -= 1

        state.solution = solution
        state.ratio *= self.decrease
        END = time.time()
        print("#################repair_prob COST: ", int(END - START), " SECONDS##############")
        return state

    '''*******************************************ALNS************************************************'''
    def perform_ALNS(self, init_sol, stop_crit, stop_value):
        if stop_crit == 'time':
            stop = MaxRuntime(stop_value)
        elif stop_crit == 'iteration':
            stop = MaxIterations(stop_value)
        elif stop_crit == 'improve':
            stop = NoImprovement(stop_value)
        else:
            raise Exception('Invalid stopping criterion')
        alns = ALNS()
        # add destroy operator
        alns.add_destroy_operator(self.destroy_demand)
        alns.add_destroy_operator(self.destroy_route)
        alns.add_destroy_operator(self.destroy_related)
        alns.add_destroy_operator(self.destroy_worst_distance)
        alns.add_destroy_operator(self.destroy_worst_time)
        alns.add_destroy_operator(self.destroy_timewindow)
        alns.add_destroy_operator(self.destroy_proximity)
        alns.add_destroy_operator(self.destroy_random)
        # add repair operator
        alns.add_repair_operator(self.repair_random)
        alns.add_repair_operator(self.repair_regret)
        alns.add_repair_operator(self.repair_time)
        alns.add_repair_operator(self.repair_greedy)
        alns.add_repair_operator(self.repair_random_noise)
        alns.add_repair_operator(self.repair_regret_noise)
        alns.add_repair_operator(self.repair_time_noise)
        alns.add_repair_operator(self.repair_greedy_noise)

        crit = SimulatedAnnealing(start_temperature=1_000,
                                  end_temperature=1,
                                  step=1 - 1e-3,
                                  method="exponential")

        weights = SimpleWeights(scores=[5, 2, 1, 0.5],
                                num_destroy=8,
                                num_repair=8,
                                op_decay=0.8)
        # Keep memory of weights
        weights._d_weights = self.weights_destroy
        weights._r_weights = self.weights_repair

        result = alns.iterate(init_sol, weights, crit, stop)

        # Update weights
        self.weights_destroy = weights._d_weights
        self.weights_repair = weights._r_weights

        return result#.best_state

    def crowd_ALNS(self, init_sol, stop_crit, stop_value):
        if stop_crit == 'time':
            stop = MaxRuntime(stop_value)
        elif stop_crit == 'iteration':
            stop = MaxIterations(stop_value)
        elif stop_crit == 'improve':
            stop = NoImprovement(stop_value)
        else:
            raise Exception('Invalid stopping criterion')
        alns = ALNS()
        # add destroy operator
        alns.add_destroy_operator(self.crowd_destroy_cost)
        alns.add_destroy_operator(self.crowd_destroy_skip)
        alns.add_destroy_operator(self.crowd_destroy_distance)
        alns.add_destroy_operator(self.crowd_destroy_tw)
        # add repair operator
        alns.add_repair_operator(self.crowd_repair_cost)
        alns.add_repair_operator(self.crowd_repair_skip)
        alns.add_repair_operator(self.crowd_repair_prob)
        alns.add_repair_operator(self.crowd_repair_distance)

        crit = SimulatedAnnealing(start_temperature=1_000,
                                  end_temperature=1,
                                  step=1 - 1e-3,
                                  method="exponential")

        weights = SimpleWeights(scores=[5, 2, 1, 0.5],
                                num_destroy=4,
                                num_repair=4,
                                op_decay=0.8)
        result = alns.iterate(init_sol, weights, crit, stop)

        return result   #.best_state

    def generate_scenario(self):
        fail_crowdship = []
        for _, ele in self.env.delivery_info.items():
            if np.random.uniform() < ele['p_failed']:
                fail_crowdship.append(ele['id'])
        return fail_crowdship

    def compute_delivery_to_crowdship(self, deliveries):
        # generate scenarios for crowdsourcing
        scenarios = []
        for s in range(self.n_scenarios):
            failed = self.generate_scenario()
            scenarios.append(failed)
        self.scenarios = scenarios
        print(scenarios)
        if len(deliveries) == 0:
            return []
        init_nodes = np.ones(len(deliveries))
        threshold = np.quantile(self.distance[0, :], self.quantile)

        for i in range(len(self.distance[0, :])):
            if self.distance[0, i] > threshold:
                init_nodes[i-1] = 0
        START = time.time()
        init_state = Crowd_state(init_nodes, scenarios, self.env)
        print(init_state.objective())
        ALNS_result = self.crowd_ALNS(init_state, 'iteration', 10)
        END = time.time()
        print("#################COST: ", int(END-START), " SECONDS##############")
        state_best= ALNS_result.best_state
        crowdshipping = state_best.solution
        id_to_crowdship = np.nonzero(crowdshipping==0)[0]
        id_to_crowdship =list(id_to_crowdship+1)
        try:
            print("Plot")
            _, ax = plt.subplots(figsize=(12, 6))
            ALNS_result.plot_objectives(ax=ax, lw=2)


            plt.figure()
            plt.plot(np.minimum.accumulate(ALNS_result.statistics.objectives))
            plt.show()

        except:
            print("ERROR")

        return crowdshipping,id_to_crowdship

    def learn_and_save(self):
        distance_matrix = self.env.distance_matrix
        conflict_matrix = self.build_conflict_graph(distance_matrix, self.env.get_delivery())

        id_deliveries_to_crowdship = self.compute_delivery_to_crowdship_ref(self.env.get_delivery())
        # for i in range(len(id_deliveries_to_crowdship)):
        #     id_deliveries_to_crowdship[i] = str(id_deliveries_to_crowdship)
        remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)

        Vehicles = self.env.get_vehicles()
        index_sorted_vehicles = self.sort_vehicles()
        # VRP = self.greedy_insertion(remaining_deliveries, distance_matrix, Vehicles, index_sorted_vehicles, conflict_matrix)
        # return VRP
        START = time.time()
        initial_route = self.greedy_insertion(self.env.get_delivery(), distance_matrix, Vehicles,
                                                    index_sorted_vehicles, conflict_matrix)
        init_sol = VRP_state(initial_route, [], self.env)
        result = self.perform_ALNS(init_sol,'time',250)
        solution = result.best_state
        END = time.time()
        try:
            _, ax = plt.subplots(figsize=(12, 6))
            result.plot_objectives(ax=ax, lw=2)


            plt.figure()
            plt.plot(np.minimum.accumulate(result.statistics.objectives))

        except:
            print("ERROR")

        print("#######################################COST: ", int(END-START), " SECONDS#############################")
        return solution.solution



        #VRP_solution = self.compute_VRP(remaining_deliveries, self.env.get_vehicles())

        # obj = self.env.evaluate_VRP(VRP_solution)
        #
        # self.data_improving_quantile['quantile'].append(self.quantile)
        # self.data_improving_quantile['performace'].append(tot_crowd_cost + obj)

    def start_test(self):
        pass

if __name__ == '__main__':
    pass