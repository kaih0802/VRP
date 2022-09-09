# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from envs.deliveryNetwork import DeliveryNetwork
from agents.new_agent import NEW_agent

if __name__ == '__main__':
    np.random.seed(0)
    # log_name = "./logs/test_heur1.log"v
    # logging.basicConfig(
    #     filename=log_name,
    #     format='%(asctime)s %(levelname)s: %(message)s',
    #     level=logging.INFO, datefmt="%H:%M:%S",
    #     filemode='w'
    # )
    fp = open("./cfg/setting_1.json", 'r')
    settings = json.load(fp)
    fp.close()

    data_csv = "./cfg/distance_matrix.csv"

    fp2 = open("./cfg/delivery_info.json", 'r')
    delivery_info = json.load(fp2)
    fp2.close()
    delivery_info = {int(k): v for k, v in delivery_info.items()}

    distance = np.genfromtxt(data_csv, delimiter=',')
    env = DeliveryNetwork(settings,delivery_info, distance)

    print(env.vehicles)
    print(env.delivery_info)
    env.prepare_crowdsourcing_scenario()
    print("####################")

    agent = NEW_agent(env)

    S = agent.compute_delivery_to_crowdship(env.get_delivery())


    print(S[0])
    print(S[1])

    # print('*************************')
    #
    # X = agent.learn_and_save()
    # print(X)
    # print('*************************')
    # # State = VRP_state(X,[],env)
    # # print(State.objective())
    # id_deliveries_to_crowdship = agent.compute_delivery_to_crowdship(env.get_delivery())
    # remaining_deliveries, tot_crowd_cost = env.run_crowdsourcing(id_deliveries_to_crowdship)
    # env.render_tour(remaining_deliveries, X)
    # obj = env.evaluate_VRP(X)
    # print("obj: ", obj)
    #
    #
    # # state1 = agent.destroy_demand(State)
    # # print(state1.solution)
    # #
    # # #env.render_tour(remaining_deliveries, x_1)
    # # state2 = agent.repair_random(state1)
    # # print(state2.solution)
    # # env.render_tour(remaining_deliveries, state2.solution)
    # # obj = env.evaluate_VRP(new_solution)
    # # print("obj: ", obj)
    #
    # # env.prepare_crowdsourcing_scenario()
    # #
    # #
    # # id_deliveries_to_crowdship = agent.compute_delivery_to_crowdship(
    # #     env.get_delivery()
    # # )
    # # print("id_deliveries_to_crowdship: ", id_deliveries_to_crowdship)
    # # remaining_deliveries, tot_crowd_cost = env.run_crowdsourcing(id_deliveries_to_crowdship)
    # # print("remaining_deliveries: ", remaining_deliveries )
    # # print("tot_crowd_cost: ", tot_crowd_cost)
    # # VRP_solution = agent.nearestneighbor(remaining_deliveries)
    # # #VRP_solution = agent.constructNN(remaining_deliveries)
    # # print("VRP_solution_exact: ", VRP_solution)
    # #
    # # env.render_tour(remaining_deliveries, VRP_solution)
    # # obj = env.evaluate_VRP(VRP_solution)
    # # print("obj: ", obj)
    #
