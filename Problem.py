from Params import *


def Prob(p_type, n_depot, n_customer, n_vehicle, seed):
    np.random.seed(seed)

    if p_type == 0: # MDVRP-(u)
        ## Uncapacitated : Depot has only one vehicle
        Car = Vehicle(0, 0, 1, 10000)

        Depots = []
        for i in range(n_depot):
            Depots.append(Depot(i, [copy.copy(Car)], np.random.random(2)))

        Customers = []
        for i in range(n_customer):
            Customers.append(Customer(i, 1, np.random.random(2)))

        p = Params(Depots, Customers)

    elif p_type == 1: # HMDVRP-(u)
        ## Uncapacitated : Depot has only one vehicle
        Car = Vehicle(0, 0, 1, 10000)
        Bike = Vehicle(0, 1, 1.2, 10000)
        Truck = Vehicle(0, 2, 0.5, 10000)
        veh_set = [Car, Bike, Truck]

        Depots = []
        for i in range(n_depot):
            Depots.append(Depot(i, [copy.copy(veh_set[i % 3])], np.random.random(2)))

        Customers = []
        for i in range(n_customer):
            Customers.append(Customer(i, 1, np.random.random(2)))

        p = Params(Depots, Customers)

    elif p_type == 2: # MDVRP-(c)
        ## Capacitated
        Car = Vehicle(0, 0, 1, int(40 * n_customer / n_depot))

        Depots = []
        for i in range(n_depot):
            Vehicles = []
            for j in range(int(n_customer / n_depot)):
                Vehicles.append(copy.copy(Car))
            Depots.append(Depot(i, Vehicles, np.random.random(2)))

        Customers = []
        for i in range(n_customer):
            Customers.append(Customer(i, np.random.randint(20, 30), np.random.random(2)))

        p = Params(Depots, Customers)

    elif p_type == 3:  # HMDVRP-(c)
        ## Capacitated
        Car = Vehicle(0, 0, 1, int(40 * n_customer / n_depot))
        Bike = Vehicle(0, 1, 1.2, int(30 * n_customer / n_depot))
        Truck = Vehicle(0, 2, 0.5, int(80 * n_customer / n_depot))
        veh_set = [Car, Bike, Truck]

        Depots = []
        for i in range(n_depot):
            Vehicles = []
            for j in range(int(n_customer / n_depot)):
                Vehicles.append(copy.copy(veh_set[i % 3]))
            Depots.append(Depot(i, Vehicles, np.random.random(2)))

        Customers = []
        for i in range(n_customer):
            Customers.append(Customer(i, np.random.randint(20, 30), np.random.random(2)))

        p = Params(Depots, Customers)

    elif p_type == 4:  # MDVRP-(c)-LV
        ## Capacitated
        Car = Vehicle(0, 0, 1, int(40 * n_customer / n_depot))

        Depots = []
        for i in range(n_depot):
            Vehicles = []
            for j in range(n_vehicle):
                Vehicles.append(copy.copy(Car))
            Depots.append(Depot(i, Vehicles, np.random.random(2)))

        Customers = []
        for i in range(n_customer):
            Customers.append(Customer(i, np.random.randint(20, 30), np.random.random(2)))

        p = Params(Depots, Customers)

    elif p_type == 5:  # HMDVRP-(c)-LV
        ## Capacitated
        Car = Vehicle(0, 0, 1, int(40 * n_customer / n_depot))
        Bike = Vehicle(0, 1, 1.2, int(30 * n_customer / n_depot))
        Truck = Vehicle(0, 2, 0.5, int(80 * n_customer / n_depot))
        veh_set = [Car, Bike, Truck]

        Depots = []
        for i in range(n_depot):
            Vehicles = []
            for j in range(n_vehicle):
                Vehicles.append(copy.copy(veh_set[i % 3]))
            Depots.append(Depot(i, Vehicles, np.random.random(2)))

        Customers = []
        for i in range(n_customer):
            Customers.append(Customer(i, np.random.randint(20, 30), np.random.random(2)))

        p = Params(Depots, Customers)

    else:
        Car = Vehicle(0, 0, 1, 10000)
        Depots = [Depot(0, [copy.copy(Car)], np.random.random(2))]
        Customers = []
        for i in range(10):
            Customers.append(Customer(i, 1, np.random.random(2)))
        p = Params(Depots, Customers)

    return p