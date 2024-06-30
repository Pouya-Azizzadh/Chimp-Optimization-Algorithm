import numpy as np


def chaos(index, max_iter, Value):
    O = np.zeros(max_iter)
    x = np.zeros(max_iter + 1)
    x[0] = 0.7
    G = np.zeros(max_iter)

    if index == 1:  # Chebyshev map
        for i in range(max_iter):
            x[i + 1] = np.cos((i + 1) * np.arccos(x[i]))
            G[i] = ((x[i] + 1) * Value) / 2

    elif index == 2:  # Circle map
        a = 0.5
        b = 0.2
        for i in range(max_iter):
            x[i + 1] = (x[i] + b - (a / (2 * np.pi)) * np.sin(2 * np.pi * x[i])) % 1
            G[i] = x[i] * Value

    elif index == 3:  # Gauss/mouse map
        for i in range(max_iter):
            if x[i] == 0:
                x[i + 1] = 0
            else:
                x[i + 1] = 1 / x[i] % 1
            G[i] = x[i] * Value

    elif index == 4:  # Iterative map
        a = 0.7
        for i in range(max_iter):
            x[i + 1] = np.sin((a * np.pi) / x[i])
            G[i] = ((x[i] + 1) * Value) / 2

    elif index == 5:  # Logistic map
        a = 4
        for i in range(max_iter):
            x[i + 1] = a * x[i] * (1 - x[i])
            G[i] = x[i] * Value

    elif index == 6:  # Piecewise map
        P = 0.4
        for i in range(max_iter):
            if 0 <= x[i] < P:
                x[i + 1] = x[i] / P
            elif P <= x[i] < 0.5:
                x[i + 1] = (x[i] - P) / (0.5 - P)
            elif 0.5 <= x[i] < 1 - P:
                x[i + 1] = (1 - P - x[i]) / (0.5 - P)
            elif 1 - P <= x[i] < 1:
                x[i + 1] = (1 - x[i]) / P
            G[i] = x[i] * Value

    elif index == 7:  # Sine map
        for i in range(max_iter):
            x[i + 1] = np.sin(np.pi * x[i])
            G[i] = x[i] * Value

    elif index == 8:  # Singer map
        u = 1.07
        for i in range(max_iter):
            x[i + 1] = u * (7.86 * x[i] - 23.31 * (x[i] ** 2) + 28.75 * (x[i] ** 3) - 13.302875 * (x[i] ** 4))
            G[i] = x[i] * Value

    elif index == 9:  # Sinusoidal map
        for i in range(max_iter):
            x[i + 1] = 2.3 * (x[i] ** 2) * np.sin(np.pi * x[i])
            G[i] = x[i] * Value

    elif index == 10:  # Tent map
        x[0] = 0.6
        for i in range(max_iter):
            if x[i] < 0.7:
                x[i + 1] = x[i] / 0.7
            else:
                x[i + 1] = (10 / 3) * (1 - x[i])
            G[i] = x[i] * Value

    O = G
    return O





















def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = np.size(ub)
    # If the boundaries of all variables are equal and user enter a single number for both ub and lb
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    # If each variable has a different lb and ub
    elif Boundary_no > 1:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions



def Chimp(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    # initialize Attacker, Barrier, Chaser, and Driver
    Attacker_pos = np.zeros(dim)
    Attacker_score = np.inf  # change this to -np.inf for maximization problems
    Barrier_pos = np.zeros(dim)
    Barrier_score = np.inf  # change this to -np.inf for maximization problems
    Chaser_pos = np.zeros(dim)
    Chaser_score = np.inf  # change this to -np.inf for maximization problems
    Driver_pos = np.zeros(dim)
    Driver_score = np.inf  # change this to -np.inf for maximization problems

    # Initialize the positions of search agents
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter+1)
    l = 0  # Loop counter

    # Main loop
    while l < Max_iter:
        for i in range(Positions.shape[0]):
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

            # Calculate objective function for each search agent
            fitness = fobj(Positions[i, :])
            fitness = fitness['cost']

            # Update Attacker, Barrier, Chaser, and Driver
            if fitness < Attacker_score:
                Attacker_score = fitness  # Update Attacker
                Attacker_pos = Positions[i, :]

            if fitness > Attacker_score and fitness < Barrier_score:
                Barrier_score = fitness  # Update Barrier
                Barrier_pos = Positions[i, :]

            if fitness > Attacker_score and fitness > Barrier_score and fitness < Chaser_score:
                Chaser_score = fitness  # Update Chaser
                Chaser_pos = Positions[i, :]

            if fitness > Attacker_score and fitness > Barrier_score and fitness > Chaser_score and fitness > Driver_score:
                Driver_score = fitness  # Update Driver
                Driver_pos = Positions[i, :]

        f = 2 - l * ((2) / Max_iter)  # a decreases linearly from 2 to 0

        # The Dynamic Coefficient of f Vector as Table 1.

        # Group 1
        C1G1 = 1.95 - ((2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)))
        C2G1 = (2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)) + 0.5

        # Group 2
        C1G2 = 1.95 - ((2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)))
        C2G2 = (2 * (l ** 3) / (Max_iter ** 3)) + 0.5

        # Group 3
        C1G3 = (-2 * (l ** 3) / (Max_iter ** 3)) + 2.5
        C2G3 = (2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)) + 0.5

        # Group 4
        C1G4 = (-2 * (l ** 3) / (Max_iter ** 3)) + 2.5
        C2G4 = (2 * (l ** 3) / (Max_iter ** 3)) + 0.5

        # Update the Position of search agents including omegas
        for i in range(Positions.shape[0]):
            for j in range(Positions.shape[1]):
                # Please note that to choose a other groups you should use the related group strategies
                r11 = C1G1 * np.random.rand()  # r1 is a random number in [0,1]
                r12 = C2G1 * np.random.rand()  # r2 is a random number in [0,1]

                r21 = C1G2 * np.random.rand()  # r1 is a random number in [0,1]
                r22 = C2G2 * np.random.rand()  # r2 is a random number in [0,1]

                r31 = C1G3 * np.random.rand()  # r1 is a random number in [0,1]
                r32 = C2G3 * np.random.rand()  # r2 is a random number in [0,1]

                r41 = C1G4 * np.random.rand()  # r1 is a random number in [0,1]
                r42 = C2G4 * np.random.rand()  # r2 is a random number in [0,1]

                A1 = 2 * f * r11 - f  # Equation (3)
                C1 = 2 * r12  # Equation (4)

                # Please note that to choose various Chaotic maps you should use the related Chaotic maps strategies
                m = chaos(3, 1, 1)  # Equation (5)
                D_Attacker = abs(C1 * Attacker_pos[j] - m * Positions[i, j])  # Equation (6)
                X1 = Attacker_pos[j] - A1 * D_Attacker  # Equation (7)

                A2 = 2 * f * r21 - f  # Equation (3)
                C2 = 2 * r22  # Equation (4)

                D_Barrier = abs(C2 * Barrier_pos[j] - m * Positions[i, j])  # Equation (6)
                X2 = Barrier_pos[j] - A2 * D_Barrier  # Equation (7)

                A3 = 2 * f * r31 - f  # Equation (3)
                C3 = 2 * r32  # Equation (4)

                D_Driver = abs(C3 * Chaser_pos[j] - m * Positions[i, j])  # Equation (6)
                X3 = Chaser_pos[j] - A3 * D_Driver  # Equation (7)

                A4 = 2 * f * r41 - f  # Equation (3)
                C4 = 2 * r42  # Equation (4)

                D_Driver = abs(C4 * Driver_pos[j] - m * Positions[i, j])  # Equation (6)
                X4 = Chaser_pos[j] - A4 * D_Driver  # Equation (7)

                Positions[i, j] = (X1 + X2 + X3 + X4) / 4  # Equation (8)

        l += 1
        Convergence_curve[l] = Attacker_score
    result = fobj(Attacker_pos[0])

    return Attacker_score, Attacker_pos, Convergence_curve, result
