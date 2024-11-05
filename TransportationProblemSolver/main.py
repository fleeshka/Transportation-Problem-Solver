import numpy as np

def check_balance(supply, demand):
    """Check if the problem is balanced."""
    total_supply = sum(supply)
    total_demand = sum(demand)
    if total_supply != total_demand:
        print("The problem is not balanced!")
        return False
    return True

def print_table(cost_matrix, supply, demand):
    """Print input parameter table."""
    print("Cost Matrix (C):")
    print(cost_matrix)
    print("Supply (S):", supply)
    print("Demand (D):", demand)

def north_west_corner(supply, demand):
    """North-West Corner Method for finding initial feasible solution."""
    m, n = len(supply), len(demand)
    x = np.zeros((m, n))
    i, j = 0, 0
    while i < m and j < n:
        allocation = min(supply[i], demand[j])
        x[i, j] = allocation
        supply[i] -= allocation
        demand[j] -= allocation
        if supply[i] == 0:
            i += 1
        if demand[j] == 0:
            j += 1
    return x

def vogel_approximation(supply, demand, cost_matrix):
    """Vogel's Approximation Method for finding initial feasible solution."""
    m, n = len(supply), len(demand)
    x = np.zeros((m, n))
    cost_matrix = cost_matrix.astype(float)  # Convert cost matrix to float to use np.inf

    while any(supply) and any(demand):
        # Calculate penalties for rows
        penalty_rows = []
        for row in cost_matrix:
            finite_values = sorted([val for val in row if val != np.inf])
            if len(finite_values) > 1:
                penalty_rows.append(finite_values[1] - finite_values[0])
            elif len(finite_values) == 1:
                penalty_rows.append(finite_values[0])
            else:
                penalty_rows.append(0)

        # Calculate penalties for columns
        penalty_cols = []
        for col in cost_matrix.T:
            finite_values = sorted([val for val in col if val != np.inf])
            if len(finite_values) > 1:
                penalty_cols.append(finite_values[1] - finite_values[0])
            elif len(finite_values) == 1:
                penalty_cols.append(finite_values[0])
            else:
                penalty_cols.append(0)

        # Determine row or column with the highest penalty
        row_idx = np.argmax(penalty_rows)
        col_idx = np.argmax(penalty_cols)
        if penalty_rows[row_idx] >= penalty_cols[col_idx]:
            i = row_idx
            j = np.argmin(cost_matrix[i])
        else:
            j = col_idx
            i = np.argmin(cost_matrix[:, j])
        
        # Allocate to the chosen cell
        allocation = min(supply[i], demand[j])
        x[i, j] = allocation
        supply[i] -= allocation
        demand[j] -= allocation
        cost_matrix[i, j] = np.inf  # Mark the cell as allocated

    return x

def russell_approximation(supply, demand, cost_matrix):
    """Russell's Approximation Method for finding initial feasible solution."""
    m, n = len(supply), len(demand)
    x = np.zeros((m, n))
    u = np.zeros(m)  # Row potentials
    v = np.zeros(n)  # Column potentials
    cost_matrix = cost_matrix.astype(float)  # Convert cost matrix to float

    for i in range(m):
        u[i] = min(cost_matrix[i])
    for j in range(n):
        v[j] = min(cost_matrix[:, j])

    for i in range(m):
        for j in range(n):
            cost_adjusted = cost_matrix[i, j] - u[i] - v[j]
            if cost_adjusted < 0:
                allocation = min(supply[i], demand[j])
                x[i, j] = allocation
                supply[i] -= allocation
                demand[j] -= allocation
    return x

# Main function to solve the transportation problem
def transportation_problem(supply, demand, cost_matrix):
    # Convert cost_matrix to float for compatibility with np.inf
    cost_matrix = cost_matrix.astype(float)
    
    # Check if the problem is balanced
    if not check_balance(supply, demand):
        return
    
    print("The input parameter table:")
    print_table(cost_matrix, supply, demand)
    
    # North-West Corner Method
    nw_corner_solution = north_west_corner(supply.copy(), demand.copy())
    print("\nInitial Basic Feasible Solution using North-West Corner Method:")
    print(nw_corner_solution)
    
    # Vogel's Approximation Method
    vogel_solution = vogel_approximation(supply.copy(), demand.copy(), cost_matrix.copy())
    print("\nInitial Basic Feasible Solution using Vogel's Approximation Method:")
    print(vogel_solution)
    
    # Russell's Approximation Method
    russell_solution = russell_approximation(supply.copy(), demand.copy(), cost_matrix.copy())
    print("\nInitial Basic Feasible Solution using Russell's Approximation Method:")
    print(russell_solution)

# Example input
supply = np.array([20, 30, 25])
demand = np.array([10, 35, 30])
cost_matrix = np.array([[8, 6, 10],
                        [9, 12, 13],
                        [14, 9, 16]])

# Solve the transportation problem
transportation_problem(supply, demand, cost_matrix)
