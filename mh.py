import sys
from yogi import read
import time
import math
import random


def write_output(penalty: int, solution: list[int], s: float) -> None:
    """
    Writes penalty of solution, how long it took to find and solution in file that was passed in terminal safely

    Args:
        penalty (int): Penalty of the solution
        solution (list[int]): Solution (ordered cars)
        s (float): Start time
    """

    try:
        with open(sys.argv[1], "w") as f:  # Opens first file passed in terminal
            f.write(
                f"{penalty} {time.time() - s:.1f}\n"
            )  # Writes penalty, time it took to find and solution
            f.write(" ".join(map(str, solution)) + "\n")
    except:
        print("Error writting in file")


def is_solution_picked(delta: int, T: float) -> bool:
    """
    Returns whether worse solution is picked

    Args:
        delta (int): penalty(new_solution) - penalty(prev_solution)
        T (float): Temperature

    Returns:
        bool: Whether worse solution is picked
    """
    p = math.exp(-(delta) / T)
    return True if random.random() < p else False


def calculate_penalty_diff(
    solution: list[int],
    i: int,
    j: int,
    classes: list[list[int]],
    M: int,
    ne: list[int],
    ce: list[int],
    C: int,
) -> int:
    """
    Returns the penalty difference between two solutions, bearing in mind new soltution is same as old solution permutating two cars

    Args:
        solution (list[int]): Solution (list of ordered cars)
        i (int): Car to switch
        j (int): Car to switch
        classes (list[list[int]]): List of classes
        M (int): Number of upgrades
        ne (list[int]): List of ne's
        ce (list[int]): List of ce's
        C (int): Number of cars

    Returns:
        int: Penalty difference between the solutions
    """

    delta = 0
    if i > j:
        i, j = j, i

    for m in range(M):
        win_len, win_max = ne[m], ce[m]

        # Range of windows affected by i [i, i + win_len - 1]
        # Range of windows affected by j [j, j + win_len - 1]
        # Windows reach to C + win_len - 1
        limit = C + win_len - 1
        if j < i + win_len:
            ranges = [(i, min(limit, j + win_len - 1))]
        else:
            ranges = [
                (i, i + win_len - 1),
                (j, min(limit, j + win_len - 1)),
            ]

        # Calculate penalty for new solution
        for start_k, end_k in ranges:
            current_sum = 0
            # Initialize window that ends in start_k
            for idx in range(max(0, start_k - win_len + 1), start_k + 1):
                if idx < C:
                    current_sum += classes[solution[idx]][m + 2]

            if current_sum > win_max:   # Add to delta new penalty
                delta += current_sum - win_max

            # Slide window to end_k
            for k in range(start_k + 1, end_k + 1):
                if k < C: # New car enters window
                    current_sum += classes[solution[k]][m + 2]
                if k - win_len >= 0: # Old car exits window
                    current_sum -= classes[solution[k - win_len]][m + 2]

                if current_sum > win_max:
                    delta += current_sum - win_max

        # Change to old solution to calculate penalty difference
        solution[i], solution[j] = solution[j], solution[i]

        # Calculate penalty for old solution
        for start_k, end_k in ranges:
            current_sum = 0
            for idx in range(max(0, start_k - win_len + 1), start_k + 1):
                if idx < C:
                    current_sum += classes[solution[idx]][m + 2]

            if current_sum > win_max:   # Remove from delta penalty
                delta -= current_sum - win_max

            for k in range(start_k + 1, end_k + 1):
                if k < C:
                    current_sum += classes[solution[k]][m + 2]
                if k - win_len >= 0:
                    current_sum -= classes[solution[k - win_len]][m + 2]

                if current_sum > win_max:
                    delta -= current_sum - win_max

        # Back to new solution
        solution[i], solution[j] = solution[j], solution[i]

    return delta



def calculate_penalty(
    solution: list[int], classes: list[list[int]], M: int, ne: list[int], ce: list[int]
) -> int:
    """Returns penalty of given solution

    Args:
        solution (list[int]): Solution (list of ordered cars)
        classes (list[list[int]]): List of classes
        M (int): Number of upgrades
        ne (list[int]): List of ne's
        ce (list[int]): List of ce's

    Returns:
        int: Penalty
    """

    global_penalty = 0

    for m in range(M):  # Iterate over 'millores'
        window_len, window_max = ne[m], ce[m]
        window_count = 0  # Store count for window for 'millora' m

        for i in range(len(solution) + window_len):
            if i - window_len >= 0:
                window_count -= classes[solution[i - window_len]][m + 2]

            if i < len(solution):
                window_count += classes[solution[i]][m + 2]

            if window_count > window_max:
                global_penalty += window_count - window_max

    return global_penalty


def generate_greedy_initial_solution(
    C: int,
    M: int,
    K: int,
    ce: list[int],
    ne: list[int],
    classes: list[list[int]],
    stock: list[int],
) -> list[int]:
    """
    Performs a greedy algorithm to find a reasonable solution to the problem
    Starts with an empty solution and adds the car that gives less penalty at each step. In case of tie, picks the one with more stock (to not run out of choices too soon)

    Args:
        C (int): Number of cars
        M (int): Number of upgrades
        K (int): Number of classes
        ce (list[int]): List of ce's
        ne (list[int]): List of ne's
        classes (list[list[int]]): List of classes
        stock (list[int]): Remaining stock of cars (separated by classes)

    Returns:
        list[int]: Solution
    """

    def update_current_penalty(
        solution: list[int],
        classes: list[list[int]],
        M: int,
        ne: list[int],
        ce: list[int],
        last_index: int,
        current_total: int,
    ) -> int:
        """
        Updates the current penalty when adding a car

        Args:
            solution (list[int]): Partial solution (list of ordered cars)
            classes (list[list[int]]): List of classes
            M (int): Number of upgrades
            ne (list[int]): List of ne's
            ce (list[int]): List of ce's
            last_index (int): Solution length
            current_total (int): Current penalty

        Returns:
            int: Updated penalty
        """
        extra_penalty = 0

        for m in range(M):
            window_len = ne[m]
            window_max = ce[m]

            # Define the start of the current window
            start = max(0, last_index - window_len + 1)

            count = 0
            for i in range(start, last_index + 1):
                if classes[solution[i]][m + 2] == 1:
                    count += 1

            if count > window_max:
                extra_penalty += count - window_max

        return current_total + extra_penalty

    solution = []
    total_penalty = 0

    for _ in range(C):
        best_class = -1
        min_new_penalty = 10**18 # Infinity
        max_remaining_stock = -1

        # For each car (class)
        for i in range(K):
            if stock[i] > 0:
                solution.append(i)

                # Simulates the penalty if adding a car
                penalty_with_i = update_current_penalty(
                    solution, classes, M, ne, ce, len(solution) - 1, total_penalty
                )

                # Selection criteria: 1. Les penalty, 2. More stock
                if (penalty_with_i < min_new_penalty) or (
                    penalty_with_i == min_new_penalty and stock[i] > max_remaining_stock
                ):
                    min_new_penalty = penalty_with_i
                    best_class = i
                    max_remaining_stock = stock[i]

                solution.pop()

        if best_class != -1:
            solution.append(best_class)
            stock[best_class] -= 1
            total_penalty = min_new_penalty

    return solution


def pick_random_neighbour(solution: list[int], C: int) -> tuple[int, int]:
    """
    Returns two random indexs such that solution[i] != solution[j]. Neighbours are those solutions you get by swapping two cars

    Args:
        solution (list[int]): Solution (ordered cars)
        C (int): Number of cars

    Returns:
        tuple[int,int]: Random indexes (i,j) such that solution[i] != solution[j]
    """

    i = random.randint(0, C - 1)
    j = random.randint(0, C - 1)
    while i == j or solution[i] == solution[j]: # There is a risk of staying in loop if all cars are the same!!
        j = random.randint(0, C - 1)
    return i, j


def mh(
    C: int,
    M: int,
    K: int,
    ce: list[int],
    ne: list[int],
    classes: list[list[int]],
    s: float,
    T0: float = 10,
    alpha: float = 0.9995,
) -> None:
    """
    Writes in output file best solution found using simulated annealing

    Args:
        C (int): Number of cars
        M (int): Number of upgrades
        K (int): Number of classes
        ce (list[int]): List of ce's
        ne (list[int]): List of ne's
        classes (list[list[int]]): List of classes
        s (float): Start Time
        T0 (float, optional): Temperature. Defaults to 5.
        alpha (float, optional): For neg exp. Defaults to 0.99.
    """

    stock_cotxes: list[int] = []
    for i in range(K):
        cantidad = classes[i][1]
        stock_cotxes.append(cantidad)
        
    # Initialize with greedy solution
    solution = generate_greedy_initial_solution(C, M, K, ce, ne, classes, stock_cotxes)
    s_penalty = calculate_penalty(solution, classes, M, ne, ce)
    best_penalty = s_penalty
    write_output(s_penalty, solution, s)

    T = T0
    k = 0

    while k < 100000:

        i, j = pick_random_neighbour(solution, C)
        solution[i], solution[j] = solution[j], solution[i]
        delta = calculate_penalty_diff(solution, i, j, classes, M, ne, ce, C)

        sol_accepted = False

        if delta < 0:
            sol_accepted = True
        elif delta > 0:
            if is_solution_picked(delta, T):    # Even if solution is worse might choose it
                sol_accepted = True

        if sol_accepted:
            s_penalty += delta
            if s_penalty < best_penalty:
                best_penalty = s_penalty
                write_output(best_penalty, solution, s) # Write new best solution
        else:
            solution[i], solution[j] = solution[j], solution[i]

        T = alpha * T
        k += 1


def main() -> None:
    """
    Main function to read user input
    """

    C, M, K = read(int), read(int), read(int)
    ce = [read(int) for _ in range(M)]
    ne = [read(int) for _ in range(M)]
    classes = [[read(int) for _ in range(M + 2)] for _ in range(K)]
    s = time.time()
    mh(C, M, K, ce, ne, classes, s)


if __name__ == "__main__":
    main()
