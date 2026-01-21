import time
from yogi import read
import sys


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


def greedy(
    C: int,
    M: int,
    K: int,
    ce: list[int],
    ne: list[int],
    classes: list[list[int]],
    stock: list[int],
    s: float,
) -> tuple[list[int], int]:
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
        s (float): Start time

    Returns:
        tuple (list[int], int): Solution and penalty
    """
    solution: list[int] = []
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

    final_penalty = calculate_penalty(
        solution, classes, M, ne, ce
    )  # Final penalty calculated because update_penalty is fast and good enough for greedy decisions but may have a little error at the end

    write_output(final_penalty, solution, s)
    return solution, final_penalty


def main() -> None:
    """
    Main function to read user input
    """

    # Llegim dades del canal estandar d'entrada
    C, M, K = read(int), read(int), read(int)
    ce = [read(int) for _ in range(M)]
    ne = [read(int) for _ in range(M)]
    classes = [[read(int) for _ in range(M + 2)] for _ in range(K)]

    stock_cotxes: list[int] = []
    for i in range(K):
        cantidad = classes[i][1]
        stock_cotxes.append(cantidad)

    s = time.perf_counter()
    _, _ = greedy(C, M, K, ce, ne, classes, stock_cotxes, s)


if __name__ == "__main__":
    main()
