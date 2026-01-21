import sys
from yogi import read
import time

MAX_CAPACITY: list[list[int]]


def precalculate_capacity(M: int, C: int, ne: list[int], ce: list[int]) -> None:
    """
    Updates MAX_CAPACITY table by filling in maximum number of cars that can fit to complete solution without getting a penalty for all upgrades
    Aproximates how many cars can be added and calculates minimum penalty that will be added

    Args:
        M (int): Number of upgrades
        C (int): Number of cars
        ne (list[int]): List of ne's
        ce (list[int]): List of ce's
    """
    global MAX_CAPACITY  #  Stores maximum capacity for 'millora' m when we have k pending cars to not have penalty

    MAX_CAPACITY = [
        [0 for _ in range(C + 1)] for _ in range(M)
    ]  # Stores maximum capacity for 'millora' m when we have k pending cars to not have penalty
    max_capacity = 0

    for m in range(M):
        for k in range(C):
            num_full_windows = (
                k // ne[m]
            )  # Get number of full windows we can fit in remaining space
            remainder_slots = (
                k % ne[m]
            )  # Get number of remaining slots we can fit in remaining space
            max_capacity = (num_full_windows * ce[m]) + min(
                remainder_slots, ce[m]
            )  # Calculate max capacity to store cars without penalty
            MAX_CAPACITY[m][k] = max_capacity


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


def update_exiting_window_penalty(
    solution: list[int],
    classes: list[list[int]],
    M: int,
    ne: list[int],
    ce: list[int],
    best_cost: int,
    final_index: int,
    prev_cost: int,
) -> int:
    """
    Returns sum of current complete solution penalty adding the penalty of exiting window (not calculated before)
    Penalty is the sum of penalties for all m in M

    Args:
        solution (list[int]): Solution (ordered cars)
        classes (list[list[int]]): List of classes
        M (int): Number of upgrades
        ne (list[int]): List of ne's
        ce (list[int]): List of ce's
        best_cost (int): Best cost/penalty found yet
        final_index (int): Number of used cars
        prev_cost (int): Previous cost/penalty

    Returns:
        int: Current window penalty
    """

    global_penalty = prev_cost

    for m in range(M):  # Iterate over all 'millores'
        window_len, window_max = ne[m], ce[m]
        tail: list[int] = []  # Store 1 if car needs 'millora' or 0 otherwise

        start_index = max(0, final_index - window_len + 1)
        for i in range(
            final_index - 1, start_index - 1, -1
        ):  # Iterate backwards because pop() is O(1) while pop(0) is O(n)
            tail.append(classes[solution[i]][m + 2])

        current_sum = sum(tail)  # Number of 'millores'

        while len(tail) > 0:  # Simulate exiting window
            if current_sum > window_max:
                global_penalty += (
                    current_sum - window_max
                )  # Add to global_penalty if we surpass maximum

            if global_penalty >= best_cost:  # Prune
                return 10**18

            removed_car = tail.pop()  # Remove last car
            current_sum -= removed_car

    return global_penalty


def update_current_penalty(
    solution: list[int],
    classes: list[list[int]],
    M: int,
    ne: list[int],
    ce: list[int],
    best_cost: int,
    final_index: int,
    prev_cost: int,
) -> int:
    """
    Updates current partial solution penalty taking into account entering sliding window and full windows in solution (doesn't count exiting window)
    Penalty is the sum of penalties for all m in M

    Args:
        solution (list[int]): Solution (ordered cars)
        classes (list[list[int]]): List of classes
        M (int): Number of upgrades
        ne (list[int]): List of ne's
        ce (list[int]): List of ce's
        best_cost (int): Best cost/penalty found yet
        final_index (int): Number of used cars
        prev_cost (int): Previous cost/penalty

    Returns:
        int: Current penalty
    """

    global_penalty = prev_cost  # We add new possible penalty to prev_cost

    for m in range(M):  # Iterate over 'millores'
        window_len, window_max = ne[m], ce[m]
        window_count = 0  # Store count for window for 'millora' m

        for i in range(
            max(0, final_index - window_len + 1), final_index + 1
        ):  # Iterate over new window
            window_count += classes[solution[i]][m + 2]

        if window_count > window_max:
            global_penalty += (
                window_count - window_max
            )  # Add to global_penalty if we surpass maximum

        if global_penalty >= best_cost:  # Prune
            return 10**18

    return global_penalty


def get_lower_bound(
    curr_cost: int,
    C: int,
    M: int,
    best_cost: int,
    solution_lentgh: int,
    remaining_millores: list[int],
) -> int:
    """
    Returns a conservative lower bound by adding all minimum penalties (for all m in M) taking into accoutn all cars in pending (cars that have to be added to partial solution)
    Pending stores how many cars are yet to be added to solution from each class [class 0, class 1, ..., class C-1]

    Args:
        curr_cost (int): Current penalty
        C (int): Number of cars
        M (int): Number of upgrades
        best_cost (int): Best penalty found yet
        solution_lentgh (int): Lenght of the current partial solution
        remaining_millores (list[int]): How many cars in pending need each millora [millora 0, millora 1, ..., millora M-1]

    Returns:
        int: Lower bound
    """

    if curr_cost >= best_cost:  # Prune
        return 10**18

    global_penalty = curr_cost
    remaining_slots = C - solution_lentgh

    for m in range(M):  # Iterate over 'millores'

        needed_for_m = remaining_millores[m]
        max_capacity = MAX_CAPACITY[m][remaining_slots]

        if needed_for_m == 0:  # If no car needs 'millora' we skip
            continue

        if needed_for_m > max_capacity:
            global_penalty += (
                needed_for_m - max_capacity
            )  # Add to global_penalty if we surpass maximum
            if global_penalty >= best_cost:  # Prune
                return 10**18

    return global_penalty


def exh(
    partial_sol: list[int],
    best_cost: int,
    i: int,
    C: int,
    M: int,
    K: int,
    ce: list[int],
    ne: list[int],
    classes: list[list[int]],
    s: float,
    pending: list[int],
    curr_cost: int,
    remaining_millores: list[int],
) -> tuple[list[int], int]:
    """
    Returns list of current partial or complete solution and best cost found yet. 
    Uses recursivity to generate all combinations and prunes efficiently

    Args:
        partial_sol (list[int]): Partial solution
        best_cost (int): Best cost found yet
        i (int): Cars used
        C (int): Number of cars
        M (int): Number of millores
        K (int): Number of classes
        ce (list[int]): List of ce's
        ne (list[int]): List of ne's
        classes (list[list[int]]): List of classes
        s (float): Start Time
        pending (list[int]): How many cars of each class need to be added to partial solution
        curr_cost (int): Curent penalty
        remaining_millores (list[int]): How many cars in pending need each millora [millora 0, millora 1, ..., millora M-1]

    Returns:
        tuple[list[int], int]: Partial or complete solution with the best cost found yet
    """

    if curr_cost >= best_cost:  # Prune
        return partial_sol, best_cost

    if i == C:  # Base case
        final_penalty = update_exiting_window_penalty(  # Calculate final penalty taking into account exiting window for complete solution
            partial_sol, classes, M, ne, ce, best_cost, i, curr_cost
        )
        if final_penalty < best_cost:  # Check if complete solution is best solution
            best_sol = list(partial_sol)
            best_cost = final_penalty
            write_output(
                best_cost, best_sol, s
            )  # Write solution, cost and time in file
            return best_sol, best_cost

        return partial_sol, best_cost

    if (
        get_lower_bound(curr_cost, C, M, best_cost, i, remaining_millores) < best_cost
    ):  # Prune using lower bound

        candidates: list[tuple[int, float, int]] = (
            []
        )  # Stores ordered candidates that can be added to solution
        for j in range(K):
            if pending[j] > 0:
                partial_sol[i] = j
                next_cost = (
                    update_current_penalty(  # Calculate cost of adding certain car
                        partial_sol, classes, M, ne, ce, best_cost, i, curr_cost
                    )
                )

                if next_cost < best_cost:
                    # Calculate weight in case of draw
                    
                    weight: float = 0.0
                    for m in range(M):
                        if classes[j][m + 2]:
                            weight += (ne[m] / ce[m])   # Proportion of restriction
                    
                    candidates.append((next_cost, -weight, j))
        
        candidates.sort()   # Sort to get best candidates first
        
        for cost, _, j in candidates:
            if (
                cost >= best_cost
            ):  # Prune, since list is ordered all other candidates in list will be worse than this cost
                break

            pending[j] -= 1
            partial_sol[i] = j

            for m in range(M):
                if classes[j][m + 2]:
                    remaining_millores[m] -= 1

            _, best_cost = exh(  # Recursivity
                partial_sol,
                best_cost,
                i + 1,
                C,
                M,
                K,
                ce,
                ne,
                classes,
                s,
                pending,
                cost,
                remaining_millores,
            )
            pending[j] += 1

            for m in range(M):
                if classes[j][m + 2]:
                    remaining_millores[m] += 1

    return partial_sol, best_cost


def main() -> None:
    """
    Main function to read user input
    """

    C, M, K = read(int), read(int), read(int)
    ce = [read(int) for _ in range(M)]
    ne = [read(int) for _ in range(M)]
    classes = [[read(int) for _ in range(M + 2)] for _ in range(K)]

    precalculate_capacity(M, C, ne, ce)

    remaining_millores = [
        0
    ] * M  # Will store number of cars not in solution that require each 'millora'
    pending = [
        row[1] for row in classes
    ]  # Will store how many cars of each class need to be added to any partial solution

    for k in range(K):  # Fill in remaining_millores list
        count = pending[k]
        for m in range(M):
            if classes[k][m + 2]:
                remaining_millores[m] += count

    INF = 10**18
    s = time.time()

    _, _ = exh(
        [-1] * C, INF, 0, C, M, K, ce, ne, classes, s, pending, 0, remaining_millores
    )


if __name__ == "__main__":
    main()
