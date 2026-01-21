# AP3 Assignement: Car Sequencing Optimization

A system that optimizes the production sequence of cars to minimize penalty costs on the assembly line.

## Installation

This project requires the **Codon** compiler (for the Python scripts) and **Yogi** for reading input:

* **Codon:** Follow instructions at [exaloop/codon](https://github.com/exaloop/codon)
* **Yogi:** `pip install yogi` 

## Usage

First, compile the desired algorithm (e.g., the greedy approach):

```bash
codon build -release greedy.cc
```

Then run the executable with the input file and desired output file. The solution will be written in solution.txt:

```bash
./greedy solution.txt < input.txt
```

## How does it work?

### The Problem

We are helping a car brand improve efficiency. Cars have different improvements (GPS, Sunroof, etc.), and each improvement is installed at a specific station.

* **Constraints:** Each station has a constraint $(c_e, n_e)$, meaning in any sequence of $n_e$ cars, at most $c_e$ can require the improvement.
* **Penalties:** If a window of $n_e$ cars has $k$ requests where $k > c_e$, a penalty of $k - c_e$ is applied.

### Optimization Algorithms

We implemented three different approaches to solve this problem, contained in three specific files:

* **exh.cc** -> **Exhaustive Search.** This algorithm explores the entire search space to find the exact optimal ordering. It continually updates the output file whenever a better solution is found, ensuring that if the program is stopped early, the best valid solution found so far is saved.

* **greedy.cc** -> **Greedy Algorithm.** This implements a "greedy" strategy to construct a sequence step-by-step. It is designed to be extremely fast (under 0.5 seconds even for large inputs), though it does not guarantee an optimal solution.

* **mh.cc** -> **Metaheuristic.** This implements a local search strategy (such as Hill Climbing or Simulated Annealing). Like the exhaustive search, it overwrites the output file whenever it discovers a sequence with a lower total penalty cost.

## Authors

* Oscar Senesi Aladjem
* Alexander Cameron Hall

Universitat Politècnica de Catalunya, 2026