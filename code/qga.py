
"""
Quality Genetic Algorithm (QGA) — Reference Implementation

Features
--------
- Modular QGA class with clean APIs and runtime-adjustable parameters
- Pluggable cost function (minimization) with bounds per-dimension
- Error handling & input validation
- Type hints & docstrings
- Elitism, tournament selection (quality-weighted), SBX crossover, Gaussian mutation
- Early stopping on stagnation (optional)
- Example `main()` showing usage and CLI overrides

Usage (examples)
----------------
# Run with defaults (Sphere function in 5D)
python qga.py

# Override via CLI
python qga.py --dims 10 --pop 120 --iters 800 --seed 42 --func sphere

# Switch to Rastrigin with wider bounds and fewer iterations
python qga.py --func rastrigin --bounds -5.12 5.12 --iters 400

# Turn off early stopping
python qga.py --no-early-stop
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple, List

import numpy as np
np.set_printoptions(precision=6, suppress=True)


# ---------- Exceptions ----------

class QGAError(ValueError):
    """Base class for QGA-related errors."""


class InvalidParameterError(QGAError):
    """Raised when user passes an invalid parameter."""


class CostFunctionError(QGAError):
    """Raised when cost function fails or returns invalid values."""


# ---------- Utility Types ----------

Array = np.ndarray
Bounds = Iterable[Tuple[float, float]]


# ---------- Built-in Benchmark Cost Functions (minimize) ----------

def sphere(x: Array) -> float:
    """Sphere function: f(x) = sum(x_i^2), global min at 0."""
    return float(np.sum(np.square(x)))

def manhattan_norm(x: Array) -> float:
    """Manhattan norm: f(x) = sum(|x_i|), global min at 0."""
    return float(np.sum(np.abs(x)))

def quartic(x: Array) -> float:
    """Quartic function: f(x) = sum(x_i^4 - 16*x_i^2 + 5*x_i), global min at -39.16599..."""
    return float(np.sum(x**4 - 16*x**2 + 5*x))

def sinusoidal(x: Array) -> float:
    """Sinusoidal function: f(x) = sum(sin(x_i)^2), global min at 0."""
    return float(np.sum(np.sin(x)**2)) 

def composite_sine(x: Array) -> float:
    """Composite sine: f(x) = sum(|x_i| + 10*sin(x_i)), global min at -7.945..."""
    return float(np.sum(np.abs(x) + 10 * np.sin(x)))

def rastrigin(x: Array) -> float:
    """Rastrigin function (A=10), global min at 0."""
    A = 10.0
    return float(A * x.size + np.sum(x * x - A * np.cos(2 * math.pi * x)))

def sine_deviation(x: Array) -> float:
    """Sine and squared deviations: f(x) = sum(sin(x_i)^2 + 0.1*(x_i^2 - 1)^2), global min at 0.737..."""
    return float(np.sum(np.sin(x)**2 + 0.1 * (x**2 - 1)**2))

def exponential_squared(x: Array) -> float:
    """Exponential minus squared: f(x) = sum(exp(x_i) - x_i^2), global min at 1.535..."""
    return float(np.sum(np.exp(x) - x**2))

def sine_cosine(x: Array) -> float:
    """Custom function with sine and cosine: f(x) = sum((x_i - 1)^2 * (sin(x_i) + cos(x_i))), global min at 0."""
    return float(np.sum((x - 1)**2 * (np.sin(x) + np.cos(x))))

def sum_square(x: Array) -> float:
    """Sum Square Function: f(x) = sum(i * x_i^2), global min at 0."""
    return float(np.sum(np.arange(1, x.size + 1) * x**2))

def dixon(x: Array) -> float:
    """Dixon Function: f(x) = (x1 - 1)^2 + sum(i * (2*x_i^2 - x_{i-1} - 1)^2), global min at 0."""
    if x.size < 2:
        raise CostFunctionError("Dixon function requires at least 2 dimensions.")
    term1 = (x[0] - 1)**2
    term2 = np.sum(np.arange(2, x.size + 1) * (2 * x[1:]**2 - x[:-1] - 1)**2)
    return float(term1 + term2)

def zakharov(x: Array) -> float:
    """Zakharov Function: f(x) = sum(x_i^2) + (0.5*sum(i*x_i))^2 + (0.5*sum(i*x_i))^4, global min at 0."""
    i = np.arange(1, x.size + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return float(sum1 + sum2**2 + sum2**4)

BUILTIN_FUNCS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "manhattan": manhattan_norm,
    "quartic": quartic,
    "sinusoidal": sinusoidal,
    "composite_sine": composite_sine,
    "sine_deviation": sine_deviation,
    "exponential_squared": exponential_squared,
    "sine_cosine": sine_cosine,
    "sum_square": sum_square,
    "dixon": dixon,
    "zakharov": zakharov,
}


# ---------- Config Dataclass ----------

@dataclass
class QGAConfig:
    dims: int = 2
    population_size: int = 100
    iterations: int = 10000
    bounds: Tuple[Tuple[float, float], ...] = field(default_factory=lambda: tuple([(-10.0, 10.0)] * 2))
    mutation_rate: float = 0.15                # Probability a gene mutates
    mutation_sigma: float = 0.1                # Stddev for Gaussian noise
    crossover_rate: float = 0.9                # Probability to crossover
    tournament_k: int = 3                      # Tournament size
    elite_fraction: float = 0.05               # Fraction of elite survivors
    early_stopping_patience: int = 50          # Stop if no improvement over this many iters
    seed: Optional[int] = None                 # RNG seed for reproducibility

    # SBX (Simulated Binary Crossover) parameter
    sbx_eta: float = 15.0

    def validate(self) -> None:
        """Validate configuration values and raise helpful errors."""
        if self.dims <= 0:
            raise InvalidParameterError("dims must be a positive integer.")
        if self.population_size < 4:
            raise InvalidParameterError("population_size must be >= 4.")
        if self.iterations <= 0:
            raise InvalidParameterError("iterations must be a positive integer.")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise InvalidParameterError("mutation_rate must be in [0, 1].")
        if self.mutation_sigma <= 0.0:
            raise InvalidParameterError("mutation_sigma must be > 0.")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise InvalidParameterError("crossover_rate must be in [0, 1].")
        if self.tournament_k <= 0:
            raise InvalidParameterError("tournament_k must be >= 1.")
        if not (0.0 <= self.elite_fraction < 0.5):
            # Restrict elitism fraction to avoid premature convergence
            raise InvalidParameterError("elite_fraction must be in [0, 0.5).")
        if self.early_stopping_patience < 0:
            raise InvalidParameterError("early_stopping_patience must be >= 0.")
        if self.sbx_eta <= 0.0:
            raise InvalidParameterError("sbx_eta must be > 0.")

        # Validate bounds
        try:
            b = tuple(self.bounds)
        except Exception as e:
            raise InvalidParameterError(f"bounds must be an iterable of (low, high): {e}")

        if len(b) not in (1, self.dims):
            raise InvalidParameterError(
                f"bounds length must be 1 or dims ({self.dims}); got {len(b)}"
            )
        for lo, hi in b:
            if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
                raise InvalidParameterError("Each bound pair must be numeric (low, high).")
            if lo >= hi:
                raise InvalidParameterError(f"Each bound must satisfy low < high; found ({lo}, {hi}).")


# ---------- QGA Core ----------

class QGA:
    """
    Quality Genetic Algorithm (QGA)

    This GA variant derives parent selection probabilities from a "quality" score:
        quality = (max_cost - cost) / (max_cost - min_cost + eps)
    (i.e., lower cost -> higher quality), refreshed each generation.

    By default:
      - Minimizes the given cost function
      - Uses tournament selection (quality-weighted), SBX crossover, Gaussian mutation
      - Supports elitism and optional early stopping on stagnation

    Public API
    ----------
    - set_cost_function(func)
    - set_dimensions(d)
    - set_population_size(n)
    - set_iterations(t)
    - set_bounds(bounds)
    - set_seed(seed)
    - set_hyperparams(...)
    - fit() -> dict with best solution & history
    """

    def __init__(self, cost_fn: Callable[[Array], float], config: QGAConfig):
        if not callable(cost_fn):
            raise InvalidParameterError("cost_fn must be callable.")
        self.cost_fn = cost_fn
        self.config = config
        self.config.validate()
        self._rng = np.random.default_rng(self.config.seed)

        # Internals
        self._bounds = self._normalize_bounds(self.config.bounds, self.config.dims)
        self._pop: Array = np.empty((0, self.config.dims))
        self._costs: Array = np.empty((0,))
        self.history: List[float] = []  # best cost per generation

    # ----- Runtime Adjusters -----

    def set_cost_function(self, func: Callable[[Array], float]) -> "QGA":
        if not callable(func):
            raise InvalidParameterError("func must be callable.")
        self.cost_fn = func
        return self

    def set_dimensions(self, dims: int) -> "QGA":
        self.config.dims = int(dims)
        self.config.validate()
        self._bounds = self._normalize_bounds(self.config.bounds, self.config.dims)
        return self

    def set_population_size(self, n: int) -> "QGA":
        self.config.population_size = int(n)
        self.config.validate()
        return self

    def set_iterations(self, iters: int) -> "QGA":
        self.config.iterations = int(iters)
        self.config.validate()
        return self

    def set_bounds(self, bounds: Bounds) -> "QGA":
        self.config.bounds = tuple(bounds)
        self.config.validate()
        self._bounds = self._normalize_bounds(self.config.bounds, self.config.dims)
        return self

    def set_seed(self, seed: Optional[int]) -> "QGA":
        self.config.seed = seed
        self._rng = np.random.default_rng(seed)
        return self

    def set_hyperparams(
        self,
        *,
        mutation_rate: Optional[float] = None,
        mutation_sigma: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        tournament_k: Optional[int] = None,
        elite_fraction: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        sbx_eta: Optional[float] = None,
    ) -> "QGA":
        if mutation_rate is not None:
            self.config.mutation_rate = float(mutation_rate)
        if mutation_sigma is not None:
            self.config.mutation_sigma = float(mutation_sigma)
        if crossover_rate is not None:
            self.config.crossover_rate = float(crossover_rate)
        if tournament_k is not None:
            self.config.tournament_k = int(tournament_k)
        if elite_fraction is not None:
            self.config.elite_fraction = float(elite_fraction)
        if early_stopping_patience is not None:
            self.config.early_stopping_patience = int(early_stopping_patience)
        if sbx_eta is not None:
            self.config.sbx_eta = float(sbx_eta)
        self.config.validate()
        return self

    # ----- Core Methods -----

    def fit(self, verbose: bool = True, early_stopping: bool = True) -> dict:
        """
        Run the GA and return a dict containing:
            - best_x: np.ndarray  (best solution)
            - best_cost: float
            - history: list[float] (best cost per generation)
            - config: QGAConfig
        """
        cfg = self.config
        rng = self._rng

        # Initialize population uniformly within bounds
        self._pop = self._init_population()
        self._costs = self._evaluate_population(self._pop)
        best_cost = float(np.min(self._costs))
        best_x = self._pop[np.argmin(self._costs)].copy()
        self.history = [best_cost]
        best_iter = 0

        if verbose:
            print(f"[QGA] Start: best_cost={best_cost:.6f}")

        elite_count = max(1, int(round(cfg.elite_fraction * cfg.population_size)))

        for it in range(1, cfg.iterations + 1):
            # Elitism
            elite_idx = np.argsort(self._costs)[:elite_count]
            elites = self._pop[elite_idx]

            # Selection (parents)
            parents = self._select_parents(self._pop, self._costs, cfg.population_size - elite_count)

            # Crossover
            offspring = self._crossover(parents, cfg.crossover_rate)

            # Mutation
            offspring = self._mutate(offspring, cfg.mutation_rate, cfg.mutation_sigma)

            # Repair to bounds
            offspring = self._clip_to_bounds(offspring)

            # Form next generation
            self._pop = np.vstack([elites, offspring])
            self._costs = self._evaluate_population(self._pop)

            # Track best
            gen_best_idx = int(np.argmin(self._costs))
            gen_best_cost = float(self._costs[gen_best_idx])
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_x = self._pop[gen_best_idx].copy()
                best_iter = it

            self.history.append(best_cost)

            # Optional progress print
            if verbose and (it % max(1, cfg.iterations // 10) == 0 or it == 1):
                print(f"[QGA] Iter {it:4d}/{cfg.iterations} | best_cost={best_cost:.6f}")

            # Early stopping
            if early_stopping and cfg.early_stopping_patience > 0:
                if it - best_iter >= cfg.early_stopping_patience:
                    if verbose:
                        print(f"[QGA] Early stopped at iter {it} (no improvement in "
                              f"{cfg.early_stopping_patience} iterations).")
                    break

        return {
            "best_x": best_x,
            "best_cost": best_cost,
            "history": self.history,
            "config": cfg,
        }

    # ----- Internals -----

    @staticmethod
    def _normalize_bounds(bounds: Bounds, dims: int) -> Array:
        """Expand single (lo,hi) pair to per-dimension bounds; return as (dims, 2) array."""
        b = tuple(bounds)
        if len(b) == 1:
            b = b * dims
        out = np.asarray(b, dtype=float)
        if out.shape != (dims, 2):
            raise InvalidParameterError(f"bounds should shape to ({dims}, 2); got {out.shape}")
        return out

    def _init_population(self) -> Array:
        """Uniformly sample initial population within bounds."""
        low = self._bounds[:, 0]
        high = self._bounds[:, 1]
        return self._rng.uniform(low, high, size=(self.config.population_size, self.config.dims))

    def _evaluate_population(self, pop: Array) -> Array:
        """Safely evaluate cost function over population; return 1D array of costs."""
        costs = np.empty(pop.shape[0], dtype=float)
        for i, x in enumerate(pop):
            try:
                c = float(self.cost_fn(np.asarray(x, dtype=float)))
            except Exception as e:
                raise CostFunctionError(f"Cost function raised an error at index {i}: {e}")
            if not np.isfinite(c):
                raise CostFunctionError(f"Cost function returned non-finite value at index {i}: {c}")
            costs[i] = c
        return costs

    def _quality_scores(self, costs: Array) -> Array:
        """Convert costs to quality scores in [0,1], higher is better."""
        cmin = float(np.min(costs))
        cmax = float(np.max(costs))
        eps = 1e-12
        # Normalize so best (min cost) -> 1, worst -> 0
        qual = (cmax - costs) / (max(eps, cmax - cmin))
        return qual

    def _select_parents(self, pop: Array, costs: Array, n_needed: int) -> Array:
        """
        Tournament selection with quality weighting.
        Higher-quality individuals more likely to win tournaments.
        """
        k = self.config.tournament_k
        qual = self._quality_scores(costs) + 1e-12  # avoid all-zero
        probs = qual / np.sum(qual)

        idx = np.arange(pop.shape[0])
        parents = []
        for _ in range(n_needed):
            # sample k contestants weighted by quality
            contestants = self._rng.choice(idx, size=k, replace=False, p=probs)
            # winner is the min-cost contestant (strict minimization)
            winner = contestants[np.argmin(costs[contestants])]
            parents.append(pop[winner])
        return np.asarray(parents)

    def _crossover(self, parents: Array, crossover_rate: float) -> Array:
        """
        Simulated Binary Crossover (SBX) pairwise. If odd, last parent is copied.
        """
        rng = self._rng
        n, d = parents.shape
        offspring = np.empty_like(parents)
        eta = self.config.sbx_eta

        def sbx_pair(p1: Array, p2: Array) -> Tuple[Array, Array]:
            u = rng.random(d)
            beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta + 1.0)),
                            (1 / (2 * (1 - u))) ** (1.0 / (eta + 1.0)))
            c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            return c1, c2

        i = 0
        while i + 1 < n:
            p1, p2 = parents[i], parents[i + 1]
            if rng.random() < crossover_rate:
                c1, c2 = sbx_pair(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring[i], offspring[i + 1] = c1, c2
            i += 2

        if n % 2 == 1:
            offspring[-1] = parents[-1].copy()

        return offspring

    def _mutate(self, pop: Array, rate: float, sigma: float) -> Array:
        """
        Gaussian mutation applied gene-wise with probability `rate`.
        """
        rng = self._rng
        n, d = pop.shape
        mask = rng.random((n, d)) < rate
        noise = rng.normal(loc=0.0, scale=sigma, size=(n, d))
        mutated = pop.copy()
        mutated[mask] += noise[mask]
        return mutated

    def _clip_to_bounds(self, pop: Array) -> Array:
        """Project solutions into their bounds (simple clipping)."""
        low = self._bounds[:, 0]
        high = self._bounds[:, 1]
        return np.clip(pop, low, high)


# ---------- Main / Demo ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality Genetic Algorithm (QGA) demo")
    p.add_argument("--dims", type=int, default=5, help="Dimensionality of the search space")
    p.add_argument("--pop", type=int, default=100, help="Population size")
    p.add_argument("--iters", type=int, default=500, help="Max iterations")
    p.add_argument("--bounds", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   default=None, help="Uniform bounds for all dimensions, e.g., --bounds -5 5")
    p.add_argument("--func", type=str, default="sphere", choices=tuple(BUILTIN_FUNCS.keys()),
                   help="Built-in cost function to minimize")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    p.add_argument("--verbose", action="store_true", help="Verbose training logs")

    # Hyperparameters
    p.add_argument("--mutation-rate", type=float, default=0.15, help="Gene mutation probability")
    p.add_argument("--mutation-sigma", type=float, default=0.1, help="Stddev of Gaussian mutation")
    p.add_argument("--crossover-rate", type=float, default=0.9, help="Crossover probability")
    p.add_argument("--tournament-k", type=int, default=3, help="Tournament size")
    p.add_argument("--elite-frac", type=float, default=0.05, help="Elite fraction [0, 0.5)")
    p.add_argument("--sbx-eta", type=float, default=15.0, help="SBX distribution index (higher -> offspring closer to parents)")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (iters without improvement)")

    return p.parse_args(argv)


def build_bounds(dims: int, bounds_arg: Optional[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    if bounds_arg is None:
        return tuple([(-5.0, 5.0)] * dims)
    lo, hi = bounds_arg
    if lo >= hi:
        raise InvalidParameterError(f"--bounds requires LOW < HIGH, got ({lo}, {hi}).")
    return tuple([(lo, hi)] * dims)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)

        # Create config
        cfg = QGAConfig(
            dims=args.dims,
            population_size=args.pop,
            iterations=args.iters,
            bounds=build_bounds(args.dims, args.bounds),
            mutation_rate=args.mutation_rate,
            mutation_sigma=args.mutation_sigma,
            crossover_rate=args.crossover_rate,
            tournament_k=args.tournament_k,
            elite_fraction=args.elite_frac,
            early_stopping_patience=args.patience,
            seed=args.seed,
            sbx_eta=args.sbx_eta,
        )

        # Choose a cost function (you can also define your own and set via set_cost_function)
        cost_fn = BUILTIN_FUNCS[args.func]

        # Instantiate QGA
        qga = QGA(cost_fn=cost_fn, config=cfg)

        # --- Example of runtime adjustments (optional) ---
        # You can uncomment these lines to tweak parameters programmatically at runtime:
        # qga.set_dimensions(10).set_population_size(150).set_iterations(600)
        # qga.set_bounds([(-5.12, 5.12)] * 10)
        # qga.set_hyperparams(mutation_rate=0.2, tournament_k=4, elite_fraction=0.08)

        # Fit
        result = qga.fit(verbose=args.verbose, early_stopping=not args.no_early_stop)

        best_x = result["best_x"]
        best_cost = result["best_cost"]
        print("\n=== QGA Result ===")
        print(f"Best cost: {best_cost:.6f}")
        print(f"Best x   : {best_x}")

        # Optional: show a tiny summary of convergence
        hist = np.asarray(result["history"], dtype=float)
        print(f"Iterations performed: {hist.size - 1}")
        print(f"Initial -> Final best: {hist[0]:.6f} -> {hist[-1]:.6f}")
        return 0

    except (InvalidParameterError, CostFunctionError) as e:
        print(f"[Input/Config Error] {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[Interrupted] Exiting cleanly.", file=sys.stderr)
        return 130
    except Exception as e:
        # Catch-all with minimal leakage; useful for unexpected issues
        print(f"[Unexpected Error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
