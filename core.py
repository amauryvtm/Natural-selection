"""
core.py — Moteur de simulation d'évolution sous sélection naturelle.

Classes :
    Individual  — porteur du génome et des traits phénotypiques
    Operators   — opérateurs génétiques (mutation, croisement, sélection, innovation)
    Population  — ensemble d'individus + dynamique temporelle
    simulate()  — point d'entrée unique : lit config.py, produit history.npz + stats.json
"""

from __future__ import annotations
from dataclasses import dataclass, field
from config import CONFIG
from typing import List
import json
import math
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "data" / "work"


def _load_config(*args, **kwargs) -> dict:
    return CONFIG


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """
    Un individu porteur d'un génome binaire.

    Attributes
    ----------
    genome      : tableau numpy de 0/1 de longueur genome_len
    trait_x     : phénotype A — moyenne des gènes de la première moitié
    trait_y     : phénotype B — moyenne des gènes de la seconde moitié
    fitness     : score de survie calculé par Operators.score()
    alive       : l'individu est-il encore vivant ?
    age         : nombre de générations vécues
    cognitive   : a-t-il déjà réalisé une innovation cognitive ?
    """

    genome: np.ndarray
    trait_x: float = 0.0
    trait_y: float = 0.0
    fitness: float = 0.0
    alive: bool = True
    age: int = 0
    cognitive: bool = False

    def __post_init__(self) -> None:
        self._compute_traits()

    def _compute_traits(self) -> None:
        half = len(self.genome) // 2
        self.trait_x = float(np.mean(self.genome[:half]))
        self.trait_y = float(np.mean(self.genome[half:]))

    def snapshot(self) -> dict:
        return {
            "trait_x": self.trait_x,
            "trait_y": self.trait_y,
            "fitness": self.fitness,
            "alive": self.alive,
            "age": self.age,
            "cognitive": self.cognitive,
        }


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class Operators:
    """
    Opérateurs génétiques stateless.

    Tous les paramètres proviennent de la config injectée à l'instanciation.
    """

    def __init__(self, cfg: dict) -> None:
        ops = cfg["operators"]
        sim = cfg["simulation"]
        self.mutation_rate: float = ops["mutation_rate"]
        self.crossover_rate: float = ops["crossover_rate"]
        self.cognitive_bonus: float = ops["cognitive_bonus"]
        self.fitness_fn: str = sim["fitness_fn"]
        self.death_threshold: float = sim["death_threshold"]

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def score(self, ind: Individual) -> float:
        """Calcule le fitness d'un individu selon la fonction configurée."""
        x, y = ind.trait_x, ind.trait_y
        base = self._fitness_fn(x, y)
        bonus = self.cognitive_bonus if ind.cognitive else 0.0
        return float(np.clip(base + bonus, 0.0, 1.0))

    def _fitness_fn(self, x: float, y: float) -> float:
        fn = self.fitness_fn
        if fn == "quadratic":
            return 1.0 - 2.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
        elif fn == "plateau":
            dist = math.hypot(x - 0.5, y - 0.5)
            return 1.0 if dist < 0.2 else max(0.0, 1.0 - dist * 3)
        elif fn == "rugged":
            return float(
                0.5 * math.sin(4 * math.pi * x) * math.cos(4 * math.pi * y) + 0.5
            )
        else:
            raise ValueError(f"fitness_fn inconnue : {fn!r}")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, ind: Individual) -> Individual:
        """Flip stochastique de bits selon mutation_rate."""
        genome = ind.genome.copy()
        mask = np.random.random(len(genome)) < self.mutation_rate
        genome[mask] = 1 - genome[mask]
        return Individual(genome=genome, age=0)

    # ------------------------------------------------------------------
    # Croisement
    # ------------------------------------------------------------------

    def crossover(self, parent_a: Individual, parent_b: Individual) -> Individual:
        """Croisement en un point avec probabilité crossover_rate."""
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent_a.genome) - 1)
            child_genome = np.concatenate(
                [parent_a.genome[:point], parent_b.genome[point:]]
            )
        else:
            child_genome = parent_a.genome.copy()
        return Individual(genome=child_genome)

    # ------------------------------------------------------------------
    # Sélection
    # ------------------------------------------------------------------

    def select(self, population: List[Individual], n: int) -> List[Individual]:
        """Sélection par tournoi binaire."""
        alive = [ind for ind in population if ind.alive]
        if not alive:
            return []
        selected = []
        for _ in range(n):
            a, b = random.choices(alive, k=2)
            selected.append(a if a.fitness >= b.fitness else b)
        return selected

    # ------------------------------------------------------------------
    # Innovation cognitive
    # ------------------------------------------------------------------

    def cognitive_innovate(self, ind: Individual) -> bool:
        """Un individu avec fitness élevé peut réaliser une innovation cognitive."""
        if not ind.cognitive and ind.fitness > 0.75:
            p = (ind.fitness - 0.75) * 0.4
            if random.random() < p:
                ind.cognitive = True
                return True
        return False

    # ------------------------------------------------------------------
    # Élimination
    # ------------------------------------------------------------------

    def apply_death(self, population: List[Individual]) -> int:
        """Tue les individus dont le fitness est en-dessous du seuil."""
        deaths = 0
        for ind in population:
            if ind.alive and ind.fitness < self.death_threshold:
                ind.alive = False
                deaths += 1
        return deaths


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class Population:
    """Gère l'ensemble des individus et pilote les transitions générationnelles."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.ops = Operators(cfg)
        pop_cfg = cfg["population"]
        self.target_size: int = pop_cfg["size"]
        self.genome_len: int = pop_cfg["genome_len"]
        self.individuals: List[Individual] = []

    def initialize(self) -> None:
        """Crée la population initiale aléatoire."""
        self.individuals = []
        for _ in range(self.target_size):
            genome = np.random.randint(0, 2, size=self.genome_len)
            ind = Individual(genome=genome)
            ind.fitness = self.ops.score(ind)
            self.individuals.append(ind)

    def step(self) -> dict:
        """Avance d'une génération."""
        # 1. Scoring
        for ind in self.individuals:
            if ind.alive:
                ind.fitness = self.ops.score(ind)
                ind.age += 1

        # 2. Innovation cognitive
        cognitive_events = sum(
            1 for ind in self.individuals
            if ind.alive and self.ops.cognitive_innovate(ind)
        )

        # 3. Mort
        deaths = self.ops.apply_death(self.individuals)

        # 4. Reproduction
        alive = [ind for ind in self.individuals if ind.alive]
        n_offspring = self.target_size - len(alive)
        if n_offspring > 0 and len(alive) >= 2:
            parents = self.ops.select(alive, n_offspring * 2)
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child = self.ops.crossover(parents[i], parents[i + 1])
                child = self.ops.mutate(child)
                child.fitness = self.ops.score(child)
                offspring.append(child)
            self.individuals = alive + offspring[:n_offspring]
        elif n_offspring > 0 and len(alive) == 1:
            offspring = []
            for _ in range(n_offspring):
                child = self.ops.mutate(alive[0])
                child.fitness = self.ops.score(child)
                offspring.append(child)
            self.individuals = alive + offspring
        else:
            self.individuals = alive

        return {
            "deaths": deaths,
            "cognitive_events": cognitive_events,
            "pop_size": len(self.individuals),
        }

    def snapshot(self) -> dict:
        """Sérialise l'état courant pour l'historique."""
        alive = [ind for ind in self.individuals if ind.alive]
        if not alive:
            return {
                "trait_x": [],
                "trait_y": [],
                "fitness": [],
                "pop_size": 0,
                "mean_fitness": 0.0,
                "diversity": 0.0,
            }
        trait_x = [ind.trait_x for ind in alive]
        trait_y = [ind.trait_y for ind in alive]
        fitnesses = [ind.fitness for ind in alive]
        diversity = float(np.std(trait_x) + np.std(trait_y)) / 2.0
        return {
            "trait_x": trait_x,
            "trait_y": trait_y,
            "fitness": fitnesses,
            "pop_size": len(alive),
            "mean_fitness": float(np.mean(fitnesses)),
            "diversity": diversity,
        }


# ---------------------------------------------------------------------------
# simulate()
# ---------------------------------------------------------------------------

def simulate() -> None:
    """
    Point d'entrée unique de la simulation.
    Lit CONFIG → initialise → boucle T générations →
    écrit data/work/history.npz + data/work/stats.json
    """
    cfg = _load_config()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    n_periods: int = cfg["simulation"]["n_periods"]

    pop = Population(cfg)
    pop.initialize()

    history: list[dict] = []
    stats = {
        "pop_size": [],
        "mean_fitness": [],
        "diversity": [],
        "cognitive_events": [],
    }

    print(f"[simulate] Démarrage : {n_periods} générations, pop={pop.target_size}")

    for t in range(n_periods):
        step_info = pop.step()
        snap = pop.snapshot()
        history.append(snap)

        stats["pop_size"].append(snap["pop_size"])
        stats["mean_fitness"].append(snap["mean_fitness"])
        stats["diversity"].append(snap["diversity"])
        stats["cognitive_events"].append(step_info["cognitive_events"])

        if (t + 1) % 10 == 0 or t == 0:
            print(
                f"  t={t+1:4d} | pop={snap['pop_size']:4d} "
                f"| fitness_moy={snap['mean_fitness']:.3f} "
                f"| diversité={snap['diversity']:.3f} "
                f"| cog={step_info['cognitive_events']}"
            )

        if snap["pop_size"] == 0:
            print("[simulate] Population éteinte.")
            break

    # Sérialisation history.npz
    max_t = len(history)
    max_n = max((len(h["trait_x"]) for h in history), default=0)

    arr_tx = np.full((max_t, max_n), np.nan)
    arr_ty = np.full((max_t, max_n), np.nan)
    arr_fit = np.full((max_t, max_n), np.nan)

    for t, snap in enumerate(history):
        n = len(snap["trait_x"])
        if n > 0:
            arr_tx[t, :n] = snap["trait_x"]
            arr_ty[t, :n] = snap["trait_y"]
            arr_fit[t, :n] = snap["fitness"]

    npz_path = WORK_DIR / "history.npz"
    np.savez_compressed(npz_path, trait_x=arr_tx, trait_y=arr_ty, fitness=arr_fit)
    print(f"[simulate] history.npz enregistré → {npz_path}")

    # Sérialisation stats.json
    json_path = WORK_DIR / "stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[simulate] stats.json enregistré  → {json_path}")
    print("[simulate] Terminé.")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    simulate()