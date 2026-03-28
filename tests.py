"""
tests.py — Suite de tests pytest pour evo4d.

Couvre :
    - Individual : création, traits, snapshot
    - Operators  : score (3 fonctions), mutation, croisement, sélection,
                   innovation cognitive, mort
    - Population : initialisation, step, snapshot, extinction
    - simulate() : exécution complète + fichiers de sortie
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Ajout du répertoire racine au path pour les imports ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core import Individual, Operators, Population, simulate, _load_config

# ── Config minimale pour les tests ──────────────────────────────────────────
MINIMAL_CFG = {
    "population": {"size": 20, "genome_len": 16},
    "simulation": {
        "n_periods": 5,
        "death_threshold": 0.05,
        "fitness_fn": "quadratic",
    },
    "operators": {
        "mutation_rate": 0.01,
        "crossover_rate": 0.7,
        "cognitive_bonus": 0.15,
    },
}


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def genome_half():
    """Génome 50/50 (tous les bits à 1 sur la 1ère moitié, 0 sur la 2ème)."""
    g = np.zeros(16, dtype=int)
    g[:8] = 1
    return g


@pytest.fixture
def ind_center():
    """Individu centré : trait_x ≈ 0.5, trait_y ≈ 0.5."""
    g = np.array([1, 0] * 8, dtype=int)  # alternance → moyenne 0.5
    return Individual(genome=g)


@pytest.fixture
def ops():
    return Operators(MINIMAL_CFG)


@pytest.fixture
def pop():
    p = Population(MINIMAL_CFG)
    p.initialize()
    return p


# ===========================================================================
# Individual
# ===========================================================================

class TestIndividual:

    def test_traits_computed_on_init(self, genome_half):
        ind = Individual(genome=genome_half)
        # Première moitié = 1 → trait_x = 1.0 ; seconde moitié = 0 → trait_y = 0.0
        assert ind.trait_x == pytest.approx(1.0)
        assert ind.trait_y == pytest.approx(0.0)

    def test_traits_range(self):
        for _ in range(50):
            g = np.random.randint(0, 2, 16)
            ind = Individual(genome=g)
            assert 0.0 <= ind.trait_x <= 1.0
            assert 0.0 <= ind.trait_y <= 1.0

    def test_default_values(self):
        g = np.zeros(8, dtype=int)
        ind = Individual(genome=g)
        assert ind.alive is True
        assert ind.age == 0
        assert ind.cognitive is False
        assert ind.fitness == pytest.approx(0.0)

    def test_snapshot_keys(self, ind_center):
        snap = ind_center.snapshot()
        assert set(snap.keys()) == {"trait_x", "trait_y", "fitness", "alive", "age", "cognitive"}

    def test_snapshot_values_match_attributes(self, ind_center):
        ind_center.fitness = 0.77
        ind_center.age = 3
        snap = ind_center.snapshot()
        assert snap["fitness"] == pytest.approx(0.77)
        assert snap["age"] == 3


# ===========================================================================
# Operators — score
# ===========================================================================

class TestOperatorsScore:

    def test_quadratic_center_is_best(self, ind_center, ops):
        score = ops.score(ind_center)
        # Centre ≈ 0.5,0.5 → fitness ≈ 1.0
        assert score > 0.9

    def test_quadratic_corner_is_low(self, ops):
        g = np.zeros(16, dtype=int)
        ind = Individual(genome=g)  # trait_x=0, trait_y=0
        score = ops.score(ind)
        assert score < 0.6

    def test_score_in_range(self, ops):
        for _ in range(100):
            g = np.random.randint(0, 2, 16)
            ind = Individual(genome=g)
            s = ops.score(ind)
            assert 0.0 <= s <= 1.0

    def test_plateau_fitness_fn(self):
        cfg = {**MINIMAL_CFG, "simulation": {**MINIMAL_CFG["simulation"], "fitness_fn": "plateau"}}
        ops_p = Operators(cfg)
        g = np.array([1, 0] * 8, dtype=int)
        ind = Individual(genome=g)
        s = ops_p.score(ind)
        assert 0.0 <= s <= 1.0

    def test_rugged_fitness_fn(self):
        cfg = {**MINIMAL_CFG, "simulation": {**MINIMAL_CFG["simulation"], "fitness_fn": "rugged"}}
        ops_r = Operators(cfg)
        g = np.random.randint(0, 2, 16)
        ind = Individual(genome=g)
        s = ops_r.score(ind)
        assert 0.0 <= s <= 1.0

    def test_invalid_fitness_fn_raises(self):
        cfg = {**MINIMAL_CFG, "simulation": {**MINIMAL_CFG["simulation"], "fitness_fn": "unknown"}}
        ops_bad = Operators(cfg)
        g = np.zeros(16, dtype=int)
        ind = Individual(genome=g)
        with pytest.raises(ValueError, match="fitness_fn inconnue"):
            ops_bad.score(ind)

    def test_cognitive_bonus_applied(self, ops):
        g = np.array([1, 0] * 8, dtype=int)
        ind_no_cog = Individual(genome=g)
        ind_cog = Individual(genome=g.copy())
        ind_cog.cognitive = True
        s_no = ops.score(ind_no_cog)
        s_yes = ops.score(ind_cog)
        assert s_yes >= s_no


# ===========================================================================
# Operators — mutation
# ===========================================================================

class TestMutation:

    def test_mutate_returns_new_individual(self, ops):
        g = np.zeros(16, dtype=int)
        ind = Individual(genome=g)
        child = ops.mutate(ind)
        assert child is not ind

    def test_mutate_genome_length_preserved(self, ops):
        g = np.random.randint(0, 2, 16)
        ind = Individual(genome=g)
        child = ops.mutate(ind)
        assert len(child.genome) == 16

    def test_mutate_high_rate_changes_genome(self):
        cfg = {**MINIMAL_CFG, "operators": {**MINIMAL_CFG["operators"], "mutation_rate": 0.99}}
        ops_high = Operators(cfg)
        g = np.zeros(16, dtype=int)
        ind = Individual(genome=g)
        child = ops_high.mutate(ind)
        # Avec un taux de 99 %, presque tous les bits devraient avoir flippé
        assert np.sum(child.genome) > 10

    def test_mutate_zero_rate_preserves_genome(self):
        cfg = {**MINIMAL_CFG, "operators": {**MINIMAL_CFG["operators"], "mutation_rate": 0.0}}
        ops_zero = Operators(cfg)
        g = np.ones(16, dtype=int)
        ind = Individual(genome=g)
        child = ops_zero.mutate(ind)
        np.testing.assert_array_equal(child.genome, g)


# ===========================================================================
# Operators — croisement
# ===========================================================================

class TestCrossover:

    def test_crossover_child_length(self, ops):
        ga = np.ones(16, dtype=int)
        gb = np.zeros(16, dtype=int)
        a = Individual(genome=ga)
        b = Individual(genome=gb)
        child = ops.crossover(a, b)
        assert len(child.genome) == 16

    def test_crossover_child_values_come_from_parents(self, ops):
        ga = np.ones(16, dtype=int)
        gb = np.zeros(16, dtype=int)
        a = Individual(genome=ga)
        b = Individual(genome=gb)
        child = ops.crossover(a, b)
        assert all(bit in (0, 1) for bit in child.genome)

    def test_crossover_zero_rate_copies_parent_a(self):
        cfg = {**MINIMAL_CFG, "operators": {**MINIMAL_CFG["operators"], "crossover_rate": 0.0}}
        ops_no = Operators(cfg)
        ga = np.ones(16, dtype=int)
        gb = np.zeros(16, dtype=int)
        a = Individual(genome=ga)
        b = Individual(genome=gb)
        child = ops_no.crossover(a, b)
        np.testing.assert_array_equal(child.genome, ga)


# ===========================================================================
# Operators — sélection
# ===========================================================================

class TestSelection:

    def test_select_returns_n_individuals(self, ops, pop):
        selected = ops.select(pop.individuals, 5)
        assert len(selected) == 5

    def test_select_returns_alive_only(self, ops, pop):
        for ind in pop.individuals[:5]:
            ind.alive = False
        selected = ops.select(pop.individuals, 5)
        assert all(ind.alive for ind in selected)

    def test_select_empty_population(self, ops):
        result = ops.select([], 5)
        assert result == []

    def test_select_all_dead(self, ops, pop):
        for ind in pop.individuals:
            ind.alive = False
        result = ops.select(pop.individuals, 5)
        assert result == []


# ===========================================================================
# Operators — innovation cognitive
# ===========================================================================

class TestCognitiveInnovate:

    def test_no_innovation_below_threshold(self, ops):
        g = np.zeros(16, dtype=int)
        ind = Individual(genome=g)
        ind.fitness = 0.5  # sous le seuil de 0.75
        result = ops.cognitive_innovate(ind)
        assert result is False
        assert ind.cognitive is False

    def test_no_double_innovation(self, ops):
        g = np.ones(16, dtype=int)
        ind = Individual(genome=g)
        ind.fitness = 1.0
        ind.cognitive = True  # déjà innovateur
        result = ops.cognitive_innovate(ind)
        assert result is False

    def test_innovation_possible_above_threshold(self, ops):
        # Avec fitness=1.0, p≈0.1 → en 200 essais, au moins une innovation
        innovated = False
        for _ in range(200):
            g = np.ones(16, dtype=int)
            ind = Individual(genome=g)
            ind.fitness = 1.0
            if ops.cognitive_innovate(ind):
                innovated = True
                break
        assert innovated


# ===========================================================================
# Operators — mort
# ===========================================================================

class TestApplyDeath:

    def test_death_below_threshold(self, ops):
        g = np.zeros(16, dtype=int)
        ind = Individual(genome=g)
        ind.fitness = 0.01  # sous le seuil de 0.05
        deaths = ops.apply_death([ind])
        assert deaths == 1
        assert ind.alive is False

    def test_survival_above_threshold(self, ops):
        g = np.array([1, 0] * 8, dtype=int)
        ind = Individual(genome=g)
        ind.fitness = 0.9
        deaths = ops.apply_death([ind])
        assert deaths == 0
        assert ind.alive is True

    def test_already_dead_not_counted(self, ops):
        g = np.zeros(16, dtype=int)
        ind = Individual(genome=g)
        ind.fitness = 0.01
        ind.alive = False
        deaths = ops.apply_death([ind])
        assert deaths == 0


# ===========================================================================
# Population
# ===========================================================================

class TestPopulation:

    def test_initialize_creates_correct_size(self, pop):
        assert len(pop.individuals) == MINIMAL_CFG["population"]["size"]

    def test_initialize_all_alive(self, pop):
        assert all(ind.alive for ind in pop.individuals)

    def test_initialize_fitness_computed(self, pop):
        assert all(0.0 <= ind.fitness <= 1.0 for ind in pop.individuals)

    def test_step_returns_dict(self, pop):
        info = pop.step()
        assert "deaths" in info
        assert "cognitive_events" in info
        assert "pop_size" in info

    def test_step_pop_stays_positive(self, pop):
        for _ in range(5):
            info = pop.step()
        assert info["pop_size"] >= 0

    def test_snapshot_keys(self, pop):
        snap = pop.snapshot()
        expected = {"trait_x", "trait_y", "fitness", "pop_size", "mean_fitness", "diversity"}
        assert expected.issubset(set(snap.keys()))

    def test_snapshot_mean_fitness_in_range(self, pop):
        snap = pop.snapshot()
        assert 0.0 <= snap["mean_fitness"] <= 1.0

    def test_snapshot_diversity_non_negative(self, pop):
        snap = pop.snapshot()
        assert snap["diversity"] >= 0.0

    def test_snapshot_empty_population(self):
        p = Population(MINIMAL_CFG)
        snap = p.snapshot()
        assert snap["pop_size"] == 0

    def test_multiple_steps_consistent(self, pop):
        for _ in range(3):
            info = pop.step()
            snap = pop.snapshot()
            assert snap["pop_size"] == info["pop_size"]


# ===========================================================================
# simulate() — test d'intégration
# ===========================================================================

class TestSimulate:

    def test_simulate_creates_npz(self, tmp_path, monkeypatch):
        """simulate() doit créer history.npz dans data/work/."""
        # On redirige WORK_DIR vers tmp_path
        import src.core as core_mod
        monkeypatch.setattr(core_mod, "WORK_DIR", tmp_path)

        # Config très réduite pour la rapidité
        cfg = {
            "population": {"size": 10, "genome_len": 8},
            "simulation": {"n_periods": 3, "death_threshold": 0.0, "fitness_fn": "quadratic"},
            "operators": {"mutation_rate": 0.01, "crossover_rate": 0.7, "cognitive_bonus": 0.15},
        }
        # Écrit un config.yaml temporaire
        import yaml
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        core_mod.simulate(cfg_path)
        assert (tmp_path / "history.npz").exists()

    def test_simulate_creates_stats_json(self, tmp_path, monkeypatch):
        import src.core as core_mod
        monkeypatch.setattr(core_mod, "WORK_DIR", tmp_path)
        import yaml
        cfg = {
            "population": {"size": 10, "genome_len": 8},
            "simulation": {"n_periods": 3, "death_threshold": 0.0, "fitness_fn": "quadratic"},
            "operators": {"mutation_rate": 0.01, "crossover_rate": 0.7, "cognitive_bonus": 0.15},
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        core_mod.simulate(cfg_path)
        stats_path = tmp_path / "stats.json"
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        assert "pop_size" in stats
        assert "mean_fitness" in stats

    def test_simulate_npz_shape(self, tmp_path, monkeypatch):
        import src.core as core_mod
        monkeypatch.setattr(core_mod, "WORK_DIR", tmp_path)
        import yaml
        n_periods = 4
        cfg = {
            "population": {"size": 10, "genome_len": 8},
            "simulation": {"n_periods": n_periods, "death_threshold": 0.0, "fitness_fn": "quadratic"},
            "operators": {"mutation_rate": 0.01, "crossover_rate": 0.7, "cognitive_bonus": 0.15},
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        core_mod.simulate(cfg_path)
        data = np.load(tmp_path / "history.npz")
        assert data["trait_x"].shape[0] == n_periods
        assert data["trait_y"].shape[0] == n_periods
        assert data["fitness"].shape[0] == n_periods

    def test_simulate_values_in_range(self, tmp_path, monkeypatch):
        import src.core as core_mod
        monkeypatch.setattr(core_mod, "WORK_DIR", tmp_path)
        import yaml
        cfg = {
            "population": {"size": 15, "genome_len": 8},
            "simulation": {"n_periods": 3, "death_threshold": 0.0, "fitness_fn": "quadratic"},
            "operators": {"mutation_rate": 0.01, "crossover_rate": 0.7, "cognitive_bonus": 0.15},
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        core_mod.simulate(cfg_path)
        data = np.load(tmp_path / "history.npz")
        tx = data["trait_x"]
        fit = data["fitness"]
        # Toutes les valeurs valides (non-NaN) doivent être dans [0, 1]
        assert np.all((tx[~np.isnan(tx)] >= 0.0) & (tx[~np.isnan(tx)] <= 1.0))
        assert np.all((fit[~np.isnan(fit)] >= 0.0) & (fit[~np.isnan(fit)] <= 1.0))
