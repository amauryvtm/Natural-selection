"""
render.py — Visualisation 4D de l'évolution.

X = trait phénotypique A
Y = trait phénotypique B
Z = fitness moyen (par période)
T = temps (animation FuncAnimation)

Classes :
    Renderer4D — charge history.npz, produit une FuncAnimation 3D, exporte MP4 ou GIF.

Point d'entrée CLI : python -m src.render
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from config import CONFIG

import matplotlib
matplotlib.use("Agg")  # backend non-interactif, compatible serveurs sans affichage

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "data" / "work"


def _load_config(*args, **kwargs) -> dict:
    return CONFIG


# ---------------------------------------------------------------------------
# Renderer4D
# ---------------------------------------------------------------------------

class Renderer4D:
    """
    Charge history.npz et génère une animation 3D (scatter) avec
    le temps comme 4ème dimension (T → frames d'animation).

    L'axe Z représente le fitness moyen de la population à chaque génération.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.viz = cfg["visualization"]
        self.export_fmt: str = self.viz.get("export", "mp4").lower()

        # Chargement des données
        npz_path = WORK_DIR / "history.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"history.npz introuvable : {npz_path}\n"
                "Lancez d'abord : python core.py"
            )
        data = np.load(npz_path)
        self.trait_x: np.ndarray = data["trait_x"]   # shape (T, N)
        self.trait_y: np.ndarray = data["trait_y"]   # shape (T, N)
        self.fitness: np.ndarray = data["fitness"]   # shape (T, N)
        self.n_frames: int = self.trait_x.shape[0]

        # Chargement optionnel des stats
        stats_path = WORK_DIR / "stats.json"
        self.stats: dict = {}
        if stats_path.exists():
            with open(stats_path) as f:
                self.stats = json.load(f)

    # ------------------------------------------------------------------

    def _frame_data(self, t: int):
        """Renvoie les coordonnées valides à la génération t."""
        tx = self.trait_x[t]
        ty = self.trait_y[t]
        fit = self.fitness[t]
        mask = ~np.isnan(tx)
        return tx[mask], ty[mask], fit[mask]

    # ------------------------------------------------------------------

    def build(self) -> animation.FuncAnimation:
        """Construit et renvoie la FuncAnimation."""
        fig = plt.figure(figsize=(10, 7), facecolor="#0d1117")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

        mean_fit_all = np.nanmean(self.fitness, axis=1)
        cmap = plt.get_cmap("plasma")

        fig.suptitle(
            "Évolution de population — Simulation 4D\n"
            "X: trait A · Y: trait B · Z: fitness · T: temps",
            color="white", fontsize=11, y=0.98
        )

        ax.set_xlabel("Trait A (trait_x)", color="lightgray", labelpad=8)
        ax.set_ylabel("Trait B (trait_y)", color="lightgray", labelpad=8)
        ax.set_zlabel("Fitness", color="lightgray", labelpad=8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
            spine.label.set_color("lightgray")
            spine._axinfo["tick"]["color"] = "gray"

        ax.tick_params(colors="gray")
        fig.patch.set_facecolor("#0d1117")

        time_ax = fig.add_axes([0.12, 0.04, 0.76, 0.07], facecolor="#1a1f2e")
        time_ax.plot(mean_fit_all, color="#e8c84a", linewidth=1.2, alpha=0.8)
        time_ax.set_xlim(0, self.n_frames - 1)
        time_ax.set_ylim(0, 1)
        time_ax.set_xlabel("Génération", color="gray", fontsize=8)
        time_ax.set_ylabel("Fitness moy.", color="gray", fontsize=8)
        time_ax.tick_params(colors="gray", labelsize=7)
        time_line = time_ax.axvline(0, color="white", linewidth=1.5, alpha=0.9)
        time_ax.spines[:].set_color("#333")

        scatter = [None]

        def update(t: int):
            if scatter[0] is not None:
                scatter[0].remove()

            tx, ty, fit = self._frame_data(t)

            if len(tx) == 0:
                scatter[0] = ax.scatter([], [], [], s=5)
            else:
                colors = cmap(fit)
                sizes = 5 + fit * 25
                scatter[0] = ax.scatter(
                    tx, ty, fit,
                    c=colors, s=sizes, alpha=0.75, depthshade=True
                )

            pop_n = len(tx)
            mf = float(np.mean(fit)) if len(fit) > 0 else 0.0
            ax.set_title(
                f"Génération {t+1}/{self.n_frames}  |  Pop: {pop_n}  |  Fitness moy: {mf:.3f}",
                color="white", fontsize=9, pad=6
            )

            time_line.set_xdata([t, t])
            return scatter[0], time_line

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=self.n_frames,
            interval=80,
            blit=False,
        )
        self._fig = fig
        return anim

    # ------------------------------------------------------------------

    def export(self, anim: animation.FuncAnimation) -> Path:
        """Exporte l'animation en MP4 ou GIF selon la config."""
        WORK_DIR.mkdir(parents=True, exist_ok=True)

        if self.export_fmt == "mp4":
            out_path = WORK_DIR / "evolution_4d.mp4"
            try:
                writer = animation.FFMpegWriter(fps=25, bitrate=1800)
                anim.save(str(out_path), writer=writer, dpi=120)
                print(f"[render] MP4 exporté → {out_path}")
            except Exception as e:
                print(f"[render] FFMpeg indisponible ({e}). Repli sur GIF.")
                out_path = WORK_DIR / "evolution_4d.gif"
                writer = animation.PillowWriter(fps=15)
                anim.save(str(out_path), writer=writer, dpi=80)
                print(f"[render] GIF exporté → {out_path}")
        else:
            out_path = WORK_DIR / "evolution_4d.gif"
            writer = animation.PillowWriter(fps=15)
            anim.save(str(out_path), writer=writer, dpi=80)
            print(f"[render] GIF exporté → {out_path}")

        plt.close(self._fig)
        return out_path

    # ------------------------------------------------------------------

    def plot_stats(self) -> Optional[Path]:
        """Génère un graphique PNG des statistiques si stats.json existe."""
        if not self.stats:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d1117")
        fig.suptitle("Statistiques de simulation", color="white", fontsize=13)

        keys = ["pop_size", "mean_fitness", "diversity", "cognitive_events"]
        labels = ["Taille de population", "Fitness moyen", "Diversité phénotypique", "Événements cognitifs"]
        colors = ["#4fa3e0", "#e8c84a", "#5bc96a", "#e05c5c"]

        for ax, key, label, color in zip(axes.flat, keys, labels, colors):
            data = self.stats.get(key, [])
            ax.plot(data, color=color, linewidth=1.5)
            ax.set_title(label, color="white", fontsize=10)
            ax.set_facecolor("#1a1f2e")
            ax.tick_params(colors="gray")
            ax.spines[:].set_color("#333")
            ax.set_xlabel("Génération", color="gray", fontsize=8)

        plt.tight_layout()
        out_path = WORK_DIR / "stats.png"
        plt.savefig(out_path, dpi=120, facecolor="#0d1117")
        plt.close(fig)
        print(f"[render] Statistiques PNG → {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# render() — point d'entrée
# ---------------------------------------------------------------------------

def render() -> None:
    """Lance le rendu complet : animation 4D + graphique des stats."""
    cfg = _load_config()
    renderer = Renderer4D(cfg)
    print(f"[render] {renderer.n_frames} frames chargées depuis history.npz")
    anim = renderer.build()
    renderer.export(anim)
    renderer.plot_stats()
    print("[render] Terminé.")


# ---------------------------------------------------------------------------
# CLI : python render.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render()