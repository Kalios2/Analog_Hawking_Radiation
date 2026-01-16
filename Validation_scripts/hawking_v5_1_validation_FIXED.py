#!/usr/bin/env python3
"""
================================================================================
HAWKING V5.1 - VALIDATION ALICIA (CORRIGÉ)
================================================================================
Basé sur Palier 4A VALIDÉ (100% GO - 6/6 checks)

MODÈLE CORRECT: Chaîne spatiale avec localisation à l'horizon
  - N qubits en chaîne linéaire
  - Profil J(x) avec minimum à l'horizon (barrière)
  - Kick initial crée excitations à l'horizon
  - Mesure: occupation par site → localisation

TESTS ALICIA:
  1. BASELINE S=0 : Pas de kick → occupations ~0 partout
  2. STANDARD S>0 : Kick → excitations localisées à l'horizon
  3. KICK_EFFECT : n_near_horizon(kick) - n_near_horizon(baseline) > seuil

Auteur: Sebastien Icard - QMC Research Lab
Review: Alicia
Date: Janvier 2026
Version: 5.1-FIXED (Palier 4A model)
================================================================================
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os
import argparse

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION V5.1-FIXED
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Architecture (comme Palier 4A validé)
    "N": 20,                        # Qubits dans la chaîne
    "backend_name": "ibm_fez",
    "shots": 4096,
    
    # Paramètres Trotter (identiques à 4A validé)
    "S": 2,                         # Steps Trotter
    "dt": 0.5,                      # Pas de temps
    
    # Profil de couplage J(x)
    "J_max": 1.0,                   # Couplage loin de l'horizon
    "J_min": 0.1,                   # Couplage à l'horizon (barrière)
    
    # Kick
    "kick_strength": 0.6,           # Force du kick (comme 4A)
    
    # Projet
    "project_name": "HAWKING_V5.1_FIXED",
    "auto_confirm": False,
    
    # Seuils GO/NO-GO (basés sur 4A validé)
    "thresholds": {
        "min_gradient_inside": 0.5,      # gradient > 50%
        "min_gradient_outside": 0.5,     # gradient > 50%
        "min_kick_effect": 0.05,         # kick_effect > 0.05
        "max_baseline_occupation": 0.02,  # occupation < 2% sans kick
        "min_n_near_horizon": 0.10,      # occupation > 10% avec kick
    }
}

# Configs pour différentes échelles
CONFIGS = {
    "mini": {"N": 20, "S": 2, "description": "20 qubits - Palier 4A mini"},
    "medium": {"N": 40, "S": 2, "description": "40 qubits - Palier 4A standard"},
    "large": {"N": 80, "S": 2, "description": "80 qubits - Palier 4A avancé"},
    "extreme": {"N": 120, "S": 1, "description": "120 qubits - Palier 4A monstre"},
}

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False
    print("⚠️ Aer non disponible")

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class HawkingV51Fixed:
    """
    HAWKING V5.1 - Modèle Palier 4A (Chaîne spatiale)
    
    Architecture:
      - Chaîne de N qubits
      - Horizon au milieu (x_h = N/2)
      - Profil J(x) avec minimum à l'horizon
      - Kick crée excitations localisées
    
    Métriques (identiques à Palier 4A validé):
      - occupations[i] : probabilité d'excitation au site i
      - n_near_horizon : occupation moyenne près de l'horizon
      - gradient_inside/outside : décroissance depuis l'horizon
      - kick_effect : différence avec baseline
    """
    
    def __init__(self, config=None, config_name="mini"):
        """Initialise avec config Palier 4A."""
        if config is None:
            base_config = CONFIGS.get(config_name, CONFIGS["mini"])
            self.config = {**CONFIG, **base_config}
        else:
            self.config = config
            
        self.N = self.config['N']
        self.x_h = self.N // 2  # Position horizon (milieu)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculer le profil J(x)
        self.J_profile = self._compute_J_profile()
        
        self.results = {
            "version": "5.1-FIXED",
            "model": "Palier_4A_spatial",
            "timestamp": self.timestamp,
            "config": {k: v for k, v in self.config.items() if not callable(v)},
            "J_profile": self.J_profile.tolist(),
            "baseline_test": None,
            "standard_test": None,
            "verdict": None
        }
        
        # Setup output
        self.output_dir = Path("qmc_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"  HAWKING V5.1-FIXED - Modèle Palier 4A")
        print(f"  {self.config.get('description', f'N={self.N} qubits')}")
        print(f"  Horizon: x_h = {self.x_h}")
        print(f"{'='*70}")
    
    def _compute_J_profile(self):
        """
        Calcule le profil de couplage J(x).
        
        J(x) = J_min + (J_max - J_min) * tanh²((x - x_h) / width)
        
        → Minimum à l'horizon (barrière)
        → Maximum loin de l'horizon
        """
        N = self.N
        x_h = N // 2
        J_max = self.config.get('J_max', 1.0)
        J_min = self.config.get('J_min', 0.1)
        
        # Largeur de la barrière (adaptatif)
        width = max(1, N // 10)
        
        J_profile = np.zeros(N - 1)
        for i in range(N - 1):
            # Position du lien (entre i et i+1)
            x = (i + 0.5 - x_h) / width
            # tanh² donne une forme en "V" inversé
            J_profile[i] = J_min + (J_max - J_min) * np.tanh(x)**2
        
        return J_profile
    
    # ══════════════════════════════════════════════════════════════════════════
    # CONSTRUCTION DES CIRCUITS (Modèle Palier 4A)
    # ══════════════════════════════════════════════════════════════════════════
    
    def build_circuit(self, with_kick=True):
        """
        Construit le circuit chaîne XY (Palier 4A).
        
        Structure:
          1. État initial |0...0⟩
          2. Kick à l'horizon (si with_kick=True)
          3. Évolution Trotter (S steps)
          4. Mesure Z
        """
        N = self.N
        S = self.config.get('S', 2)
        dt = self.config.get('dt', 0.5)
        kick_strength = self.config.get('kick_strength', 0.6)
        
        qr = QuantumRegister(N, 'q')
        cr = ClassicalRegister(N, 'c')
        qc = QuantumCircuit(qr, cr)
        
        x_h = self.x_h
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: État initial |0...0⟩ (spin down = pas d'excitation)
        # ═══════════════════════════════════════════════════════════════════════
        # Rien à faire - déjà |0⟩
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Kick initial (création d'excitations à l'horizon)
        # ═══════════════════════════════════════════════════════════════════════
        
        if with_kick and kick_strength > 0:
            # Kick sur les liens autour de l'horizon
            # RXX + RYY crée des paires d'excitations
            for link in [x_h - 1, x_h]:
                if 0 <= link < N - 1:
                    qc.rxx(kick_strength, link, link + 1)
                    qc.ryy(kick_strength, link, link + 1)
            qc.barrier()
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Évolution Trotter
        # ═══════════════════════════════════════════════════════════════════════
        
        for step in range(S):
            # ---- Termes de couplage XX + YY (modèle XY isotrope) ----
            for i in range(N - 1):
                J = self.J_profile[i]
                theta = J * dt
                qc.rxx(theta, i, i + 1)
                qc.ryy(theta, i, i + 1)  # MÊME signe = isotrope
            
            if step < S - 1:
                qc.barrier()
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Mesure en base Z (occupation = |1⟩)
        # ═══════════════════════════════════════════════════════════════════════
        
        qc.measure(qr, cr)
        
        qc.metadata = {
            'N': N,
            'S': S,
            'with_kick': with_kick,
            'kick_strength': kick_strength if with_kick else 0.0
        }
        
        return qc
    
    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSE DES RÉSULTATS (Métriques Palier 4A)
    # ══════════════════════════════════════════════════════════════════════════
    
    def analyze_counts(self, counts, with_kick=True):
        """
        Analyse les counts et calcule les métriques Palier 4A.
        
        Métriques:
          - occupations[i] : P(qubit i = |1⟩)
          - n_near_horizon : moyenne occupations autour de l'horizon
          - n_far_inside/outside : moyenne loin de l'horizon
          - gradient_inside/outside : fraction de sites décroissants
        """
        N = self.N
        x_h = self.x_h
        shots = sum(counts.values())
        
        # ═══════════════════════════════════════════════════════════════════════
        # 1. Calculer les occupations par site
        # ═══════════════════════════════════════════════════════════════════════
        
        occupations = np.zeros(N)
        for bitstring, count in counts.items():
            # Qiskit: bitstring est en ordre inverse
            bits = bitstring[::-1]
            for i in range(min(N, len(bits))):
                if bits[i] == '1':
                    occupations[i] += count
        occupations /= shots
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. Calculer les métriques spatiales
        # ═══════════════════════════════════════════════════════════════════════
        
        # Distances depuis l'horizon
        distances = np.arange(N) - x_h
        
        # Occupation près de l'horizon (|d| <= 1)
        near_mask = np.abs(distances) <= 1
        n_near_horizon = np.mean(occupations[near_mask])
        
        # Occupation loin de l'horizon (inside = d < -3, outside = d > 3)
        inside_far_mask = distances < -3
        outside_far_mask = distances > 3
        
        n_far_inside = np.mean(occupations[inside_far_mask]) if np.any(inside_far_mask) else 0
        n_far_outside = np.mean(occupations[outside_far_mask]) if np.any(outside_far_mask) else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. Calculer les gradients
        # ═══════════════════════════════════════════════════════════════════════
        
        # Inside (d < 0): occupation doit décroître quand on s'éloigne de l'horizon
        inside_indices = np.where(distances < 0)[0]
        inside_gradient = self._compute_gradient(occupations, inside_indices, x_h)
        
        # Outside (d > 0): idem
        outside_indices = np.where(distances > 0)[0]
        outside_gradient = self._compute_gradient(occupations, outside_indices, x_h)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4. Corrélateurs XX, YY, ZZ (pour diagnostic)
        # ═══════════════════════════════════════════════════════════════════════
        
        # On calcule les corrélateurs ZZ depuis les counts directement
        corr_zz = []
        for d in range(1, 6):  # d = 1 à 5
            if x_h + d < N:
                zz = self._compute_zz_correlator(counts, x_h, x_h + d, shots)
                corr_zz.append(zz)
        
        return {
            "N": N,
            "x_horizon": x_h,
            "with_kick": with_kick,
            "occupations": occupations.tolist(),
            "distances": distances.tolist(),
            "n_near_horizon": float(n_near_horizon),
            "n_far_inside": float(n_far_inside),
            "n_far_outside": float(n_far_outside),
            "gradient_inside": float(inside_gradient),
            "gradient_outside": float(outside_gradient),
            "corr_zz": corr_zz,
            "max_occupation": float(np.max(occupations)),
            "mean_occupation": float(np.mean(occupations))
        }
    
    def _compute_gradient(self, occupations, indices, x_h):
        """
        Calcule le gradient (fraction de sites avec occupation décroissante
        quand on s'éloigne de l'horizon).
        """
        if len(indices) < 2:
            return 0.0
        
        # Trier par distance depuis l'horizon
        sorted_indices = sorted(indices, key=lambda i: abs(i - x_h))
        
        decreasing_count = 0
        total = len(sorted_indices) - 1
        
        for i in range(total):
            idx_near = sorted_indices[i]
            idx_far = sorted_indices[i + 1]
            if occupations[idx_near] >= occupations[idx_far]:
                decreasing_count += 1
        
        return decreasing_count / total if total > 0 else 0.0
    
    def _compute_zz_correlator(self, counts, i, j, shots):
        """Calcule <Z_i Z_j> depuis les counts."""
        corr = 0
        for bitstring, count in counts.items():
            bits = bitstring[::-1]
            if i < len(bits) and j < len(bits):
                zi = 1 - 2 * int(bits[i])  # |0⟩ → +1, |1⟩ → -1
                zj = 1 - 2 * int(bits[j])
                corr += zi * zj * count
        return corr / shots
    
    # ══════════════════════════════════════════════════════════════════════════
    # EXÉCUTION
    # ══════════════════════════════════════════════════════════════════════════
    
    def run_simulation(self, mode="aer"):
        """Exécute en simulation."""
        if not AER_AVAILABLE:
            print("❌ Aer non disponible")
            return None
        
        print(f"\n🔬 Exécution simulation ({mode})...")
        
        sim = AerSimulator()
        shots = self.config.get('shots', 4096)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 1. Circuit BASELINE (sans kick)
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n   [1/2] Circuit BASELINE (S={self.config['S']}, kick=0)...")
        qc_baseline = self.build_circuit(with_kick=False)
        
        pm = generate_preset_pass_manager(optimization_level=1, backend=sim)
        qc_baseline_t = pm.run(qc_baseline)
        
        job_baseline = sim.run(qc_baseline_t, shots=shots)
        counts_baseline = job_baseline.result().get_counts()
        
        baseline_analysis = self.analyze_counts(counts_baseline, with_kick=False)
        self.results['baseline_test'] = baseline_analysis
        
        print(f"      max_occupation = {baseline_analysis['max_occupation']:.4f}")
        print(f"      mean_occupation = {baseline_analysis['mean_occupation']:.4f}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. Circuit STANDARD (avec kick)
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n   [2/2] Circuit STANDARD (S={self.config['S']}, kick={self.config['kick_strength']})...")
        qc_standard = self.build_circuit(with_kick=True)
        
        qc_standard_t = pm.run(qc_standard)
        
        job_standard = sim.run(qc_standard_t, shots=shots)
        counts_standard = job_standard.result().get_counts()
        
        standard_analysis = self.analyze_counts(counts_standard, with_kick=True)
        self.results['standard_test'] = standard_analysis
        
        print(f"      n_near_horizon = {standard_analysis['n_near_horizon']:.4f}")
        print(f"      gradient_inside = {standard_analysis['gradient_inside']:.2f}")
        print(f"      gradient_outside = {standard_analysis['gradient_outside']:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. Calcul du verdict
        # ═══════════════════════════════════════════════════════════════════════
        
        verdict = self._compute_verdict(baseline_analysis, standard_analysis)
        self.results['verdict'] = verdict
        
        return self.results
    
    def _compute_verdict(self, baseline, standard):
        """
        Calcule le verdict GO/NO-GO basé sur les critères Palier 4A.
        """
        thresholds = self.config['thresholds']
        
        # Kick effect = différence d'occupation près de l'horizon
        kick_effect = standard['n_near_horizon'] - baseline['n_near_horizon']
        
        checks = {
            # 1. Baseline doit être quasi-vide
            "baseline_clean": baseline['max_occupation'] < thresholds['max_baseline_occupation'],
            
            # 2. Gradient inside > seuil
            "gradient_inside": standard['gradient_inside'] >= thresholds['min_gradient_inside'],
            
            # 3. Gradient outside > seuil
            "gradient_outside": standard['gradient_outside'] >= thresholds['min_gradient_outside'],
            
            # 4. Kick effect significatif
            "kick_effect": kick_effect >= thresholds['min_kick_effect'],
            
            # 5. Occupation à l'horizon > seuil
            "excitation_profile": standard['n_near_horizon'] >= thresholds['min_n_near_horizon'],
            
            # 6. Localisation (près > loin)
            "localization": standard['n_near_horizon'] > max(standard['n_far_inside'], standard['n_far_outside'])
        }
        
        n_passed = sum(checks.values())
        n_total = len(checks)
        percentage = 100.0 * n_passed / n_total
        
        if percentage >= 100:
            verdict = "GO"
        elif percentage >= 80:
            verdict = "GO_MARGINAL"
        else:
            verdict = "NO-GO"
        
        return {
            "verdict": verdict,
            "checks": checks,
            "n_passed": n_passed,
            "n_total": n_total,
            "percentage": percentage,
            "kick_effect": kick_effect,
            "message": f"{verdict} ({n_passed}/{n_total} = {percentage:.1f}%)"
        }
    
    # ══════════════════════════════════════════════════════════════════════════
    # RAPPORT ET SAUVEGARDE
    # ══════════════════════════════════════════════════════════════════════════
    
    def print_summary(self):
        """Affiche le résumé des résultats."""
        print(f"\n{'='*70}")
        print(f"  RÉSUMÉ V5.1-FIXED (Modèle Palier 4A)")
        print(f"{'='*70}")
        
        # Baseline
        baseline = self.results.get('baseline_test', {})
        print(f"\n📋 TEST BASELINE (sans kick):")
        print(f"   max_occupation: {baseline.get('max_occupation', 'N/A'):.4f}" if baseline else "   N/A")
        print(f"   mean_occupation: {baseline.get('mean_occupation', 'N/A'):.4f}" if baseline else "   N/A")
        
        # Standard
        standard = self.results.get('standard_test', {})
        print(f"\n📋 TEST STANDARD (avec kick):")
        print(f"   n_near_horizon: {standard.get('n_near_horizon', 'N/A'):.4f}" if standard else "   N/A")
        print(f"   n_far_inside: {standard.get('n_far_inside', 'N/A'):.4f}" if standard else "   N/A")
        print(f"   n_far_outside: {standard.get('n_far_outside', 'N/A'):.4f}" if standard else "   N/A")
        print(f"   gradient_inside: {standard.get('gradient_inside', 'N/A'):.2f}" if standard else "   N/A")
        print(f"   gradient_outside: {standard.get('gradient_outside', 'N/A'):.2f}" if standard else "   N/A")
        
        # Verdict
        verdict = self.results.get('verdict', {})
        print(f"\n{'='*70}")
        print(f"  VERDICT: {verdict.get('verdict', 'N/A')}")
        print(f"  {verdict.get('message', '')}")
        
        if verdict.get('checks'):
            print(f"\n  Checks détaillés:")
            for check, passed in verdict['checks'].items():
                status = "✅" if passed else "❌"
                print(f"    {status} {check}")
        
        print(f"{'='*70}")
    
    def save_results(self):
        """Sauvegarde les résultats en JSON."""
        config_name = self.config.get('description', f"N{self.N}")
        safe_name = config_name.replace(' ', '_').replace('-', '_')[:30]
        
        output_path = self.output_dir / f"hawking_v5.1_fixed_{safe_name}_{self.timestamp}.json"
        
        # Sérialisation propre (évite circular reference)
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return str(obj)
        
        # Créer une copie propre des résultats
        results_clean = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_clean[key] = {k: convert(v) if not isinstance(v, dict) else 
                                      {k2: convert(v2) for k2, v2 in v.items()} 
                                      for k, v in value.items()}
            else:
                results_clean[key] = convert(value)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, default=convert)
        
        print(f"\n💾 Résultats sauvegardés: {output_path}")
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="HAWKING V5.1-FIXED - Validation (Modèle Palier 4A)"
    )
    parser.add_argument(
        '--config', '-c',
        choices=['mini', 'medium', 'large', 'extreme'],
        default='mini',
        help='Configuration (default: mini = 20 qubits)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['aer', 'noise', 'qpu'],
        default='aer',
        help='Mode d\'exécution (default: aer)'
    )
    
    args = parser.parse_args()
    
    # Créer l'instance
    validator = HawkingV51Fixed(config_name=args.config)
    
    # Exécuter
    if args.mode == 'qpu':
        print("❌ Mode QPU non implémenté dans cette version de test")
        print("   Utiliser le QMC Framework pour QPU réel")
        return
    
    results = validator.run_simulation(mode=args.mode)
    
    # Afficher et sauvegarder
    if results:
        validator.print_summary()
        output_path = validator.save_results()
        print(f"\n✅ Validation terminée!")


if __name__ == "__main__":
    main()
