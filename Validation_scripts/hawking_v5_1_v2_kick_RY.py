#!/usr/bin/env python3
"""
================================================================================
HAWKING V5.1 - VALIDATION ALICIA (V2 - KICK RY CORRECT)
================================================================================
Basé sur Palier 4A VALIDÉ (100% GO - 6/6 checks)

FIX CRITIQUE: Le kick doit être RY (rotation Y), pas RXX+RYY !
  - RXX(θ)·RYY(θ)|00⟩ = |00⟩  ← S'ANNULE!
  - RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩  ← CRÉE DES EXCITATIONS!

MODÈLE: Chaîne spatiale avec localisation à l'horizon
  - N qubits en chaîne linéaire
  - Profil J(x) avec minimum à l'horizon (barrière)
  - Kick RY initial crée excitations à l'horizon
  - Mesure: occupation par site → localisation

Auteur: Sebastien Icard - QMC Research Lab
Review: Alicia
Date: Janvier 2026
Version: 5.1-V2 (Kick RY correct)
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
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Architecture (comme Palier 4A validé)
    "N": 20,
    "backend_name": "ibm_fez",
    "shots": 4096,
    
    # Paramètres Trotter
    "S": 2,
    "dt": 0.5,
    
    # Profil de couplage J(x)
    "J_max": 1.0,
    "J_min": 0.1,
    
    # Kick RY (CORRECT - comme vrai Palier 4A)
    "kick_strength": 0.6,      # Angle du kick RY
    "kick_width": 2.0,         # Décroissance exponentielle du kick
    
    # Projet
    "project_name": "HAWKING_V5.1_V2",
    "auto_confirm": False,
    
    # Seuils GO/NO-GO
    "thresholds": {
        "min_gradient_inside": 0.5,
        "min_gradient_outside": 0.5,
        "min_kick_effect": 0.05,
        "max_baseline_occupation": 0.02,
        "min_n_near_horizon": 0.10,
    }
}

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


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class HawkingV51V2:
    """
    HAWKING V5.1-V2 - Modèle Palier 4A avec kick RY CORRECT
    
    Le kick RY crée des excitations |1⟩ :
      RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
      P(|1⟩) = sin²(θ/2)
    
    Pour kick_strength=0.6 : P(|1⟩) ≈ 8.7% par qubit
    """
    
    def __init__(self, config=None, config_name="mini"):
        if config is None:
            base_config = CONFIGS.get(config_name, CONFIGS["mini"])
            self.config = {**CONFIG, **base_config}
        else:
            self.config = config
            
        self.N = self.config['N']
        self.x_h = self.N // 2
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.J_profile = self._compute_J_profile()
        
        self.results = {
            "version": "5.1-V2",
            "model": "Palier_4A_kick_RY",
            "timestamp": self.timestamp,
            "config": {k: v for k, v in self.config.items() if not callable(v)},
            "J_profile": self.J_profile.tolist(),
            "baseline_test": None,
            "standard_test": None,
            "verdict": None
        }
        
        self.output_dir = Path("qmc_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"  HAWKING V5.1-V2 - Kick RY CORRECT")
        print(f"  {self.config.get('description', f'N={self.N} qubits')}")
        print(f"  Horizon: x_h = {self.x_h}")
        print(f"{'='*70}")
    
    def _compute_J_profile(self):
        """Profil J(x) avec minimum à l'horizon (barrière)."""
        N = self.N
        x_h = N // 2
        J_max = self.config.get('J_max', 1.0)
        J_min = self.config.get('J_min', 0.1)
        width = max(1, N // 10)
        
        J_profile = np.zeros(N - 1)
        for i in range(N - 1):
            x = (i + 0.5 - x_h) / width
            J_profile[i] = J_min + (J_max - J_min) * np.tanh(x)**2
        
        return J_profile
    
    # ══════════════════════════════════════════════════════════════════════════
    # CONSTRUCTION DES CIRCUITS
    # ══════════════════════════════════════════════════════════════════════════
    
    def build_circuit(self, with_kick=True):
        """
        Construit le circuit avec kick RY CORRECT.
        
        Le kick RY crée des excitations visibles en mesure Z:
          RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        """
        N = self.N
        S = self.config.get('S', 2)
        dt = self.config.get('dt', 0.5)
        kick_strength = self.config.get('kick_strength', 0.6)
        kick_width = self.config.get('kick_width', 2.0)
        
        qr = QuantumRegister(N, 'q')
        cr = ClassicalRegister(N, 'c')
        qc = QuantumCircuit(qr, cr)
        
        x_h = self.x_h
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: État initial |0...0⟩
        # ═══════════════════════════════════════════════════════════════════════
        # (déjà en |0⟩)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: KICK RY (CORRECT - comme vrai Palier 4A)
        # ═══════════════════════════════════════════════════════════════════════
        
        if with_kick and kick_strength > 0:
            # Kick RY localisé autour de l'horizon
            # Force décroît exponentiellement avec la distance
            kick_start = max(0, x_h - 3)
            kick_end = min(N, x_h + 4)
            
            for i in range(kick_start, kick_end):
                distance = abs(i - x_h)
                # Décroissance exponentielle
                kick_angle = kick_strength * np.exp(-distance / kick_width)
                # RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                qc.ry(2 * kick_angle, i)
            
            qc.barrier()
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Évolution Trotter (modèle XY isotrope)
        # ═══════════════════════════════════════════════════════════════════════
        
        for step in range(S):
            # Termes de couplage XX + YY
            for i in range(N - 1):
                J = self.J_profile[i]
                theta = J * dt
                qc.rxx(theta, i, i + 1)
                qc.ryy(theta, i, i + 1)
            
            if step < S - 1:
                qc.barrier()
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Mesure Z
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
    # ANALYSE DES RÉSULTATS
    # ══════════════════════════════════════════════════════════════════════════
    
    def analyze_counts(self, counts, with_kick=True):
        """Analyse les counts et calcule les métriques Palier 4A."""
        N = self.N
        x_h = self.x_h
        shots = sum(counts.values())
        
        # 1. Occupations par site
        occupations = np.zeros(N)
        for bitstring, count in counts.items():
            bits = bitstring[::-1]
            for i in range(min(N, len(bits))):
                if bits[i] == '1':
                    occupations[i] += count
        occupations /= shots
        
        # 2. Métriques spatiales
        distances = np.arange(N) - x_h
        
        near_mask = np.abs(distances) <= 1
        n_near_horizon = np.mean(occupations[near_mask])
        
        inside_far_mask = distances < -3
        outside_far_mask = distances > 3
        
        n_far_inside = np.mean(occupations[inside_far_mask]) if np.any(inside_far_mask) else 0
        n_far_outside = np.mean(occupations[outside_far_mask]) if np.any(outside_far_mask) else 0
        
        # 3. Gradients
        inside_indices = np.where(distances < 0)[0]
        inside_gradient = self._compute_gradient(occupations, inside_indices, x_h)
        
        outside_indices = np.where(distances > 0)[0]
        outside_gradient = self._compute_gradient(occupations, outside_indices, x_h)
        
        # 4. Corrélateurs ZZ (diagnostic)
        corr_zz = []
        for d in range(1, 6):
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
        """Fraction de sites avec occupation décroissante depuis l'horizon."""
        if len(indices) < 2:
            return 0.0
        
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
        """Calcule <Z_i Z_j>."""
        corr = 0
        for bitstring, count in counts.items():
            bits = bitstring[::-1]
            if i < len(bits) and j < len(bits):
                zi = 1 - 2 * int(bits[i])
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
        
        # 1. BASELINE (sans kick)
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
        
        # 2. STANDARD (avec kick RY)
        print(f"\n   [2/2] Circuit STANDARD (S={self.config['S']}, kick_RY={self.config['kick_strength']})...")
        qc_standard = self.build_circuit(with_kick=True)
        
        qc_standard_t = pm.run(qc_standard)
        
        job_standard = sim.run(qc_standard_t, shots=shots)
        counts_standard = job_standard.result().get_counts()
        
        standard_analysis = self.analyze_counts(counts_standard, with_kick=True)
        self.results['standard_test'] = standard_analysis
        
        print(f"      n_near_horizon = {standard_analysis['n_near_horizon']:.4f}")
        print(f"      n_far_inside = {standard_analysis['n_far_inside']:.4f}")
        print(f"      n_far_outside = {standard_analysis['n_far_outside']:.4f}")
        print(f"      gradient_inside = {standard_analysis['gradient_inside']:.2f}")
        print(f"      gradient_outside = {standard_analysis['gradient_outside']:.2f}")
        
        # 3. Verdict
        verdict = self._compute_verdict(baseline_analysis, standard_analysis)
        self.results['verdict'] = verdict
        
        return self.results
    
    def _compute_verdict(self, baseline, standard):
        """Calcule le verdict GO/NO-GO."""
        thresholds = self.config['thresholds']
        
        kick_effect = standard['n_near_horizon'] - baseline['n_near_horizon']
        
        checks = {
            "baseline_clean": baseline['max_occupation'] < thresholds['max_baseline_occupation'],
            "gradient_inside": standard['gradient_inside'] >= thresholds['min_gradient_inside'],
            "gradient_outside": standard['gradient_outside'] >= thresholds['min_gradient_outside'],
            "kick_effect": kick_effect >= thresholds['min_kick_effect'],
            "excitation_profile": standard['n_near_horizon'] >= thresholds['min_n_near_horizon'],
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
    # RAPPORT
    # ══════════════════════════════════════════════════════════════════════════
    
    def print_summary(self):
        """Affiche le résumé."""
        print(f"\n{'='*70}")
        print(f"  RÉSUMÉ V5.1-V2 (Kick RY CORRECT)")
        print(f"{'='*70}")
        
        baseline = self.results.get('baseline_test', {})
        print(f"\n📋 TEST BASELINE (sans kick):")
        print(f"   max_occupation: {baseline.get('max_occupation', 0):.4f}")
        print(f"   mean_occupation: {baseline.get('mean_occupation', 0):.4f}")
        
        standard = self.results.get('standard_test', {})
        print(f"\n📋 TEST STANDARD (avec kick RY):")
        print(f"   n_near_horizon: {standard.get('n_near_horizon', 0):.4f}")
        print(f"   n_far_inside: {standard.get('n_far_inside', 0):.4f}")
        print(f"   n_far_outside: {standard.get('n_far_outside', 0):.4f}")
        print(f"   gradient_inside: {standard.get('gradient_inside', 0):.2f}")
        print(f"   gradient_outside: {standard.get('gradient_outside', 0):.2f}")
        
        # Profil d'occupation
        if standard.get('occupations'):
            occs = standard['occupations']
            x_h = standard.get('x_horizon', len(occs)//2)
            print(f"\n📊 Profil d'occupation autour de l'horizon:")
            for i in range(max(0, x_h-3), min(len(occs), x_h+4)):
                bar = "█" * int(occs[i] * 50)
                marker = " ← HORIZON" if i == x_h else ""
                print(f"   Site {i:2d}: {occs[i]:.4f} {bar}{marker}")
        
        verdict = self.results.get('verdict', {})
        print(f"\n{'='*70}")
        print(f"  VERDICT: {verdict.get('verdict', 'N/A')}")
        print(f"  {verdict.get('message', '')}")
        print(f"  Kick effect: {verdict.get('kick_effect', 0):.4f}")
        
        if verdict.get('checks'):
            print(f"\n  Checks:")
            for check, passed in verdict['checks'].items():
                status = "✅" if passed else "❌"
                print(f"    {status} {check}")
        
        print(f"{'='*70}")
    
    def save_results(self):
        """Sauvegarde les résultats."""
        config_name = self.config.get('description', f"N{self.N}")
        safe_name = config_name.replace(' ', '_').replace('-', '_')[:30]
        
        output_path = self.output_dir / f"hawking_v5.1_v2_{safe_name}_{self.timestamp}.json"
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        results_clean = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_clean[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        results_clean[key][k] = {k2: convert(v2) for k2, v2 in v.items()}
                    else:
                        results_clean[key][k] = convert(v)
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
    parser = argparse.ArgumentParser(
        description="HAWKING V5.1-V2 - Kick RY correct"
    )
    parser.add_argument(
        '--config', '-c',
        choices=['mini', 'medium', 'large', 'extreme'],
        default='mini',
        help='Configuration (default: mini)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['aer', 'noise', 'qpu'],
        default='aer',
        help='Mode d\'exécution (default: aer)'
    )
    
    args = parser.parse_args()
    
    validator = HawkingV51V2(config_name=args.config)
    
    if args.mode == 'qpu':
        print("❌ Mode QPU: utiliser QMC Framework")
        return
    
    results = validator.run_simulation(mode=args.mode)
    
    if results:
        validator.print_summary()
        validator.save_results()
        print(f"\n✅ Validation terminée!")


if __name__ == "__main__":
    main()
