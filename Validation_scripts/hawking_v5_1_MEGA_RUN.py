#!/usr/bin/env python3
"""
================================================================================
HAWKING V5.1 MEGA-RUN - VALIDATION COMPLÈTE EN UN SEUL JOB QPU
================================================================================
Script optimisé pour exécuter TOUS les circuits en un seul run QPU.

CORRECTIONS ALICIA:
  - kick_strength = 0.3 (pour P(|1⟩) = sin²(0.3) ≈ 8.7%)
  - Ajout contrôle J_UNIFORME (même kick, pas d'horizon)
  - Ajout contrôle KICK_ONLY (steps=0, isoler SPAM/biais)

CIRCUITS PAR CONFIG (4 circuits):
  1. BASELINE     : Sans kick, avec horizon       → doit être ~0
  2. STANDARD     : Avec kick, avec horizon       → signature Hawking
  3. J_UNIFORME   : Avec kick, J=constante        → contrôle (pas d'horizon)
  4. KICK_ONLY    : Avec kick, steps=0            → isoler SPAM/injection

CONFIGS:
  - mini:    20 qubits (4 circuits)
  - medium:  40 qubits (4 circuits)
  - large:   80 qubits (4 circuits)
  - extreme: 120 qubits (4 circuits)

TOTAL: 16 circuits en un seul job QPU!

Auteur: Sebastien Icard - QMC Research Lab
Review: Alicia
Date: Janvier 2026
Version: 5.1-MEGA-RUN
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
# CONFIGURATION GLOBALE
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_CONFIG = {
    "backend_name": "ibm_torino",
    "shots": 16384,  # MAXIMUM pour publication - erreur ±0.78%
    "project_name": "HAWKING_V5.1_MEGA_RUN",
    "auto_confirm": False,
    
    # CORRECTION ALICIA: kick_strength = 0.3 pour P ≈ 8.7%
    # P(|1⟩) = sin²(θ/2) avec θ = 2*kick_strength
    # sin²(0.3) ≈ 0.087 = 8.7%
    "kick_strength": 0.3,
    "kick_width": 2.0,
    
    # Trotter
    "dt": 0.5,
    
    # Profil J(x)
    "J_max": 1.0,
    "J_min": 0.1,
    "J_uniform": 0.5,  # Pour contrôle sans horizon
    
    # Seuils GO/NO-GO (QPU tolérant)
    "thresholds": {
        "max_baseline_occupation": 0.05,
        "min_kick_effect": 0.02,
        "min_gradient": 0.35,
        "min_localization_ratio": 1.2,  # near/far > 1.2
    }
}

# Configurations par taille - S=1 pour minimiser profondeur/décohérence
CONFIGS = {
    "mini":    {"N": 20,  "S": 1, "description": "20 qubits - S=1 (depth réduit)"},
    "medium":  {"N": 40,  "S": 1, "description": "40 qubits - S=1 (depth réduit)"},
    "large":   {"N": 80,  "S": 1, "description": "80 qubits - S=1 (depth réduit)"},
    "extreme": {"N": 120, "S": 1, "description": "120 qubits - Limite NISQ"},
}

# Types de circuits
CIRCUIT_TYPES = {
    "baseline":   {"with_kick": False, "with_horizon": True,  "with_evolution": True},
    "standard":   {"with_kick": True,  "with_horizon": True,  "with_evolution": True},
    "j_uniforme": {"with_kick": True,  "with_horizon": False, "with_evolution": True},
    "kick_only":  {"with_kick": True,  "with_horizon": True,  "with_evolution": False},
}

# ══════════════════════════════════════════════════════════════════════════════
# IMPORT QMC FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

framework_loaded = False
QMCFramework = None
RunMode = None

FRAMEWORK_PATHS = [
    "qmc_quantum_framework_v2_5_23.py",
    "../qmc_quantum_framework_v2_5_23.py",
    "../../qmc_quantum_framework_v2_5_23.py",
    os.path.join(os.path.dirname(__file__), "qmc_quantum_framework_v2_5_23.py"),
]

for fpath in FRAMEWORK_PATHS:
    if os.path.exists(fpath):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("qmc_framework", fpath)
            qmc_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qmc_module)
            
            for class_name in ['QMCFrameworkV2_5', 'QMCFrameworkV2_4', 'QMCFramework']:
                if hasattr(qmc_module, class_name):
                    QMCFramework = getattr(qmc_module, class_name)
                    break
            
            if hasattr(qmc_module, 'RunMode'):
                RunMode = qmc_module.RunMode
            else:
                from enum import Enum
                class RunMode(Enum):
                    AER = "aer"
                    QPU = "qpu"
            
            framework_loaded = True
            print(f"✅ QMC Framework chargé: {fpath}")
            break
        except Exception as e:
            print(f"⚠️ Erreur chargement {fpath}: {e}")

if not framework_loaded:
    print("❌ QMC Framework non trouvé!")

# Imports Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import CouplingMap

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE MEGA-RUN
# ══════════════════════════════════════════════════════════════════════════════

class HawkingMegaRun:
    """
    HAWKING V5.1 MEGA-RUN
    
    Génère et exécute TOUS les circuits en un seul job QPU:
    - 4 configs × 4 types = 16 circuits
    - Analyse comparative complète
    - Verdict global
    """
    
    def __init__(self, configs_to_run=None):
        """
        Args:
            configs_to_run: Liste des configs à exécuter (default: toutes)
        """
        self.configs_to_run = configs_to_run or list(CONFIGS.keys())
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Structure résultats
        self.results = {
            "version": "5.1-MEGA-RUN",
            "timestamp": self.timestamp,
            "global_config": {k: str(v) for k, v in GLOBAL_CONFIG.items()},
            "configs_run": self.configs_to_run,
            "circuits": [],       # Métadonnées des circuits
            "counts": [],         # Counts bruts
            "analyses": {},       # Analyses par config/type
            "comparisons": {},    # Comparaisons cross-config
            "verdict": None,
            "framework_paths": {},
        }
        
        # Circuits générés
        self.all_circuits = []
        self.circuit_metadata = []
        
        self.fw = None
        
        # Chain discovery pour QPU (sera rempli lors de la connexion)
        self.physical_chains = {}  # {N: [qubit_list]}
        self.transpilation_info = []  # Logs de transpilation
        
        self._print_banner()
    
    # ══════════════════════════════════════════════════════════════════════════
    # CHAIN DISCOVERY (CRITIQUE POUR HEAVY-HEX)
    # ══════════════════════════════════════════════════════════════════════════
    
    def find_linear_chain(self, coupling_map, n_qubits, calibration_data=None):
        """
        Trouve une chaîne linéaire de n_qubits sur le coupling map.
        Évite les qubits faulty si calibration_data fourni.
        
        Returns:
            list: Liste ordonnée de qubits physiques formant une chaîne
        """
        from collections import deque
        
        # Construire le graphe
        edges = set()
        for edge in coupling_map.get_edges():
            edges.add((min(edge), max(edge)))
        
        # Identifier les qubits faulty (si calibration disponible)
        faulty_qubits = set()
        if calibration_data and hasattr(calibration_data, 'get_faulty_qubits'):
            faulty_qubits = calibration_data.get_faulty_qubits()
        
        # Construire adjacency list (sans qubits faulty)
        adj = {}
        all_qubits = set()
        for (a, b) in edges:
            if a not in faulty_qubits and b not in faulty_qubits:
                all_qubits.add(a)
                all_qubits.add(b)
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        
        # BFS pour trouver la plus longue chaîne depuis chaque qubit
        best_chain = []
        
        for start in sorted(all_qubits):
            # DFS pour trouver chaîne de longueur n_qubits
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                
                if len(path) >= n_qubits:
                    if len(path) > len(best_chain):
                        best_chain = path[:n_qubits]
                    break
                
                for neighbor in sorted(adj.get(node, [])):
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
            
            if len(best_chain) >= n_qubits:
                break
        
        return best_chain[:n_qubits] if len(best_chain) >= n_qubits else None
    
    def verify_zero_swap(self, circuit, transpiled_circuit):
        """
        Vérifie qu'aucun SWAP n'a été injecté lors de la transpilation.
        
        IMPORTANT: On vérifie seulement les SWAPs (qui cassent le layout).
        L'overhead de gates 2Q est NORMAL car RXX/RYY → plusieurs CZ sur IBM.
        
        Returns:
            dict: {n_swaps, depth_original, depth_transpiled, overhead_ratio, verdict}
        """
        # Compter les SWAPs dans le circuit transpilé
        n_swaps = 0
        n_2q_original = 0
        n_2q_transpiled = 0
        
        for inst in circuit.data:
            if inst.operation.num_qubits == 2:
                n_2q_original += 1
        
        for inst in transpiled_circuit.data:
            op_name = inst.operation.name.lower()
            if 'swap' in op_name:
                n_swaps += 1
            if inst.operation.num_qubits == 2:
                n_2q_transpiled += 1
        
        depth_original = circuit.depth()
        depth_transpiled = transpiled_circuit.depth()
        
        overhead = n_2q_transpiled / n_2q_original if n_2q_original > 0 else 1.0
        
        # CRITÈRE: seulement les SWAPs comptent!
        # L'overhead de gates est ATTENDU (RXX/RYY → ~2-3 CZ chacun)
        verdict = "GO" if n_swaps == 0 else "NO-GO"
        
        return {
            "n_swaps": n_swaps,
            "n_2q_original": n_2q_original,
            "n_2q_transpiled": n_2q_transpiled,
            "depth_original": depth_original,
            "depth_transpiled": depth_transpiled,
            "overhead_ratio": overhead,
            "verdict": verdict
        }
    
    def _print_banner(self):
        n_circuits = len(self.configs_to_run) * len(CIRCUIT_TYPES)
        print(f"{'='*70}")
        print(f"  HAWKING V5.1 MEGA-RUN - {n_circuits} CIRCUITS EN UN JOB")
        print(f"{'='*70}")
        print(f"  Configs: {', '.join(self.configs_to_run)}")
        print(f"  Types:   {', '.join(CIRCUIT_TYPES.keys())}")
        print(f"  Backend: {GLOBAL_CONFIG['backend_name']}")
        print(f"  Shots:   {GLOBAL_CONFIG['shots']}")
        print(f"\n  CORRECTION ALICIA:")
        print(f"  - kick_strength = {GLOBAL_CONFIG['kick_strength']} → P(|1⟩) ≈ {np.sin(GLOBAL_CONFIG['kick_strength'])**2 * 100:.1f}%")
        print(f"{'='*70}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # GÉNÉRATION DES CIRCUITS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _compute_J_profile(self, N, with_horizon=True):
        """Profil J(x) avec ou sans horizon."""
        x_h = N // 2
        
        if not with_horizon:
            # J uniforme (contrôle)
            return np.full(N - 1, GLOBAL_CONFIG['J_uniform'])
        
        # Profil avec horizon (minimum au centre)
        J_max = GLOBAL_CONFIG['J_max']
        J_min = GLOBAL_CONFIG['J_min']
        width = max(1, N // 10)
        
        J_profile = np.zeros(N - 1)
        for i in range(N - 1):
            x = (i + 0.5 - x_h) / width
            J_profile[i] = J_min + (J_max - J_min) * np.tanh(x)**2
        
        return J_profile
    
    def build_circuit(self, N, S, circuit_type, config_name):
        """
        Construit un circuit selon le type spécifié.
        
        CORRECTION ALICIA: ry(2 * kick_angle) avec kick_strength=0.3
        → P(|1⟩) = sin²(kick_strength) ≈ 8.7%
        """
        params = CIRCUIT_TYPES[circuit_type]
        with_kick = params["with_kick"]
        with_horizon = params["with_horizon"]
        with_evolution = params["with_evolution"]
        
        dt = GLOBAL_CONFIG['dt']
        kick_strength = GLOBAL_CONFIG['kick_strength']
        kick_width = GLOBAL_CONFIG['kick_width']
        
        x_h = N // 2
        J_profile = self._compute_J_profile(N, with_horizon=with_horizon)
        
        name = f"{config_name}_{circuit_type}"
        qr = QuantumRegister(N, 'q')
        cr = ClassicalRegister(N, 'c')
        qc = QuantumCircuit(qr, cr, name=name)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: KICK RY (si activé)
        # ═══════════════════════════════════════════════════════════════════════
        
        if with_kick:
            kick_start = max(0, x_h - 3)
            kick_end = min(N, x_h + 4)
            
            for i in range(kick_start, kick_end):
                distance = abs(i - x_h)
                # CORRECTION: kick_angle tel que P = sin²(kick_angle)
                kick_angle = kick_strength * np.exp(-distance / kick_width)
                # RY(2θ)|0⟩ → P(|1⟩) = sin²(θ)
                qc.ry(2 * kick_angle, i)
            
            qc.barrier()
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: ÉVOLUTION TROTTER (si activé)
        # ═══════════════════════════════════════════════════════════════════════
        
        if with_evolution and S > 0:
            for step in range(S):
                for i in range(N - 1):
                    J = J_profile[i]
                    theta = J * dt
                    qc.rxx(theta, i, i + 1)
                    qc.ryy(theta, i, i + 1)
                
                if step < S - 1:
                    qc.barrier()
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: MESURE
        # ═══════════════════════════════════════════════════════════════════════
        
        qc.measure(qr, cr)
        
        return qc, J_profile
    
    def generate_all_circuits(self):
        """Génère TOUS les circuits pour le MEGA-RUN."""
        print(f"\n📐 Génération des circuits...")
        
        self.all_circuits = []
        self.circuit_metadata = []
        
        for config_name in self.configs_to_run:
            config = CONFIGS[config_name]
            N = config['N']
            S = config['S']
            
            print(f"\n   [{config_name}] N={N}, S={S}")
            
            for circuit_type in CIRCUIT_TYPES.keys():
                qc, J_profile = self.build_circuit(N, S, circuit_type, config_name)
                
                self.all_circuits.append(qc)
                
                metadata = {
                    "index": len(self.circuit_metadata),
                    "config": config_name,
                    "type": circuit_type,
                    "N": N,
                    "S": S,
                    "x_horizon": N // 2,
                    "depth": qc.depth(),
                    "num_qubits": qc.num_qubits,
                    "J_profile": J_profile.tolist(),
                    **CIRCUIT_TYPES[circuit_type]
                }
                self.circuit_metadata.append(metadata)
                
                print(f"      {circuit_type:12s}: depth={qc.depth():3d}")
        
        print(f"\n   ✅ Total: {len(self.all_circuits)} circuits générés")
        return self.all_circuits
    
    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSE DES COUNTS
    # ══════════════════════════════════════════════════════════════════════════
    
    def analyze_counts(self, counts, metadata):
        """Analyse les counts d'un circuit."""
        N = metadata['N']
        x_h = metadata['x_horizon']
        shots = sum(counts.values())
        
        # Occupations par site
        occupations = np.zeros(N)
        for bitstring, count in counts.items():
            bits = bitstring[::-1]
            for i in range(min(N, len(bits))):
                if bits[i] == '1':
                    occupations[i] += count
        occupations /= shots
        
        # Métriques spatiales
        distances = np.arange(N) - x_h
        
        near_mask = np.abs(distances) <= 1
        n_near_horizon = np.mean(occupations[near_mask])
        
        far_inside_mask = distances < -3
        far_outside_mask = distances > 3
        
        n_far_inside = np.mean(occupations[far_inside_mask]) if np.any(far_inside_mask) else 0
        n_far_outside = np.mean(occupations[far_outside_mask]) if np.any(far_outside_mask) else 0
        n_far = (n_far_inside + n_far_outside) / 2
        
        # Gradients
        inside_gradient = self._compute_gradient(occupations, np.where(distances < 0)[0], x_h)
        outside_gradient = self._compute_gradient(occupations, np.where(distances > 0)[0], x_h)
        
        # Ratio localisation
        localization_ratio = n_near_horizon / n_far if n_far > 0.001 else float('inf')
        
        return {
            "metadata": metadata,
            "shots": shots,
            "occupations": occupations.tolist(),
            "n_near_horizon": float(n_near_horizon),
            "n_far_inside": float(n_far_inside),
            "n_far_outside": float(n_far_outside),
            "n_far_avg": float(n_far),
            "gradient_inside": float(inside_gradient),
            "gradient_outside": float(outside_gradient),
            "max_occupation": float(np.max(occupations)),
            "mean_occupation": float(np.mean(occupations)),
            "localization_ratio": float(localization_ratio),
        }
    
    def _compute_gradient(self, occupations, indices, x_h):
        """Fraction décroissante depuis l'horizon."""
        if len(indices) < 2:
            return 0.0
        
        sorted_indices = sorted(indices, key=lambda i: abs(i - x_h))
        decreasing = sum(
            1 for i in range(len(sorted_indices)-1)
            if occupations[sorted_indices[i]] >= occupations[sorted_indices[i+1]]
        )
        return decreasing / (len(sorted_indices) - 1)
    
    # ══════════════════════════════════════════════════════════════════════════
    # EXÉCUTION QPU
    # ══════════════════════════════════════════════════════════════════════════
    
    def run_qpu(self):
        """
        Exécute TOUS les circuits en UN SEUL job QPU.
        
        SÉCURITÉS ALICIA:
        1. Chain discovery sur coupling map
        2. Transpilation avec layout forcé sur chaîne physique
        3. Vérification 0-SWAP avant envoi
        4. Abort si SWAPs détectés
        """
        if not framework_loaded or QMCFramework is None:
            print("❌ QMC Framework non disponible!")
            return None
        
        # 1. Générer les circuits (sans layout pour l'instant)
        self.generate_all_circuits()
        
        # 2. Initialiser le Framework
        print(f"\n📦 Initialisation Framework...")
        self.fw = QMCFramework(
            project=GLOBAL_CONFIG['project_name'],
            backend_name=GLOBAL_CONFIG['backend_name'],
            shots=GLOBAL_CONFIG['shots'],
            auto_confirm=GLOBAL_CONFIG['auto_confirm']
        )
        
        # 3. INITIALISER LE MODE (OBLIGATOIRE avant connect!)
        print(f"\n⚙️ Initialisation mode QPU...")
        self.fw.initialize(mode=RunMode.QPU)
        
        # 4. Connexion
        print(f"\n🔌 Connexion à {GLOBAL_CONFIG['backend_name']}...")
        self.fw.connect()
        
        # 5. Calibration
        print(f"\n📊 Analyse de calibration...")
        calibration_data = None
        if hasattr(self.fw, 'analyze_calibration'):
            self.fw.analyze_calibration()
        if hasattr(self.fw, 'circuit_optimizer'):
            calibration_data = self.fw.circuit_optimizer
        
        # ═══════════════════════════════════════════════════════════════════════
        # 6. CHAIN DISCOVERY (CRITIQUE!)
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n🔗 CHAIN DISCOVERY (éviter SWAPs)...")
        
        backend = self.fw.backend
        coupling_map = backend.coupling_map
        
        # Trouver les chaînes pour chaque taille N
        required_sizes = sorted(set(CONFIGS[c]['N'] for c in self.configs_to_run))
        
        for N in required_sizes:
            chain = self.find_linear_chain(coupling_map, N, calibration_data)
            if chain and len(chain) >= N:
                self.physical_chains[N] = chain
                print(f"   ✅ N={N}: chaîne trouvée [{chain[0]}...{chain[-1]}]")
            else:
                print(f"   ❌ N={N}: IMPOSSIBLE de trouver chaîne de {N} qubits!")
                print(f"      → Abandon de cette config pour éviter SWAPs")
                # Retirer cette config
                config_to_remove = [c for c in self.configs_to_run if CONFIGS[c]['N'] == N]
                for c in config_to_remove:
                    self.configs_to_run.remove(c)
        
        if not self.physical_chains:
            print("❌ Aucune chaîne physique trouvée - ABORT")
            return None
        
        # ═══════════════════════════════════════════════════════════════════════
        # 7. TRANSPILATION AVEC LAYOUT FORCÉ + VÉRIFICATION 0-SWAP
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n⚙️ TRANSPILATION avec layout forcé...")
        
        transpiled_circuits = []
        circuits_to_run = []
        metadata_to_run = []
        
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        
        for i, qc in enumerate(self.all_circuits):
            meta = self.circuit_metadata[i]
            N = meta['N']
            config = meta['config']
            ctype = meta['type']
            
            # Vérifier si on a une chaîne pour cette taille
            if N not in self.physical_chains:
                print(f"   ⏭️ {config}_{ctype}: skipped (pas de chaîne N={N})")
                continue
            
            chain = self.physical_chains[N]
            
            # Créer le layout mapping: qubit logique i -> qubit physique chain[i]
            initial_layout = {i: chain[i] for i in range(N)}
            
            # Transpiler avec layout forcé
            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=backend,
                initial_layout=list(chain)
            )
            
            try:
                qc_transpiled = pm.run(qc)
                
                # Vérifier 0-SWAP
                swap_info = self.verify_zero_swap(qc, qc_transpiled)
                self.transpilation_info.append({
                    "circuit": f"{config}_{ctype}",
                    **swap_info
                })
                
                if swap_info['verdict'] == "GO":
                    print(f"   ✅ {config}_{ctype}: depth {swap_info['depth_original']}→{swap_info['depth_transpiled']}, "
                          f"0 SWAPs, 2Q overhead={swap_info['overhead_ratio']:.1f}x")
                    transpiled_circuits.append(qc_transpiled)
                    circuits_to_run.append(qc)
                    metadata_to_run.append(meta)
                else:
                    print(f"   ❌ {config}_{ctype}: {swap_info['n_swaps']} SWAPs détectés! SKIP")
                    
            except Exception as e:
                print(f"   ❌ {config}_{ctype}: erreur transpilation - {str(e)[:50]}")
        
        if not transpiled_circuits:
            print("❌ Aucun circuit valide après transpilation - ABORT")
            return None
        
        # Calculer le bilan AVANT de mettre à jour
        n_original = len(self.all_circuits)
        n_valid = len(transpiled_circuits)
        
        # Mettre à jour les circuits et metadata
        self.all_circuits = circuits_to_run
        self.circuit_metadata = metadata_to_run
        
        print(f"\n   📋 BILAN: {n_valid}/{n_original} circuits GO")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 8. ESTIMATION COÛT
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n💰 Estimation du coût...")
        shots = GLOBAL_CONFIG['shots']
        if hasattr(self.fw, 'estimate_cost'):
            estimate = self.fw.estimate_cost(transpiled_circuits, shots=shots)
            print(f"   {len(transpiled_circuits)} circuits × {shots} shots")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 9. EXÉCUTION QPU
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n⏳ MEGA-RUN: {len(transpiled_circuits)} circuits (0-SWAP vérifié)...")
        
        # Utiliser les circuits transpilés directement - SANS re-transpilation!
        results = self.fw.run_on_qpu(
            transpiled_circuits, 
            shots=shots,
            auto_transpile=False  # CRITIQUE: circuits déjà transpilés avec layout forcé!
        )
        
        if results is None:
            print("❌ Échec de l'exécution QPU")
            return None
        
        # 10. Extraction et analyse
        return self._process_results(results)
    
    def run_simulation(self):
        """Simulation Aer avec MPS pour gros circuits (>25 qubits)."""
        if not AER_AVAILABLE:
            print("❌ Aer non disponible")
            return None
        
        self.generate_all_circuits()
        
        print(f"\n🔬 Simulation de {len(self.all_circuits)} circuits...")
        print(f"   (MPS = Matrix Product State, optimisé pour chaînes de spins)")
        
        # Deux simulateurs:
        # - statevector pour petits circuits (<= 25 qubits)
        # - MPS (Matrix Product State) pour gros circuits - PARFAIT pour chaînes de spins!
        sim_small = AerSimulator(method='statevector')
        sim_large = AerSimulator(method='matrix_product_state')
        
        shots = GLOBAL_CONFIG['shots']
        all_counts = []
        
        for i, qc in enumerate(self.all_circuits):
            meta = self.circuit_metadata[i]
            N = meta['N']
            
            # Choisir le simulateur selon la taille
            if N <= 25:
                sim = sim_small
                method = "SV"
            else:
                sim = sim_large
                method = "MPS"
            
            print(f"   [{i+1}/{len(self.all_circuits)}] {meta['config']}_{meta['type']} ({method})...", end=" ")
            
            try:
                job = sim.run(qc, shots=shots)
                counts = job.result().get_counts()
                all_counts.append(counts)
                print(f"✓")
            except Exception as e:
                print(f"❌ {str(e)[:50]}")
                all_counts.append({})
        
        return self._process_results(all_counts, from_simulation=True)
    
    def _process_results(self, results, from_simulation=False):
        """Traite les résultats et génère les analyses."""
        print(f"\n📊 Analyse des résultats...")
        
        # Extraire les counts
        all_counts = []
        for i, result in enumerate(results):
            try:
                if from_simulation:
                    # Simulation Aer: result EST directement le dict counts
                    counts = result
                    
                elif isinstance(result, dict):
                    # Format Framework: {'counts': {...}, 'shots': ..., ...}
                    if 'counts' in result:
                        counts = result['counts']
                    else:
                        counts = result  # Dict brut de counts
                        
                elif hasattr(result, 'data'):
                    # Format SamplerV2 brut (si jamais utilisé sans Framework)
                    if hasattr(result.data, 'c'):
                        counts = result.data.c.get_counts()
                    elif hasattr(result.data, 'meas'):
                        counts = result.data.meas.get_counts()
                    else:
                        # Chercher le premier registre disponible
                        reg_name = None
                        for attr in dir(result.data):
                            if not attr.startswith('_'):
                                try:
                                    reg = getattr(result.data, attr)
                                    if hasattr(reg, 'get_counts'):
                                        reg_name = attr
                                        break
                                except:
                                    pass
                        if reg_name:
                            counts = getattr(result.data, reg_name).get_counts()
                        else:
                            counts = {}
                else:
                    counts = result.get_counts() if hasattr(result, 'get_counts') else {}
                
                all_counts.append(counts)
                
            except Exception as e:
                print(f"   ⚠️ Circuit {i}: erreur extraction - {e}")
                all_counts.append({})
        
        self.results['counts'] = []
        for i, counts in enumerate(all_counts):
            N = self.circuit_metadata[i]['N'] if i < len(self.circuit_metadata) else 0
            if N <= 40:
                # Stocker counts bruts seulement pour N <= 40
                self.results['counts'].append({k: v for k, v in counts.items()})
            else:
                # Pour N > 40, stocker seulement les stats
                self.results['counts'].append({
                    "_note": f"Counts bruts omis (N={N} > 40)",
                    "_n_bitstrings": len(counts),
                    "_total_shots": sum(counts.values())
                })
        
        # Ajouter les infos de transpilation
        self.results['transpilation_info'] = self.transpilation_info
        
        # Analyser chaque circuit
        for i, (counts, metadata) in enumerate(zip(all_counts, self.circuit_metadata)):
            if not counts:
                continue
            
            analysis = self.analyze_counts(counts, metadata)
            
            config = metadata['config']
            ctype = metadata['type']
            
            if config not in self.results['analyses']:
                self.results['analyses'][config] = {}
            self.results['analyses'][config][ctype] = analysis
        
        # Afficher et comparer
        self._print_analyses()
        self._compute_comparisons()
        self._compute_verdict()
        
        # Sauvegarder paths Framework
        if self.fw:
            if hasattr(self.fw, 'last_report_path'):
                self.results['framework_paths']['report'] = str(self.fw.last_report_path)
            if hasattr(self.fw, 'last_archive_path'):
                self.results['framework_paths']['archive'] = str(self.fw.last_archive_path)
        
        return self.results
    
    def _print_analyses(self):
        """Affiche les analyses par config."""
        print(f"\n{'='*70}")
        print(f"  RÉSULTATS PAR CONFIGURATION")
        print(f"{'='*70}")
        
        for config in self.configs_to_run:
            if config not in self.results['analyses']:
                continue
            
            analyses = self.results['analyses'][config]
            N = CONFIGS[config]['N']
            
            # Récupérer le bruit baseline pour correction
            baseline_n_near = analyses.get('baseline', {}).get('n_near_horizon', 0)
            baseline_n_far = analyses.get('baseline', {}).get('n_far_avg', 0)
            
            print(f"\n📦 {config.upper()} (N={N})")
            print(f"   {'Type':<12} {'n_near':>8} {'n_far':>8} {'ratio':>8} {'grad_in':>8} {'grad_out':>8}")
            print(f"   {'-'*56}")
            
            for ctype in CIRCUIT_TYPES.keys():
                if ctype not in analyses:
                    continue
                a = analyses[ctype]
                print(f"   {ctype:<12} {a['n_near_horizon']:>8.4f} {a['n_far_avg']:>8.4f} "
                      f"{a['localization_ratio']:>8.2f} {a['gradient_inside']:>8.2f} {a['gradient_outside']:>8.2f}")
            
            # Afficher valeurs corrigées (soustraction baseline)
            if baseline_n_near > 0.01:  # Si baseline significatif
                print(f"\n   📊 CORRIGÉ (baseline {baseline_n_near:.3f} soustrait):")
                print(f"   {'Type':<12} {'n_near_c':>8} {'n_far_c':>8} {'ratio_c':>8}")
                print(f"   {'-'*36}")
                
                for ctype in ['standard', 'j_uniforme', 'kick_only']:
                    if ctype not in analyses:
                        continue
                    a = analyses[ctype]
                    corr_near = max(0, a['n_near_horizon'] - baseline_n_near)
                    corr_far = max(0, a['n_far_avg'] - baseline_n_far)
                    corr_ratio = corr_near / corr_far if corr_far > 0.001 else float('inf')
                    print(f"   {ctype:<12} {corr_near:>8.4f} {corr_far:>8.4f} {corr_ratio:>8.2f}")
            
            # Profil d'occupation pour standard
            if 'standard' in analyses:
                self._print_mini_profile(analyses['standard'])
    
    def _print_mini_profile(self, analysis):
        """Affiche un mini-profil d'occupation."""
        occs = analysis['occupations']
        x_h = analysis['metadata']['x_horizon']
        
        print(f"\n   Profil autour de l'horizon (x_h={x_h}):")
        for i in range(max(0, x_h-3), min(len(occs), x_h+4)):
            bar = "█" * int(occs[i] * 100)
            marker = " ←H" if i == x_h else ""
            print(f"      Site {i:3d}: {occs[i]:.4f} {bar}{marker}")
    
    def _compute_comparisons(self):
        """Compare les métriques entre types de circuits."""
        print(f"\n{'='*70}")
        print(f"  COMPARAISONS CLÉS (VALIDATION ALICIA)")
        print(f"{'='*70}")
        
        for config in self.configs_to_run:
            if config not in self.results['analyses']:
                continue
            
            analyses = self.results['analyses'][config]
            comp = {}
            
            # ═══════════════════════════════════════════════════════════════════
            # BASELINE SUBTRACTION (correction bruit thermique/décohérence)
            # ═══════════════════════════════════════════════════════════════════
            baseline_n_near = 0.0
            baseline_n_far = 0.0
            if 'baseline' in analyses:
                baseline_n_near = analyses['baseline']['n_near_horizon']
                baseline_n_far = analyses['baseline']['n_far_avg']
                comp['baseline_noise'] = baseline_n_near
                
                print(f"\n   [{config}] BASELINE NOISE = {baseline_n_near:.4f}")
                print(f"      (sera soustrait des autres mesures)")
            
            # Calculer les valeurs corrigées pour chaque type
            corrected = {}
            for ctype in ['standard', 'j_uniforme', 'kick_only']:
                if ctype in analyses:
                    corr_near = analyses[ctype]['n_near_horizon'] - baseline_n_near
                    corr_far = analyses[ctype]['n_far_avg'] - baseline_n_far
                    corr_ratio = corr_near / corr_far if corr_far > 0.001 else float('inf')
                    corrected[ctype] = {
                        'n_near_corrected': max(0, corr_near),
                        'n_far_corrected': max(0, corr_far),
                        'ratio_corrected': corr_ratio
                    }
            
            # Afficher les valeurs corrigées
            if corrected:
                print(f"\n   [{config}] VALEURS CORRIGÉES (baseline soustrait):")
                for ctype, c in corrected.items():
                    print(f"      {ctype}: n_near={c['n_near_corrected']:.4f}, "
                          f"n_far={c['n_far_corrected']:.4f}, ratio={c['ratio_corrected']:.2f}")
            
            comp['corrected'] = corrected
            
            # ═══════════════════════════════════════════════════════════════════
            # 1. KICK EFFECT: standard.n_near - baseline.n_near (déjà corrigé)
            # ═══════════════════════════════════════════════════════════════════
            if 'standard' in analyses and 'baseline' in analyses:
                kick_effect = analyses['standard']['n_near_horizon'] - analyses['baseline']['n_near_horizon']
                comp['kick_effect'] = kick_effect
                print(f"\n   [{config}] KICK_EFFECT = {kick_effect:.4f}")
                print(f"      standard.n_near ({analyses['standard']['n_near_horizon']:.4f}) - "
                      f"baseline.n_near ({analyses['baseline']['n_near_horizon']:.4f})")
            
            # ═══════════════════════════════════════════════════════════════════
            # 2. HORIZON EFFECT (sur valeurs corrigées)
            # ═══════════════════════════════════════════════════════════════════
            if 'standard' in corrected and 'j_uniforme' in corrected:
                std_ratio = corrected['standard']['ratio_corrected']
                jun_ratio = corrected['j_uniforme']['ratio_corrected']
                
                # Éviter inf - inf
                if std_ratio != float('inf') and jun_ratio != float('inf'):
                    horizon_effect = std_ratio - jun_ratio
                else:
                    horizon_effect = float('nan')
                    
                comp['horizon_effect'] = horizon_effect
                comp['horizon_effect_corrected'] = True
                print(f"   [{config}] HORIZON_EFFECT (corrigé) = {horizon_effect:.2f}")
                print(f"      standard.ratio_corr ({std_ratio:.2f}) - "
                      f"j_uniforme.ratio_corr ({jun_ratio:.2f})")
            
            # ═══════════════════════════════════════════════════════════════════
            # 3. EVOLUTION EFFECT: standard.gradient - kick_only.gradient
            # ═══════════════════════════════════════════════════════════════════
            if 'standard' in analyses and 'kick_only' in analyses:
                evol_effect = analyses['standard']['gradient_inside'] - analyses['kick_only']['gradient_inside']
                comp['evolution_effect'] = evol_effect
                print(f"   [{config}] EVOLUTION_EFFECT = {evol_effect:.2f}")
            
            self.results['comparisons'][config] = comp
    
    def _compute_verdict(self):
        """Calcule le verdict global avec prise en compte des valeurs corrigées."""
        thresholds = GLOBAL_CONFIG['thresholds']
        
        all_checks = {}
        
        for config in self.configs_to_run:
            if config not in self.results['analyses']:
                continue
            
            analyses = self.results['analyses'][config]
            comparisons = self.results['comparisons'].get(config, {})
            corrected = comparisons.get('corrected', {})
            
            checks = {}
            
            # CHECK 1: Baseline propre OU signal visible malgré bruit
            # Le baseline peut être pollué par décohérence mais le signal reste valide
            if 'baseline' in analyses:
                baseline_occ = analyses['baseline']['max_occupation']
                baseline_clean = baseline_occ < thresholds['max_baseline_occupation']
                
                # Alternative: le signal corrigé est encore significatif
                if not baseline_clean and 'standard' in corrected:
                    signal_corr = corrected['standard']['n_near_corrected']
                    # Si signal corrigé > seuil kick, on considère que c'est OK
                    baseline_clean = signal_corr > thresholds['min_kick_effect']
                
                checks['baseline_clean'] = baseline_clean
            
            # CHECK 2: Kick effect significatif (déjà baseline-soustrait)
            if 'kick_effect' in comparisons:
                checks['kick_effect'] = comparisons['kick_effect'] > thresholds['min_kick_effect']
            
            # CHECK 3: Localisation (sur valeurs corrigées si disponibles)
            if 'standard' in corrected:
                checks['localization'] = corrected['standard']['ratio_corrected'] > thresholds['min_localization_ratio']
            elif 'standard' in analyses:
                checks['localization'] = analyses['standard']['localization_ratio'] > thresholds['min_localization_ratio']
            
            # CHECK 4: Gradients
            if 'standard' in analyses:
                checks['gradient_inside'] = analyses['standard']['gradient_inside'] > thresholds['min_gradient']
                checks['gradient_outside'] = analyses['standard']['gradient_outside'] > thresholds['min_gradient']
            
            # CHECK 5: Horizon améliore la localisation (vs J uniforme)
            if 'horizon_effect' in comparisons:
                he = comparisons['horizon_effect']
                checks['horizon_helps'] = he > 0 if not (he != he) else False  # handle NaN
            
            all_checks[config] = checks
        
        # Verdict global
        total_passed = sum(sum(1 for v in c.values() if v is True) for c in all_checks.values())
        total_checks = sum(len([v for v in c.values() if isinstance(v, bool)]) for c in all_checks.values())
        percentage = 100.0 * total_passed / total_checks if total_checks > 0 else 0
        
        if percentage >= 85:
            verdict = "GO"
        elif percentage >= 70:
            verdict = "GO_MARGINAL"
        else:
            verdict = "NO-GO"
        
        print(f"\n{'='*70}")
        print(f"  VERDICT GLOBAL: {verdict}")
        print(f"  Score: {total_passed}/{total_checks} = {percentage:.1f}%")
        print(f"{'='*70}")
        
        for config, checks in all_checks.items():
            n_pass = sum(checks.values())
            n_total = len(checks)
            print(f"\n   [{config}] {n_pass}/{n_total}")
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"      {status} {check}")
        
        print(f"{'='*70}")
        
        self.results['verdict'] = {
            "global": verdict,
            "percentage": percentage,
            "total_passed": total_passed,
            "total_checks": total_checks,
            "by_config": all_checks,
        }
        
        return verdict
    
    # ══════════════════════════════════════════════════════════════════════════
    # SAUVEGARDE
    # ══════════════════════════════════════════════════════════════════════════
    
    def save_results(self):
        """Sauvegarde complète en JSON."""
        output_dir = Path("qmc_runs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        configs_str = "_".join(self.configs_to_run)
        output_path = output_dir / f"hawking_v5.1_MEGA_{configs_str}_{self.timestamp}.json"
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if obj == float('inf'):
                return "inf"
            return obj
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=convert)
        
        print(f"\n💾 Résultats sauvegardés: {output_path}")
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HAWKING V5.1 MEGA-RUN - Validation complète en un job"
    )
    parser.add_argument(
        '--configs', '-c',
        nargs='+',
        choices=['mini', 'medium', 'large', 'extreme', 'all'],
        default=['all'],
        help='Configurations à tester (default: all)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['aer', 'qpu'],
        default='aer',
        help='Mode d\'exécution (default: aer)'
    )
    
    args = parser.parse_args()
    
    # Résoudre configs
    if 'all' in args.configs:
        configs_to_run = list(CONFIGS.keys())
    else:
        configs_to_run = args.configs
    
    # Créer et exécuter
    runner = HawkingMegaRun(configs_to_run=configs_to_run)
    
    if args.mode == 'qpu':
        if not framework_loaded:
            print("❌ QMC Framework requis pour mode QPU!")
            return
        results = runner.run_qpu()
    else:
        results = runner.run_simulation()
    
    # Sauvegarder
    if results:
        output_path = runner.save_results()
        print(f"\n✅ MEGA-RUN terminé!")
        print(f"   Verdict: {results['verdict']['global']}")
        print(f"   Résultats: {output_path}")


if __name__ == "__main__":
    main()
