#!/usr/bin/env python3
"""
HAWKING V5.2.2 MEGA-RUN - Analog Hawking Radiation Simulation
==============================================================
QMC Research Lab - Janvier 2026

FIXES V5.2.2:
- Gestion robuste des formats de résultats Framework
- Extraction des counts multi-format
- Meilleure gestion des erreurs

Usage:
    python hawking_v5_2_MEGA_RUN.py --mode qpu --backend ibm_fez --trotter 1 --configs mini medium --shuffle
"""

import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

FRAMEWORK_LOADED = False

try:
    import glob
    framework_files = glob.glob("qmc_quantum_framework_v2_5_*.py")
    if framework_files:
        framework_file = sorted(framework_files)[-1]
        framework_module = framework_file.replace(".py", "")
        exec(f"from {framework_module} import QMCFrameworkV2_4 as QMCFramework, RunMode")
        FRAMEWORK_LOADED = True
        print(f"✅ QMC Framework chargé: {framework_file}")
    else:
        print("⚠️ QMC Framework non trouvé, mode standalone")
except Exception as e:
    print(f"⚠️ Erreur chargement Framework: {e}")

from qiskit import QuantumCircuit, transpile

# ============================================================================
# CONSTANTES
# ============================================================================

CONFIGS = {
    "mini":    {"n_qubits": 20,  "description": "20 qubits - Test rapide"},
    "medium":  {"n_qubits": 40,  "description": "40 qubits - Validation"},
    "large":   {"n_qubits": 80,  "description": "80 qubits - Scale test"},
    "extreme": {"n_qubits": 120, "description": "120 qubits - Maximum"},
}

DEFAULT_PARAMS = {
    "kick_strength": 0.3,
    "kick_width": 2.0,
    "dt": 0.5,
    "J_max": 1.0,
    "J_min": 0.1,
    "J_uniform": 0.5,
}

THRESHOLDS = {
    "max_baseline_occupation": 0.25,
    "min_kick_effect": 0.02,
    "min_gradient": 0.35,
    "min_localization_ratio": 1.2,
}

# ============================================================================
# FONCTIONS DE GÉNÉRATION DE CIRCUITS
# ============================================================================

def get_J_profile(n_qubits: int, profile_type: str = "horizon") -> np.ndarray:
    """Génère le profil de couplage J(x)."""
    x_h = n_qubits // 2
    
    if profile_type == "uniform":
        return np.full(n_qubits - 1, DEFAULT_PARAMS["J_uniform"])
    
    J = np.zeros(n_qubits - 1)
    for i in range(n_qubits - 1):
        x = i + 0.5
        J[i] = DEFAULT_PARAMS["J_min"] + (DEFAULT_PARAMS["J_max"] - DEFAULT_PARAMS["J_min"]) * \
               (1 - np.tanh((x - x_h) / 3)) / 2
    
    return J


def build_trotter_layer(qc: QuantumCircuit, J: np.ndarray, dt: float):
    """Applique une couche de Trotter pour l'évolution XY."""
    n = qc.num_qubits
    
    for i in range(0, n - 1, 2):
        theta = J[i] * dt
        qc.rxx(theta, i, i + 1)
        qc.ryy(theta, i, i + 1)
    
    for i in range(1, n - 1, 2):
        theta = J[i] * dt
        qc.rxx(theta, i, i + 1)
        qc.ryy(theta, i, i + 1)


def build_circuit(n_qubits: int, circuit_type: str, s_trotter: int = 1) -> Tuple[QuantumCircuit, Dict]:
    """Construit un circuit pour le type spécifié."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    x_h = n_qubits // 2
    kick_pos = x_h
    
    metadata = {
        "n_qubits": n_qubits,
        "type": circuit_type,
        "s_trotter": s_trotter,
        "x_horizon": x_h,
        "kick_pos": kick_pos,
        "with_kick": False,
        "with_horizon": False,
        "with_evolution": False,
        "shuffled": False,
    }
    
    kick_theta = 2 * np.arcsin(np.sqrt(DEFAULT_PARAMS["kick_strength"]))
    
    if circuit_type == "baseline":
        J = get_J_profile(n_qubits, "horizon")
        metadata["J_profile"] = J.tolist()
        metadata["with_evolution"] = True
        metadata["with_horizon"] = True
        
        for _ in range(s_trotter):
            build_trotter_layer(qc, J, DEFAULT_PARAMS["dt"])
    
    elif circuit_type == "standard":
        J = get_J_profile(n_qubits, "horizon")
        metadata["J_profile"] = J.tolist()
        metadata["with_kick"] = True
        metadata["with_evolution"] = True
        metadata["with_horizon"] = True
        
        qc.ry(kick_theta, kick_pos)
        
        for _ in range(s_trotter):
            build_trotter_layer(qc, J, DEFAULT_PARAMS["dt"])
    
    elif circuit_type == "j_uniforme":
        J = get_J_profile(n_qubits, "uniform")
        metadata["J_profile"] = J.tolist()
        metadata["with_kick"] = True
        metadata["with_evolution"] = True
        metadata["with_horizon"] = False
        
        qc.ry(kick_theta, kick_pos)
        
        for _ in range(s_trotter):
            build_trotter_layer(qc, J, DEFAULT_PARAMS["dt"])
    
    elif circuit_type == "kick_only":
        metadata["with_kick"] = True
        metadata["with_evolution"] = False
        metadata["with_horizon"] = True
        
        qc.ry(kick_theta, kick_pos)
    
    elif circuit_type == "shuffle":
        J = get_J_profile(n_qubits, "horizon")
        metadata["J_profile"] = J.tolist()
        metadata["with_kick"] = True
        metadata["with_evolution"] = True
        metadata["with_horizon"] = True
        metadata["shuffled"] = True
        
        qc.ry(kick_theta, kick_pos)
        
        for _ in range(s_trotter):
            build_trotter_layer(qc, J, DEFAULT_PARAMS["dt"])
        
        # SHUFFLE: Permutation pseudo-aléatoire
        np.random.seed(42)
        perm = np.random.permutation(n_qubits)
        metadata["shuffle_permutation"] = perm.tolist()
        
        # Appliquer via cycles de transpositions
        visited = [False] * n_qubits
        for i in range(n_qubits):
            if visited[i]:
                continue
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = int(perm[j])
            # Appliquer les SWAPs pour ce cycle
            for k in range(len(cycle) - 1):
                qc.swap(cycle[0], cycle[k + 1])
    
    else:
        raise ValueError(f"Type de circuit inconnu: {circuit_type}")
    
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc, metadata


# ============================================================================
# EXTRACTION ROBUSTE DES COUNTS
# ============================================================================

def extract_counts_robust(results: Any, expected_count: int) -> List[Dict[str, int]]:
    """
    Extrait les counts de manière robuste, quel que soit le format.
    
    Gère:
    - Liste de dicts avec clé 'counts'
    - Liste directe de counts
    - Dict avec clé 'counts' contenant une liste
    - Objet Result de Qiskit
    """
    all_counts = []
    
    print(f"\n🔍 DEBUG: Extraction des counts...")
    print(f"   Type de résultat: {type(results)}")
    
    # Cas 1: Liste de résultats (format Framework typique)
    if isinstance(results, list):
        print(f"   Format: liste de {len(results)} éléments")
        
        for i, item in enumerate(results):
            counts = None
            
            # Sous-cas 1a: Dict avec clé 'counts'
            if isinstance(item, dict):
                if 'counts' in item:
                    counts = item['counts']
                elif 'data' in item and isinstance(item['data'], dict):
                    counts = item['data'].get('counts', {})
                else:
                    # Le dict lui-même pourrait être les counts
                    # Vérifier si les clés ressemblent à des bitstrings
                    keys = list(item.keys())
                    if keys and all(isinstance(k, str) and set(k).issubset({'0', '1'}) for k in keys[:5]):
                        counts = item
            
            # Sous-cas 1b: L'item est directement un dict de counts
            elif hasattr(item, 'keys'):
                counts = dict(item)
            
            if counts is not None:
                # S'assurer que c'est un dict propre
                if isinstance(counts, dict):
                    all_counts.append(counts)
                    print(f"   ✅ Circuit {i}: {len(counts)} outcomes")
                else:
                    print(f"   ⚠️ Circuit {i}: format counts inattendu: {type(counts)}")
            else:
                print(f"   ⚠️ Circuit {i}: pas de counts trouvés dans {type(item)}")
    
    # Cas 2: Dict global avec clé 'counts'
    elif isinstance(results, dict):
        print(f"   Format: dict avec clés {list(results.keys())[:5]}...")
        
        if 'counts' in results:
            counts_data = results['counts']
            if isinstance(counts_data, list):
                for i, c in enumerate(counts_data):
                    if isinstance(c, dict):
                        all_counts.append(c)
                        print(f"   ✅ Circuit {i}: {len(c)} outcomes")
            elif isinstance(counts_data, dict):
                all_counts.append(counts_data)
        elif 'results' in results:
            # Récursion
            return extract_counts_robust(results['results'], expected_count)
    
    # Cas 3: Objet Result Qiskit
    elif hasattr(results, 'get_counts'):
        print(f"   Format: Qiskit Result object")
        for i in range(expected_count):
            try:
                counts = results.get_counts(i)
                all_counts.append(counts)
                print(f"   ✅ Circuit {i}: {len(counts)} outcomes")
            except Exception as e:
                print(f"   ⚠️ Circuit {i}: {e}")
    
    print(f"\n   📊 Total: {len(all_counts)} jeux de counts extraits")
    
    # Vérification finale
    if len(all_counts) != expected_count:
        print(f"   ⚠️ ATTENTION: Attendu {expected_count}, obtenu {len(all_counts)}")
    
    # Validation des counts
    for i, counts in enumerate(all_counts):
        if counts:
            total = sum(counts.values())
            print(f"   Circuit {i}: {total} shots, {len(counts)} bitstrings")
    
    return all_counts


# ============================================================================
# FONCTIONS D'ANALYSE
# ============================================================================

def analyze_counts(counts: Dict[str, int], n_qubits: int, metadata: Dict) -> Dict:
    """Analyse les counts pour extraire les métriques."""
    x_h = n_qubits // 2
    total_shots = sum(counts.values())
    
    occupations = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        # Prendre les n_qubits bits de droite (convention Qiskit)
        bits = bitstring[-n_qubits:] if len(bitstring) >= n_qubits else bitstring.zfill(n_qubits)
        for i, bit in enumerate(reversed(bits)):
            if bit == '1':
                occupations[i] += count
    occupations /= total_shots
    
    near_range = 3
    near_start = max(0, x_h - near_range)
    near_end = min(n_qubits, x_h + near_range + 1)
    
    far_inside = occupations[:near_start] if near_start > 0 else np.array([])
    far_outside = occupations[near_end:] if near_end < n_qubits else np.array([])
    near_horizon = occupations[near_start:near_end]
    
    n_near = np.mean(near_horizon) if len(near_horizon) > 0 else 0
    n_far_in = np.mean(far_inside) if len(far_inside) > 0 else 0
    n_far_out = np.mean(far_outside) if len(far_outside) > 0 else 0
    n_far = (n_far_in + n_far_out) / 2 if (len(far_inside) > 0 and len(far_outside) > 0) else max(n_far_in, n_far_out)
    
    ratio = n_near / n_far if n_far > 1e-6 else float('inf')
    
    grad_in = (occupations[x_h] - occupations[0]) / occupations[0] if occupations[0] > 1e-6 else 0
    grad_out = (occupations[x_h] - occupations[-1]) / occupations[-1] if occupations[-1] > 1e-6 else 0
    
    return {
        "metadata": metadata,
        "shots": total_shots,
        "occupations": occupations.tolist(),
        "n_near_horizon": float(n_near),
        "n_far_inside": float(n_far_in),
        "n_far_outside": float(n_far_out),
        "n_far_avg": float(n_far),
        "gradient_inside": float(grad_in),
        "gradient_outside": float(grad_out),
        "max_occupation": float(np.max(occupations)),
        "max_site": int(np.argmax(occupations)),
        "mean_occupation": float(np.mean(occupations)),
        "localization_ratio": float(ratio),
    }


def compute_comparisons(results: Dict[str, Dict], config_name: str) -> Dict:
    """Calcule les comparaisons clés."""
    baseline = results.get("baseline", {})
    standard = results.get("standard", {})
    j_uniforme = results.get("j_uniforme", {})
    kick_only = results.get("kick_only", {})
    shuffle = results.get("shuffle", {})
    
    baseline_noise = baseline.get("n_near_horizon", 0)
    
    corrected = {}
    for name, data in [("standard", standard), ("j_uniforme", j_uniforme), 
                       ("kick_only", kick_only), ("shuffle", shuffle)]:
        if data:
            n_near_c = max(0, data.get("n_near_horizon", 0) - baseline_noise)
            n_far_c = max(0, data.get("n_far_avg", 0) - baseline_noise)
            ratio_c = n_near_c / n_far_c if n_far_c > 1e-6 else float('inf')
            corrected[name] = {
                "n_near_corrected": n_near_c,
                "n_far_corrected": n_far_c,
                "ratio_corrected": ratio_c,
            }
    
    kick_effect = standard.get("n_near_horizon", 0) - baseline_noise if standard else 0
    
    std_ratio_c = corrected.get("standard", {}).get("ratio_corrected", 0)
    uni_ratio_c = corrected.get("j_uniforme", {}).get("ratio_corrected", 0)
    horizon_effect = std_ratio_c - uni_ratio_c if std_ratio_c != float('inf') and uni_ratio_c != float('inf') else float('nan')
    
    # Shuffle analysis
    std_max_site = standard.get("max_site", -1) if standard else -1
    shuf_max_site = shuffle.get("max_site", -1) if shuffle else -1
    shuffle_moved_peak = (std_max_site != shuf_max_site) if shuffle else False
    
    std_ratio = standard.get("localization_ratio", 1) if standard else 1
    shuf_ratio = shuffle.get("localization_ratio", 1) if shuffle else 1
    shuffle_ratio_change = (std_ratio - shuf_ratio) / std_ratio * 100 if std_ratio > 0 else 0
    
    kick_ratio = kick_only.get("localization_ratio", 1) if kick_only else 1
    evolution_effect = (std_ratio - kick_ratio) / kick_ratio if kick_ratio > 0 else 0
    
    return {
        "baseline_noise": baseline_noise,
        "corrected": corrected,
        "kick_effect": kick_effect,
        "horizon_effect": horizon_effect,
        "evolution_effect": evolution_effect,
        "shuffle_moved_peak": shuffle_moved_peak,
        "shuffle_ratio_change": shuffle_ratio_change,
        "std_max_site": std_max_site,
        "shuf_max_site": shuf_max_site,
    }


def compute_verdict(results: Dict[str, Dict], comparisons: Dict, config_name: str) -> Dict:
    """Calcule le verdict GO/NO-GO."""
    baseline = results.get("baseline", {})
    standard = results.get("standard", {})
    j_uniforme = results.get("j_uniforme", {})
    shuffle = results.get("shuffle", {})
    
    checks = {}
    
    checks["baseline_clean"] = baseline.get("n_near_horizon", 1) < THRESHOLDS["max_baseline_occupation"]
    checks["kick_effect"] = comparisons.get("kick_effect", 0) > THRESHOLDS["min_kick_effect"]
    checks["localization"] = standard.get("localization_ratio", 0) > THRESHOLDS["min_localization_ratio"]
    checks["gradient_inside"] = standard.get("gradient_inside", 0) > THRESHOLDS["min_gradient"]
    checks["gradient_outside"] = standard.get("gradient_outside", 0) > THRESHOLDS["min_gradient"]
    
    std_ratio = standard.get("localization_ratio", 0) if standard else 0
    uni_ratio = j_uniforme.get("localization_ratio", 0) if j_uniforme else 0
    checks["horizon_helps"] = std_ratio > uni_ratio * 1.05
    
    # Nouveau critère shuffle: le pic doit se déplacer
    if shuffle:
        checks["shuffle_moves_peak"] = comparisons.get("shuffle_moved_peak", False)
    
    return checks


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HAWKING V5.2.2 MEGA-RUN - Analog Hawking Radiation Simulation",
    )
    
    parser.add_argument('--configs', nargs='+', 
                        choices=['mini', 'medium', 'large', 'extreme', 'all'],
                        default=['mini'],
                        help='Configurations à exécuter')
    
    parser.add_argument('--mode', choices=['aer', 'qpu'], default='aer',
                        help='Mode d\'exécution')
    
    parser.add_argument('--backend', choices=['ibm_fez', 'ibm_torino'], default='ibm_fez',
                        help='Backend QPU')
    
    parser.add_argument('--trotter', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Nombre de pas de Trotter S')
    
    parser.add_argument('--shots', type=int, default=16384,
                        help='Nombre de shots')
    
    parser.add_argument('--shuffle', action='store_true',
                        help='Inclure le circuit shuffle')
    
    parser.add_argument('--no-baseline', action='store_true',
                        help='Exclure le circuit baseline')
    
    args = parser.parse_args()
    
    if 'all' in args.configs:
        configs_to_run = ['mini', 'medium', 'large', 'extreme']
    else:
        configs_to_run = args.configs
    
    circuit_types_to_run = ["baseline", "standard", "j_uniforme", "kick_only"]
    if args.shuffle:
        circuit_types_to_run.append("shuffle")
    if args.no_baseline:
        circuit_types_to_run.remove("baseline")
    
    total_circuits = len(configs_to_run) * len(circuit_types_to_run)
    
    print("=" * 70)
    print(f"  HAWKING V5.2.2 MEGA-RUN - {total_circuits} CIRCUITS EN UN JOB")
    print("=" * 70)
    print(f"  Configs: {', '.join(configs_to_run)}")
    print(f"  Types:   {', '.join(circuit_types_to_run)}")
    print(f"  Backend: {args.backend}")
    print(f"  Trotter: S={args.trotter}")
    print(f"  Shots:   {args.shots}")
    print()
    print("  CORRECTION ALICIA:")
    print(f"  - kick_strength = {DEFAULT_PARAMS['kick_strength']} → P(|1⟩) ≈ 8.7%")
    if args.shuffle:
        print("  - Circuit SHUFFLE inclus (Option C)")
    if args.trotter > 1:
        print(f"  - S={args.trotter} Trotter steps (Option B)")
    print("=" * 70)
    print()
    
    # ========================================================================
    # GÉNÉRATION DES CIRCUITS
    # ========================================================================
    
    print("📐 Génération des circuits...")
    print()
    
    all_circuits = []
    all_metadata = []
    circuits_by_config = {}
    
    for config_name in configs_to_run:
        n_qubits = CONFIGS[config_name]["n_qubits"]
        circuits_by_config[config_name] = {}
        
        print(f"   [{config_name}] N={n_qubits}, S={args.trotter}")
        
        for circuit_type in circuit_types_to_run:
            try:
                qc, meta = build_circuit(n_qubits, circuit_type, args.trotter)
                
                circuit_name = f"{config_name}_{circuit_type}"
                qc.name = circuit_name
                meta["circuit_name"] = circuit_name
                meta["config"] = config_name
                
                all_circuits.append(qc)
                all_metadata.append(meta)
                circuits_by_config[config_name][circuit_type] = {
                    "circuit": qc,
                    "metadata": meta,
                    "index": len(all_circuits) - 1,
                }
                
                depth = qc.depth()
                print(f"      {circuit_type:12s}: depth={depth:4d}")
                
            except Exception as e:
                print(f"      {circuit_type:12s}: ❌ ERREUR - {e}")
    
    print()
    print(f"   ✅ Total: {len(all_circuits)} circuits générés")
    print()
    
    # ========================================================================
    # EXÉCUTION
    # ========================================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    skipped_configs = []
    valid_circuits = all_circuits
    valid_metadata = all_metadata
    transpilation_info = []
    all_counts = []
    
    if args.mode == 'aer':
        print("📦 Mode AER (simulateur)...")
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        
        print("⏳ Transpilation et exécution...")
        transpiled = transpile(all_circuits, backend, optimization_level=1)
        
        job = backend.run(transpiled, shots=args.shots)
        result = job.result()
        
        all_counts = [result.get_counts(i) for i in range(len(all_circuits))]
        print("✅ Exécution AER terminée!")
        
    else:
        # Mode QPU avec Framework
        if not FRAMEWORK_LOADED:
            print("❌ ERREUR: QMC Framework requis pour mode QPU!")
            sys.exit(1)
        
        print("📦 Initialisation Framework...")
        print()
        print("⚙️ Initialisation mode QPU...")
        
        # Import dynamique
        import importlib
        import glob as glob_module
        framework_files = glob_module.glob("qmc_quantum_framework_v2_5_*.py")
        if framework_files:
            framework_file = sorted(framework_files)[-1]
            framework_module_name = framework_file.replace(".py", "")
            fw_module = importlib.import_module(framework_module_name)
            QMCFramework = fw_module.QMCFrameworkV2_4
            RunMode = fw_module.RunMode
        
        fw = QMCFramework(
            project="HAWKING_V5.2_MEGA_RUN",
            backend_name=args.backend,
            shots=args.shots,
            auto_confirm=False,
        )
        
        fw.initialize(mode=RunMode.QPU)
        
        print(f"\n🔌 Connexion à {args.backend}...")
        fw.connect()
        
        print("\n📊 Analyse de calibration...")
        fw.analyze_calibration()
        
        # Chain discovery
        print("\n🔗 CHAIN DISCOVERY (éviter SWAPs)...")
        
        valid_circuits = []
        valid_metadata = []
        config_chains = {}
        
        for config_name in configs_to_run:
            n_qubits = CONFIGS[config_name]["n_qubits"]
            
            try:
                # Fallback simple: utiliser les premiers qubits
                chain = list(range(n_qubits))
                
                # Essayer le layout optimal si disponible
                try:
                    result = fw.circuit_optimizer.find_optimal_layout(n_qubits)
                    if isinstance(result, (list, tuple)) and len(result) >= n_qubits:
                        chain = list(result)[:n_qubits]
                except:
                    pass
                
                print(f"   ✅ N={n_qubits}: chaîne [{chain[0]}...{chain[-1]}]")
                config_chains[config_name] = chain
                
                for circuit_type in circuit_types_to_run:
                    if circuit_type in circuits_by_config[config_name]:
                        data = circuits_by_config[config_name][circuit_type]
                        valid_circuits.append(data["circuit"])
                        valid_metadata.append(data["metadata"])
                        
            except Exception as e:
                print(f"   ❌ N={n_qubits}: Erreur - {e}")
                skipped_configs.append(config_name)
        
        if not valid_circuits:
            print("\n❌ ERREUR: Aucun circuit valide!")
            sys.exit(1)
        
        # Transpilation
        print(f"\n⚙️ TRANSPILATION avec layout forcé...")
        
        transpiled_circuits = []
        
        for config_name in configs_to_run:
            if config_name in skipped_configs:
                continue
            
            chain = config_chains.get(config_name, list(range(CONFIGS[config_name]["n_qubits"])))
            n_qubits = CONFIGS[config_name]["n_qubits"]
            
            for circuit_type in circuit_types_to_run:
                if circuit_type not in circuits_by_config[config_name]:
                    continue
                
                qc = circuits_by_config[config_name][circuit_type]["circuit"]
                meta = circuits_by_config[config_name][circuit_type]["metadata"]
                
                try:
                    transpiled = transpile(
                        qc, 
                        fw.backend,
                        initial_layout=chain[:n_qubits],
                        optimization_level=3,
                    )
                    
                    swap_count = transpiled.count_ops().get('swap', 0)
                    depth_trans = transpiled.depth()
                    
                    print(f"   ✅ {meta['circuit_name']}: depth→{depth_trans}, {swap_count} SWAPs")
                    transpiled_circuits.append(transpiled)
                    
                except Exception as e:
                    print(f"   ❌ {meta['circuit_name']}: Erreur - {e}")
        
        print(f"\n   📋 BILAN: {len(transpiled_circuits)}/{len(valid_circuits)} circuits prêts")
        
        if not transpiled_circuits:
            print("\n❌ ERREUR: Aucun circuit transpilé!")
            sys.exit(1)
        
        # Estimation coût
        print("\n💰 Estimation du coût...")
        fw.estimate_cost(transpiled_circuits, shots=args.shots)
        
        # Exécution
        print(f"\n⏳ MEGA-RUN: {len(transpiled_circuits)} circuits...")
        results = fw.run_on_qpu(transpiled_circuits, shots=args.shots)
        
        print("\n✅ Exécution QPU terminée!")
        
        # Extraction robuste des counts
        all_counts = extract_counts_robust(results, len(transpiled_circuits))
    
    # ========================================================================
    # ANALYSE DES RÉSULTATS
    # ========================================================================
    
    if not all_counts:
        print("\n❌ ERREUR: Pas de counts extraits!")
        sys.exit(1)
    
    print("\n📊 Analyse des résultats...")
    print()
    print("=" * 70)
    print("  RÉSULTATS PAR CONFIGURATION")
    print("=" * 70)
    
    all_results = {}
    all_comparisons = {}
    all_verdicts = {}
    
    count_idx = 0
    for config_name in configs_to_run:
        if config_name in skipped_configs:
            continue
            
        n_qubits = CONFIGS[config_name]["n_qubits"]
        all_results[config_name] = {}
        
        print(f"\n📦 {config_name.upper()} (N={n_qubits})")
        print(f"   {'Type':12s} {'n_near':>10s} {'n_far':>10s} {'ratio':>8s} {'max_site':>8s}")
        print(f"   {'-'*48}")
        
        for circuit_type in circuit_types_to_run:
            if count_idx >= len(all_counts):
                print(f"   ⚠️ Plus de counts disponibles (idx={count_idx})")
                break
            
            if circuit_type not in circuits_by_config.get(config_name, {}):
                continue
            
            counts = all_counts[count_idx]
            meta = valid_metadata[count_idx] if count_idx < len(valid_metadata) else {"n_qubits": n_qubits}
            
            if not counts:
                print(f"   {circuit_type:12s}: ❌ Pas de counts")
                count_idx += 1
                continue
            
            analysis = analyze_counts(counts, n_qubits, meta)
            all_results[config_name][circuit_type] = analysis
            
            print(f"   {circuit_type:12s} {analysis['n_near_horizon']:10.4f} {analysis['n_far_avg']:10.4f} "
                  f"{analysis['localization_ratio']:8.2f} {analysis['max_site']:8d}")
            
            count_idx += 1
        
        # Comparaisons
        if all_results[config_name]:
            comparisons = compute_comparisons(all_results[config_name], config_name)
            all_comparisons[config_name] = comparisons
            
            # Afficher analyse shuffle
            if "shuffle" in all_results[config_name]:
                print(f"\n   📊 ANALYSE SHUFFLE:")
                print(f"      Standard max_site: {comparisons['std_max_site']}")
                print(f"      Shuffle max_site:  {comparisons['shuf_max_site']}")
                print(f"      Peak moved: {'✅ OUI' if comparisons['shuffle_moved_peak'] else '❌ NON'}")
            
            # Verdict
            verdict = compute_verdict(all_results[config_name], comparisons, config_name)
            all_verdicts[config_name] = verdict
    
    # ========================================================================
    # VERDICT GLOBAL
    # ========================================================================
    
    total_passed = 0
    total_checks = 0
    
    for config_name, verdict in all_verdicts.items():
        passed = sum(1 for v in verdict.values() if v)
        total = len(verdict)
        total_passed += passed
        total_checks += total
    
    percentage = (total_passed / total_checks * 100) if total_checks > 0 else 0
    
    if percentage >= 90:
        global_verdict = "GO"
    elif percentage >= 70:
        global_verdict = "GO_MARGINAL"
    else:
        global_verdict = "NO_GO"
    
    print("\n" + "=" * 70)
    print(f"  VERDICT GLOBAL: {global_verdict}")
    print(f"  Score: {total_passed}/{total_checks} = {percentage:.1f}%")
    print("=" * 70)
    
    for config_name, verdict in all_verdicts.items():
        passed = sum(1 for v in verdict.values() if v)
        total = len(verdict)
        print(f"\n   [{config_name}] {passed}/{total}")
        for check, passed_val in verdict.items():
            status = "✅" if passed_val else "❌"
            print(f"      {status} {check}")
    
    print("=" * 70)
    
    # ========================================================================
    # SAUVEGARDE
    # ========================================================================
    
    configs_str = "_".join(configs_to_run)
    shuffle_str = "_shuffle" if args.shuffle else ""
    output_filename = f"hawking_v5.2.2_S{args.trotter}_{configs_str}{shuffle_str}_{timestamp}.json"
    output_dir = "qmc_runs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    output_data = {
        "version": "5.2.2-MEGA-RUN",
        "timestamp": timestamp,
        "global_config": {
            "backend_name": args.backend,
            "mode": args.mode,
            "shots": str(args.shots),
            "s_trotter": args.trotter,
            "shuffle_included": args.shuffle,
        },
        "configs_run": configs_to_run,
        "skipped_configs": skipped_configs,
        "results": all_results,
        "comparisons": all_comparisons,
        "verdict": {
            "global": global_verdict,
            "percentage": percentage,
            "total_passed": total_passed,
            "total_checks": total_checks,
            "by_config": all_verdicts,
        },
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Résultats sauvegardés: {output_path}")
    print(f"\n✅ MEGA-RUN terminé!")
    print(f"   Verdict: {global_verdict}")


if __name__ == "__main__":
    main()
