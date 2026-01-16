#!/usr/bin/env python3
"""
HAWKING V5.2.4 - CORRECTED VALIDATION SCRIPT
=============================================
CORRECTIONS APPLIQUÉES PAR RAPPORT À V5.2.3:

1. KICK: RY avec profil Gaussien décroissant (PAS Hadamard!)
2. MESURE: Mesure directe en base Z (PAS de H final global!)
3. J PROFILE: Couplage UNIFORME (J=1.0 partout) + ω différentiel à l'horizon
4. STEPS: S = 2 (comme Paliers 7/8/9 validés)

Ces corrections reproduisent exactement la méthodologie des Paliers validés
qui ont produit les ratios 36×-83×.

Author: QMC Research Lab
Date: January 2026
Framework: qmc_quantum_framework v2.5.23
"""

# =============================================================================
# IMPORTS
# =============================================================================
from qmc_quantum_framework_v2_5_23 import (
    QMCFrameworkV2_4 as QMCFramework,
    RunMode,
)
from qiskit import QuantumCircuit
import numpy as np
from datetime import datetime
import json

# =============================================================================
# CONFIGURATION - PARAMÈTRES DES PALIERS VALIDÉS
# =============================================================================
CONFIG = {
    "project_name": "HAWKING_V5_2_4_CORRECTED",
    "backend": "ibm_fez",
    "shots": 4096,  # Comme Palier 9
    "auto_confirm": False,
    
    # Multi-scale configurations
    "configurations": [
        {"name": "Mini_S2", "N": 20, "x_horizon": 10, "S": 2},
        {"name": "Medium_S2", "N": 40, "x_horizon": 20, "S": 2},
        {"name": "Large_S2", "N": 80, "x_horizon": 40, "S": 2},
    ],
    
    # Clean zones (validated in V5.2.2)
    "clean_zones": {
        "zone_1": list(range(0, 31)),      # q0-q30
        "zone_2": list(range(120, 156)),   # q120-q155
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # PARAMÈTRES CORRIGÉS (comme Paliers validés)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Couplage J uniforme (l'horizon est créé par ω, pas par J!)
    "J_coupling": 1.0,       # Uniforme partout
    
    # Fréquence on-site ω avec DIP à l'horizon
    "omega_max": 1.0,        # Fréquence loin de l'horizon
    "omega_min": 0.1,        # Fréquence à l'horizon (DIP)
    "omega_sigma": 3.0,      # Largeur du DIP
    
    # Kick parameters (profil Gaussien)
    "kick_strengths": [0.4, 0.6, 0.8],  # Comme Paliers validés
    "kick_width": 5,         # ±2 sites autour de l'horizon
    
    # Time step
    "dt": 1.0,  # Paliers utilisaient dt=1.0
    
    # Bases de mesure
    "bases": ["Z"],  # Mesure directe en Z (pas de H final!)
    
    # Validation
    "include_shuffle": True,
}

SUCCESS_CRITERIA = {
    "min_ratio": 1.8,
    "min_peak_accuracy": 1.0,
}

# =============================================================================
# PROFIL DE FRÉQUENCE ω(x) - DIP À L'HORIZON
# =============================================================================
def compute_omega_profile(N, x_h, omega_max, omega_min, sigma):
    """
    Profil de fréquence on-site avec DIP à l'horizon.
    
    ω(x) = ω_max - (ω_max - ω_min) * exp(-(x - x_h)² / 2σ²)
    
    → ω = ω_min à l'horizon (excitations piégées)
    → ω = ω_max loin (excitations se propagent)
    """
    omega = np.zeros(N)
    for i in range(N):
        dip = np.exp(-(i - x_h)**2 / (2 * sigma**2))
        omega[i] = omega_max - (omega_max - omega_min) * dip
    return omega

# =============================================================================
# CIRCUIT CORRIGÉ - REPRODUCTION EXACTE DES PALIERS VALIDÉS
# =============================================================================
def create_hawking_circuit_v524(
    N: int,
    x_horizon: int,
    S: int,
    kick_strength: float,
    J: float = 1.0,
    omega_profile: np.ndarray = None,
    dt: float = 1.0,
    kick_width: int = 5,
):
    """
    Circuit HAWKING CORRIGÉ - Reproduction exacte des Paliers validés.
    
    DIFFÉRENCES CLÉS par rapport à V5.2.3:
    
    1. KICK: RY(2*θ) avec θ Gaussien décroissant (PAS Hadamard!)
       → Crée excitation LOCALISÉE à l'horizon
       
    2. MESURE: Base Z directe (PAS de H final!)
       → n(x) = P(|1⟩) = vraie densité d'excitation
       
    3. COUPLAGE: J uniforme, ω avec DIP à l'horizon
       → L'horizon est créé par le DIP de ω, pas par J
    
    Protocole:
    ---------
    1. Kick localisé: RY avec amplitude Gaussienne décroissante
    2. S steps de Trotter:
       a. RZ(ω_i * dt) pour on-site (évolution de phase)
       b. RXX + RYY pour couplage XY
    3. Mesure directe en base Z (PAS de H final!)
    """
    qc = QuantumCircuit(N, N)
    
    # Profil ω si non fourni
    if omega_profile is None:
        omega_profile = compute_omega_profile(
            N, x_horizon, 
            CONFIG["omega_max"], 
            CONFIG["omega_min"], 
            CONFIG["omega_sigma"]
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1: KICK LOCALISÉ (RY avec profil Gaussien)
    # ═══════════════════════════════════════════════════════════════════════
    # CORRECTION: Utiliser RY, PAS Hadamard!
    # Le kick crée une excitation qui décroît avec la distance à l'horizon
    
    kick_start = max(0, x_horizon - kick_width // 2)
    kick_end = min(N, x_horizon + kick_width // 2 + 1)
    
    for i in range(kick_start, kick_end):
        distance = abs(i - x_horizon)
        # Amplitude du kick décroît avec la distance (Gaussien)
        kick_angle = kick_strength * np.exp(-distance / 2)
        # RY(2θ): |0⟩ → cos(θ)|0⟩ + sin(θ)|1⟩
        qc.ry(2 * kick_angle, i)
    
    qc.barrier()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2: ÉVOLUTION TROTTER (S steps)
    # ═══════════════════════════════════════════════════════════════════════
    for step in range(S):
        # --- 2a. Termes on-site: exp(-i ω_i n_i dt) → RZ(ω_i * dt) ---
        # Le DIP de ω à l'horizon piège les excitations
        for i in range(N):
            qc.rz(omega_profile[i] * dt, i)
        
        qc.barrier()
        
        # --- 2b. Couplage XY (brickwork even-odd) ---
        # J uniforme, l'horizon est créé par ω, pas par J
        
        # Even bonds: (0,1), (2,3), (4,5), ...
        for i in range(0, N - 1, 2):
            theta = J * dt
            qc.rxx(theta, i, i + 1)
            qc.ryy(theta, i, i + 1)
        
        # Odd bonds: (1,2), (3,4), (5,6), ...
        for i in range(1, N - 1, 2):
            theta = J * dt
            qc.rxx(theta, i, i + 1)
            qc.ryy(theta, i, i + 1)
        
        qc.barrier()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3: MESURE DIRECTE EN BASE Z
    # ═══════════════════════════════════════════════════════════════════════
    # CORRECTION: PAS de Hadamard final!
    # On mesure directement en base Z pour obtenir n(x) = P(|1⟩)
    
    # PAS DE: for i in range(N): qc.h(i)  ← C'ÉTAIT L'ERREUR!
    
    qc.measure(range(N), range(N))
    
    return qc

# =============================================================================
# ANALYSE
# =============================================================================
def compute_excitation_density(counts, N):
    """
    Calcule la densité d'excitation n(x) = ⟨n_x⟩ = P(qubit_x = |1⟩).
    
    En base Z sans rotation finale:
    - |0⟩ = pas d'excitation
    - |1⟩ = excitation présente
    
    n(x) = (nombre de shots où qubit x = 1) / (total shots)
    """
    n = np.zeros(N)
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Qiskit convention: bitstring[0] = qubit N-1, bitstring[-1] = qubit 0
        # On inverse pour avoir bits[i] = qubit i
        bits = bitstring[::-1]
        for i in range(min(N, len(bits))):
            if bits[i] == '1':
                n[i] += count
    
    return n / total

def compute_metrics(n, x_horizon, near_range=3):
    """Calcule le ratio de localisation et autres métriques."""
    N = len(n)
    
    # Zone proche: horizon ± near_range
    near_indices = range(max(0, x_horizon - near_range), 
                        min(N, x_horizon + near_range + 1))
    n_near = np.mean([n[i] for i in near_indices])
    
    # Zone lointaine: tout le reste
    far_indices = [i for i in range(N) if abs(i - x_horizon) > near_range]
    n_far = np.mean([n[i] for i in far_indices]) if far_indices else 1e-10
    
    # Ratio
    ratio = n_near / max(n_far, 1e-10)
    
    # Position du max
    max_site = int(np.argmax(n))
    
    return {
        "n_near": n_near,
        "n_far": n_far,
        "ratio": ratio,
        "max_site": max_site,
        "peak_at_horizon": (max_site == x_horizon),
    }

def apply_shuffle(counts, N, seed=42):
    """Applique une permutation aléatoire aux indices de qubits."""
    np.random.seed(seed)
    mapping = np.random.permutation(N)
    
    shuffled = {}
    for bitstring, count in counts.items():
        bits = list(bitstring[::-1])[:N]
        shuffled_bits = ['0'] * N
        for i, m in enumerate(mapping):
            if i < len(bits):
                shuffled_bits[m] = bits[i]
        new_string = ''.join(shuffled_bits[::-1])
        shuffled[new_string] = shuffled.get(new_string, 0) + count
    
    return shuffled

# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================
def main():
    print("="*70)
    print("HAWKING V5.2.4 - CORRECTED VALIDATION")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Backend: {CONFIG['backend']}")
    print()
    print("CORRECTIONS APPLIQUÉES:")
    print("  1. KICK: RY Gaussien (pas Hadamard)")
    print("  2. MESURE: Base Z directe (pas de H final)")
    print("  3. J: Uniforme + ω DIP à l'horizon")
    print("  4. S = 2 (comme Paliers validés)")
    print("="*70)
    
    # 1. Framework
    print("\n[1. FRAMEWORK]")
    fw = QMCFramework(
        project=CONFIG["project_name"],
        backend_name=CONFIG["backend"],
        shots=CONFIG["shots"],
        auto_confirm=False,
    )
    fw.initialize(mode=RunMode.QPU)
    
    # 2. Connect
    print("\n[2. BACKEND CONNECTION]")
    fw.connect()
    
    # 3. Calibration
    print("\n[3. CALIBRATION]")
    topology = fw.analyze_calibration()
    
    # 4. Exécution
    print("\n[4. CIRCUIT GENERATION & EXECUTION]")
    all_results = []
    
    for config in CONFIG["configurations"]:
        N = config["N"]
        x_h = config["x_horizon"]
        S = config["S"]
        
        print(f"\n{'='*60}")
        print(f"CONFIG: {config['name']} (N={N}, x_h={x_h}, S={S})")
        print(f"{'='*60}")
        
        # Profil ω
        omega = compute_omega_profile(
            N, x_h,
            CONFIG["omega_max"],
            CONFIG["omega_min"],
            CONFIG["omega_sigma"]
        )
        
        # Test avec plusieurs kick strengths
        for kick in CONFIG["kick_strengths"]:
            print(f"\n--- Kick strength κ = {kick} ---")
            
            # Créer circuit
            circuit = create_hawking_circuit_v524(
                N=N,
                x_horizon=x_h,
                S=S,
                kick_strength=kick,
                J=CONFIG["J_coupling"],
                omega_profile=omega,
                dt=CONFIG["dt"],
                kick_width=CONFIG["kick_width"],
            )
            
            print(f"Circuit depth: {circuit.depth()}")
            
            # Estimer coût
            estimate = fw.estimate_cost([circuit], shots=CONFIG["shots"])
            print(f"Est. QPU time: {estimate.get('estimated_qpu_time', {}).get('expected_str', 'N/A')}")
            
            # Exécuter
            print("Executing on QPU...")
            results = fw.run_on_qpu([circuit], shots=CONFIG["shots"])
            
            # Extraire counts
            if isinstance(results, list) and len(results) > 0:
                result_data = results[0]
                if isinstance(result_data, dict):
                    counts = result_data.get('counts', {})
                else:
                    counts = result_data.get_counts() if hasattr(result_data, 'get_counts') else {}
            else:
                counts = results.get_counts() if hasattr(results, 'get_counts') else {}
            
            # Analyser
            n = compute_excitation_density(counts, N)
            metrics = compute_metrics(n, x_h)
            
            # Afficher profil
            print(f"\nProfil d'excitation n(x):")
            max_n = max(n) if max(n) > 0 else 1
            for i in range(N):
                bar = '█' * int(n[i] / max_n * 30)
                marker = " ◀─ HORIZON" if i == x_h else ""
                peak = " ★ PEAK" if i == metrics['max_site'] else ""
                print(f"  Site {i:2d}: {n[i]:.4f} {bar}{marker}{peak}")
            
            print(f"\n📊 MÉTRIQUES:")
            print(f"  max_site = {metrics['max_site']} (attendu: {x_h})")
            print(f"  MATCH: {'✅' if metrics['peak_at_horizon'] else '❌'}")
            print(f"  Ratio = {metrics['ratio']:.2f}×")
            print(f"  GO: {'✅' if metrics['ratio'] > 1.8 else '❌'}")
            
            # Shuffle
            if CONFIG["include_shuffle"]:
                shuffled = apply_shuffle(counts, N)
                n_shuf = compute_excitation_density(shuffled, N)
                m_shuf = compute_metrics(n_shuf, x_h)
                
                print(f"\n🔀 SHUFFLE CONTROL:")
                print(f"  max_site after shuffle: {m_shuf['max_site']}")
                print(f"  Peak moved: {'✅ YES' if m_shuf['max_site'] != metrics['max_site'] else '❌ NO'}")
                print(f"  Shuffle ratio: {m_shuf['ratio']:.2f}×")
                
                metrics['shuffle_max_site'] = m_shuf['max_site']
                metrics['shuffle_ratio'] = m_shuf['ratio']
                metrics['peak_moved'] = (m_shuf['max_site'] != metrics['max_site'])
            
            # Stocker
            all_results.append({
                "config": config['name'],
                "N": N,
                "x_horizon": x_h,
                "S": S,
                "kick": kick,
                **metrics,
            })
    
    # 5. Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ GLOBAL")
    print("="*70)
    
    print(f"\n{'Config':<15} {'κ':<6} {'Ratio':<10} {'Peak@h':<8} {'Moved':<8}")
    print("-"*50)
    for r in all_results:
        print(f"{r['config']:<15} {r['kick']:<6.1f} {r['ratio']:<10.2f} "
              f"{'✅' if r['peak_at_horizon'] else '❌':<8} "
              f"{'✅' if r.get('peak_moved', False) else '❌':<8}")
    
    # Sauvegarder
    output_file = f"HAWKING_V5_2_4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "HAWKING_V5_2_4_CORRECTED",
            "corrections": [
                "KICK: RY Gaussian (not Hadamard)",
                "MEASUREMENT: Direct Z basis (no final H)",
                "J: Uniform + omega DIP at horizon",
                "S = 2 (as validated Paliers)",
            ],
            "results": all_results,
            "date": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    
    print(f"\n📁 Résultats sauvegardés: {output_file}")
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return all_results

if __name__ == "__main__":
    main()
