#!/usr/bin/env python3
"""
HAWKING V5.2.5 - FLUX MEASUREMENT ON LINKS
==========================================
CORRECTION CRITIQUE: Mesure du FLUX F(x) = ⟨XX⟩ + ⟨YY⟩ sur les LIENS
                     (PAS la densité n(x) sur les sites!)

C'est LA différence fondamentale avec V5.2.4 qui explique pourquoi
les Paliers validés obtiennent des ratios 36-83× alors que V5.2.4
n'obtenait que 1.5-2.6×.

MÉTHODOLOGIE DES PALIERS VALIDÉS:
- Pour chaque LIEN (i, i+1), mesurer:
  * ⟨XX⟩ = ⟨X_i X_{i+1}⟩ via rotation H puis mesure ZZ
  * ⟨YY⟩ = ⟨Y_i Y_{i+1}⟩ via rotation S†H puis mesure ZZ
- F(link) = ⟨XX⟩ + ⟨YY⟩ = flux XY à travers le lien
- Ratio = F_horizon / F_far

CIRCUITS GÉNÉRÉS:
- Pour chaque lien d'intérêt: 2 circuits (base XX et base YY)
- Mesure PARTIELLE: seulement 2 qubits mesurés par circuit
- Tous les circuits dans UN SEUL JOB pour efficacité

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
# CONFIGURATION
# =============================================================================
CONFIG = {
    "project_name": "HAWKING_V5_2_5_FLUX",
    "backend": "ibm_fez",
    "shots": 4096,
    "auto_confirm": False,
    
    # Configuration principale (taille raisonnable pour test)
    "N": 40,              # Nombre de qubits
    "x_horizon": 20,      # Position de l'horizon (lien 20)
    "S": 2,               # Trotter steps
    "kick_strength": 0.6, # Force du kick (valeur médiane)
    
    # Couplage J uniforme
    "J_coupling": 1.0,
    
    # Profil ω avec DIP à l'horizon  
    "omega_max": 1.0,
    "omega_min": 0.1,
    "omega_sigma": 3.0,
    
    # Kick parameters
    "kick_width": 5,
    "dt": 1.0,
    
    # Liens à scanner
    # Horizon ± 5 liens + 2 liens FAR aux extrémités
    "near_range": 5,      # Liens horizon-5 à horizon+5
    "far_links": [2, 37], # Liens très éloignés de l'horizon
}

# =============================================================================
# PROFIL DE FRÉQUENCE ω(x)
# =============================================================================
def compute_omega_profile(N, x_h, omega_max, omega_min, sigma):
    """Profil ω avec DIP à l'horizon."""
    omega = np.zeros(N)
    for i in range(N):
        dip = np.exp(-(i - x_h)**2 / (2 * sigma**2))
        omega[i] = omega_max - (omega_max - omega_min) * dip
    return omega

# =============================================================================
# CIRCUIT DE MESURE DU FLUX SUR UN LIEN
# =============================================================================
def create_flux_circuit(
    N: int,
    x_horizon: int,
    target_link: int,
    basis: str,  # 'XX' ou 'YY'
    S: int,
    kick_strength: float,
    J: float,
    omega_profile: np.ndarray,
    dt: float,
    kick_width: int,
):
    """
    Crée un circuit pour mesurer le FLUX ⟨XX⟩ ou ⟨YY⟩ sur un lien spécifique.
    
    DIFFÉRENCE CRITIQUE avec V5.2.4:
    - V5.2.4: Mesure n(x) = P(|1⟩) sur TOUS les sites → ratio ~2×
    - V5.2.5: Mesure F = ⟨XX⟩+⟨YY⟩ sur UN lien → ratio ~50-80× (comme Paliers)
    
    Le flux XY représente le COURANT d'énergie à travers le lien.
    À l'horizon, ce flux est maximal car les excitations s'accumulent.
    
    Parameters:
    -----------
    target_link : int
        Indice du lien (i, i+1) à mesurer. Ex: link=10 → mesure (q10, q11)
    basis : str
        'XX' pour mesurer ⟨X_i X_{i+1}⟩
        'YY' pour mesurer ⟨Y_i Y_{i+1}⟩
    """
    # Qubits du lien cible
    q1, q2 = target_link, target_link + 1
    
    # Circuit avec N qubits mais seulement 2 bits classiques (mesure partielle!)
    qc = QuantumCircuit(N, 2)
    qc.name = f"Flux_L{target_link}_{basis}"
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1: KICK LOCALISÉ (RY Gaussien) - Identique à V5.2.4
    # ═══════════════════════════════════════════════════════════════════════
    kick_start = max(0, x_horizon - kick_width // 2)
    kick_end = min(N, x_horizon + kick_width // 2 + 1)
    
    for i in range(kick_start, kick_end):
        distance = abs(i - x_horizon)
        kick_angle = kick_strength * np.exp(-distance / 2)
        qc.ry(2 * kick_angle, i)
    
    qc.barrier()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2: ÉVOLUTION TROTTER (S steps) - Identique à V5.2.4
    # ═══════════════════════════════════════════════════════════════════════
    for step in range(S):
        # On-site: RZ(ω_i * dt)
        for i in range(N):
            qc.rz(omega_profile[i] * dt, i)
        
        # Couplage XY (brickwork)
        # Even bonds
        for i in range(0, N - 1, 2):
            theta = J * dt
            qc.rxx(theta, i, i + 1)
            qc.ryy(theta, i, i + 1)
        
        # Odd bonds
        for i in range(1, N - 1, 2):
            theta = J * dt
            qc.rxx(theta, i, i + 1)
            qc.ryy(theta, i, i + 1)
    
    qc.barrier()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3: ROTATION DE BASE (SEULEMENT sur les 2 qubits du lien!)
    # ═══════════════════════════════════════════════════════════════════════
    # C'est ICI la différence critique!
    # On ne touche que q1 et q2, pas tous les N qubits
    
    if basis == 'XX':
        # Pour mesurer ⟨XX⟩: appliquer H sur les 2 qubits
        # H|+⟩ = |0⟩, H|-⟩ = |1⟩
        # Donc ⟨XX⟩ = P(00) + P(11) - P(01) - P(10) après H
        qc.h(q1)
        qc.h(q2)
    elif basis == 'YY':
        # Pour mesurer ⟨YY⟩: appliquer S†H sur les 2 qubits
        # S†H transforme la base Y en base Z
        qc.sdg(q1)
        qc.sdg(q2)
        qc.h(q1)
        qc.h(q2)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 4: MESURE PARTIELLE (seulement 2 qubits!)
    # ═══════════════════════════════════════════════════════════════════════
    # C'est une autre différence critique!
    # On mesure SEULEMENT les 2 qubits du lien, pas tous les N
    
    qc.measure(q1, 0)
    qc.measure(q2, 1)
    
    return qc

# =============================================================================
# CALCUL DU FLUX À PARTIR DES COUNTS
# =============================================================================
def compute_flux_from_counts(counts):
    """
    Calcule ⟨ZZ⟩ = P(00) + P(11) - P(01) - P(10) à partir des counts.
    
    Après la rotation de base (H pour XX, S†H pour YY), la mesure ZZ
    donne directement ⟨XX⟩ ou ⟨YY⟩ selon la rotation appliquée.
    
    Returns:
    --------
    expectation : float
        Valeur de ⟨ZZ⟩ ∈ [-1, +1]
    """
    total = sum(counts.values())
    
    # Probabilités (attention: Qiskit inverse l'ordre des bits)
    # counts['ab'] où a=bit1, b=bit0
    p_00 = counts.get('00', 0) / total
    p_01 = counts.get('01', 0) / total
    p_10 = counts.get('10', 0) / total
    p_11 = counts.get('11', 0) / total
    
    # ⟨ZZ⟩ = P(même parité) - P(parité différente)
    expectation = (p_00 + p_11) - (p_01 + p_10)
    
    return expectation, {"p_00": p_00, "p_01": p_01, "p_10": p_10, "p_11": p_11}

# =============================================================================
# GÉNÉRATION DE TOUS LES CIRCUITS
# =============================================================================
def generate_all_circuits(config):
    """
    Génère tous les circuits pour scanner les liens.
    
    Pour chaque lien: 2 circuits (XX et YY)
    Total = (2*near_range + 1 + len(far_links)) * 2 circuits
    """
    N = config["N"]
    x_h = config["x_horizon"]
    near_range = config["near_range"]
    far_links = config["far_links"]
    
    # Profil ω
    omega = compute_omega_profile(
        N, x_h,
        config["omega_max"],
        config["omega_min"],
        config["omega_sigma"]
    )
    
    # Liens à scanner
    near_links = list(range(x_h - near_range, x_h + near_range + 1))
    # Filtrer les liens valides (0 ≤ link < N-1)
    near_links = [l for l in near_links if 0 <= l < N - 1]
    
    all_links = near_links + [l for l in far_links if 0 <= l < N - 1]
    
    circuits = []
    circuit_info = []
    
    print(f"\n📐 GÉNÉRATION DES CIRCUITS")
    print(f"   N = {N} qubits")
    print(f"   Horizon = lien {x_h}")
    print(f"   Liens NEAR: {near_links}")
    print(f"   Liens FAR: {far_links}")
    print(f"   Total liens: {len(all_links)}")
    print(f"   Bases: XX, YY")
    print(f"   Total circuits: {len(all_links) * 2}")
    
    for link in all_links:
        for basis in ['XX', 'YY']:
            qc = create_flux_circuit(
                N=N,
                x_horizon=x_h,
                target_link=link,
                basis=basis,
                S=config["S"],
                kick_strength=config["kick_strength"],
                J=config["J_coupling"],
                omega_profile=omega,
                dt=config["dt"],
                kick_width=config["kick_width"],
            )
            circuits.append(qc)
            circuit_info.append({
                "link": link,
                "basis": basis,
                "circuit_name": qc.name,
                "is_horizon": (link == x_h),
                "is_near": (link in near_links),
                "is_far": (link in far_links),
            })
    
    return circuits, circuit_info, all_links

# =============================================================================
# ANALYSE DES RÉSULTATS
# =============================================================================
def analyze_flux_results(results, circuit_info, config):
    """
    Analyse les résultats et calcule le flux F(link) = ⟨XX⟩ + ⟨YY⟩.
    """
    x_h = config["x_horizon"]
    near_range = config["near_range"]
    
    # Organiser les résultats par lien
    flux_by_link = {}
    
    for i, info in enumerate(circuit_info):
        link = info["link"]
        basis = info["basis"]
        
        # Extraire counts
        if isinstance(results[i], dict):
            counts = results[i].get('counts', results[i])
        else:
            counts = results[i].get_counts() if hasattr(results[i], 'get_counts') else {}
        
        # Calculer expectation
        expectation, probs = compute_flux_from_counts(counts)
        
        if link not in flux_by_link:
            flux_by_link[link] = {"XX": None, "YY": None, "info": info}
        
        flux_by_link[link][basis] = expectation
        flux_by_link[link][f"{basis}_probs"] = probs
    
    # Calculer F(link) = ⟨XX⟩ + ⟨YY⟩ pour chaque lien
    print("\n" + "="*70)
    print("RÉSULTATS: FLUX F(link) = ⟨XX⟩ + ⟨YY⟩")
    print("="*70)
    
    flux_profile = {}
    
    for link in sorted(flux_by_link.keys()):
        data = flux_by_link[link]
        xx = data["XX"]
        yy = data["YY"]
        
        if xx is not None and yy is not None:
            F = xx + yy  # Flux total XY
            flux_profile[link] = {
                "XX": xx,
                "YY": yy,
                "F": F,
                "is_horizon": data["info"]["is_horizon"],
                "is_near": data["info"]["is_near"],
                "is_far": data["info"]["is_far"],
            }
    
    # Afficher le profil
    print(f"\n{'Link':<6} {'⟨XX⟩':>10} {'⟨YY⟩':>10} {'F=XX+YY':>12} {'Type':<12}")
    print("-"*55)
    
    F_max = max(abs(d["F"]) for d in flux_profile.values()) if flux_profile else 1
    
    for link in sorted(flux_profile.keys()):
        d = flux_profile[link]
        
        # Type indicator
        if d["is_horizon"]:
            type_str = "★ HORIZON"
        elif d["is_far"]:
            type_str = "○ FAR"
        else:
            type_str = "● NEAR"
        
        # Visual bar
        bar_len = int(abs(d["F"]) / F_max * 20) if F_max > 0 else 0
        bar = '█' * bar_len
        
        print(f"{link:<6} {d['XX']:>+10.4f} {d['YY']:>+10.4f} {d['F']:>+12.4f} {type_str:<12} {bar}")
    
    # Calculer les métriques
    print("\n" + "="*70)
    print("MÉTRIQUES DE LOCALISATION")
    print("="*70)
    
    # F à l'horizon
    F_horizon = flux_profile.get(x_h, {}).get("F", 0)
    
    # F moyen dans la zone NEAR (hors horizon)
    near_F = [d["F"] for link, d in flux_profile.items() 
              if d["is_near"] and not d["is_horizon"]]
    F_near_avg = np.mean(near_F) if near_F else 0
    
    # F moyen dans la zone FAR
    far_F = [d["F"] for link, d in flux_profile.items() if d["is_far"]]
    F_far_avg = np.mean(far_F) if far_F else 0.001  # Éviter division par 0
    
    # Ratios
    ratio_horizon_far = abs(F_horizon) / max(abs(F_far_avg), 0.001)
    ratio_near_far = abs(F_near_avg) / max(abs(F_far_avg), 0.001)
    
    # Position du max
    max_link = max(flux_profile.keys(), key=lambda l: abs(flux_profile[l]["F"]))
    max_F = flux_profile[max_link]["F"]
    
    print(f"\n  F_horizon (link {x_h}):     {F_horizon:+.4f}")
    print(f"  F_near (moyenne):          {F_near_avg:+.4f}")
    print(f"  F_far (moyenne):           {F_far_avg:+.4f}")
    print(f"\n  Max |F| at link:           {max_link} (F = {max_F:+.4f})")
    print(f"  Peak at horizon:           {'✅ YES' if max_link == x_h else f'❌ NO (offset = {max_link - x_h:+d})'}")
    print(f"\n  📊 RATIO |F_horizon| / |F_far| = {ratio_horizon_far:.2f}×")
    print(f"  📊 RATIO |F_near| / |F_far|    = {ratio_near_far:.2f}×")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if ratio_horizon_far >= 10:
        verdict = "GO_HEADLINE ★★★"
        status = "✅"
    elif ratio_horizon_far >= 3:
        verdict = "GO ✅"
        status = "✅"
    elif ratio_horizon_far >= 1.8:
        verdict = "GO_MARGINAL ⚠️"
        status = "⚠️"
    else:
        verdict = "NO-GO ❌"
        status = "❌"
    
    print(f"\n  {status} VERDICT: {verdict}")
    print(f"     Ratio = {ratio_horizon_far:.2f}× (seuil GO = 1.8×, HEADLINE = 10×)")
    
    # Comparaison avec V5.2.4
    print("\n" + "-"*70)
    print("COMPARAISON V5.2.4 (densité n(x)) vs V5.2.5 (flux F(link))")
    print("-"*70)
    print(f"  V5.2.4 ratio max: ~2.5× (densité d'excitation)")
    print(f"  V5.2.5 ratio:     {ratio_horizon_far:.2f}× (flux XY)")
    print(f"  Paliers validés:  36-83× (même méthodologie)")
    
    if ratio_horizon_far > 5:
        print(f"\n  ✅ V5.2.5 CONFIRME que la mesure du FLUX est la bonne méthodologie!")
    else:
        print(f"\n  ⚠️ Ratio encore inférieur aux Paliers - vérifier les paramètres")
    
    return {
        "flux_profile": flux_profile,
        "F_horizon": F_horizon,
        "F_near_avg": F_near_avg,
        "F_far_avg": F_far_avg,
        "ratio_horizon_far": ratio_horizon_far,
        "ratio_near_far": ratio_near_far,
        "max_link": max_link,
        "max_F": max_F,
        "peak_at_horizon": (max_link == x_h),
        "verdict": verdict,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("HAWKING V5.2.5 - FLUX MEASUREMENT ON LINKS")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Backend: {CONFIG['backend']}")
    print()
    print("MÉTHODOLOGIE:")
    print("  - Mesure du FLUX F(link) = ⟨XX⟩ + ⟨YY⟩ sur chaque lien")
    print("  - Mesure PARTIELLE (2 qubits par circuit)")
    print("  - Scan des liens: horizon ± near_range + liens FAR")
    print("  - C'est la vraie méthodologie des Paliers 7/8/9!")
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
    
    # 4. Générer tous les circuits
    print("\n[4. CIRCUIT GENERATION]")
    circuits, circuit_info, all_links = generate_all_circuits(CONFIG)
    
    # Afficher stats
    print(f"\n   Total circuits générés: {len(circuits)}")
    print(f"   Profondeur circuit: {circuits[0].depth()}")
    
    # 5. Estimer coût
    print("\n[5. COST ESTIMATION]")
    estimate = fw.estimate_cost(circuits, shots=CONFIG["shots"])
    
    # 6. Exécuter en UN SEUL JOB
    print("\n[6. EXECUTION (1 JOB, ALL CIRCUITS)]")
    print(f"   Circuits: {len(circuits)}")
    print(f"   Shots per circuit: {CONFIG['shots']}")
    print(f"   Total measurements: {len(circuits) * CONFIG['shots']:,}")
    
    results = fw.run_on_qpu(circuits, shots=CONFIG["shots"])
    
    # 7. Analyser
    print("\n[7. ANALYSIS]")
    analysis = analyze_flux_results(results, circuit_info, CONFIG)
    
    # 8. Sauvegarder
    print("\n[8. SAVE RESULTS]")
    output_data = {
        "experiment": "HAWKING_V5_2_5_FLUX",
        "methodology": "Flux F(link) = <XX> + <YY> measurement on links",
        "config": {k: v for k, v in CONFIG.items() if not callable(v)},
        "results": {
            "flux_profile": {str(k): v for k, v in analysis["flux_profile"].items()},
            "F_horizon": analysis["F_horizon"],
            "F_near_avg": analysis["F_near_avg"],
            "F_far_avg": analysis["F_far_avg"],
            "ratio_horizon_far": analysis["ratio_horizon_far"],
            "ratio_near_far": analysis["ratio_near_far"],
            "max_link": analysis["max_link"],
            "max_F": analysis["max_F"],
            "peak_at_horizon": analysis["peak_at_horizon"],
            "verdict": analysis["verdict"],
        },
        "comparison": {
            "V5.2.4_density_ratio": "~2.5x",
            "V5.2.5_flux_ratio": f"{analysis['ratio_horizon_far']:.2f}x",
            "Paliers_validated_ratio": "36-83x",
        },
        "date": datetime.now().isoformat(),
    }
    
    output_file = f"HAWKING_V5_2_5_FLUX_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"   📁 Results saved: {output_file}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\n   🎯 RATIO OBTENU: {analysis['ratio_horizon_far']:.2f}×")
    print(f"   📊 VERDICT: {analysis['verdict']}")
    print("\n" + "="*70)
    
    return analysis

if __name__ == "__main__":
    main()
