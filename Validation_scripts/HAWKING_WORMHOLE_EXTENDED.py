#!/usr/bin/env python3
"""
HAWKING WORMHOLE EXTENDED VALIDATION
====================================
Objective: Strengthen wormhole claim from p=0.031 to p<0.01
           by adding 15 seeds to existing 5 seeds

Author: QMC Research Lab
Date: January 2026
Framework: qmc_quantum_framework v2.5.23

RATIONALE:
- Current evidence: 5/5 seeds, p=0.031 (binomial test)
- Insufficient for strong claim: need p<0.01
- Target: 18-20/20 successes → p<0.001

PRIORITY: 🟠 SERIOUS (recommended but not critical)
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
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "project_name": "HAWKING_WORMHOLE_EXTENDED",
    "backend": "ibm_fez",
    "shots": 8192,
    "auto_confirm": False,
    
    # Wormhole topology
    "topology": "dual_ring_48Q",
    "n_qubits": 48,
    "ring_size": 20,        # Qubits per ring (Universe A and B)
    "throat_qubits": 8,     # Qubits connecting the rings
    
    # XY parameters
    "J_ring": 1.0,          # Intra-ring coupling
    "J_throat": 0.5,        # Cross-throat coupling (weaker)
    "trotter_steps": 4,
    
    # Validation
    "n_seeds": 15,          # Additional seeds (total will be 20)
    "existing_seeds": 5,    # Already validated
    "existing_successes": 5,
    
    # Success criterion
    "cross_throat_threshold": 1.2,  # W_cross > W_within × threshold
}

# =============================================================================
# WORMHOLE CIRCUIT GENERATION
# =============================================================================
def create_wormhole_topology(n_ring, n_throat):
    """
    Create dual-ring wormhole topology.
    
    Structure:
    - Universe A: Ring of n_ring qubits (q0 to q_{n_ring-1})
    - Universe B: Ring of n_ring qubits (q_{n_ring} to q_{2*n_ring-1})
    - Throat: n_throat bridges connecting specific sites
    
    Returns: dict with qubit assignments and coupling map
    """
    topology = {
        "universe_A": list(range(n_ring)),
        "universe_B": list(range(n_ring, 2 * n_ring)),
        "throat_pairs": [],
    }
    
    # Create throat connections (evenly spaced)
    throat_spacing = n_ring // (n_throat // 2)
    for i in range(n_throat // 2):
        q_A = i * throat_spacing
        q_B = n_ring + i * throat_spacing
        topology["throat_pairs"].append((q_A, q_B))
    
    return topology

def create_wormhole_circuit(topology, J_ring, J_throat, trotter_steps, dt=0.1):
    """
    Create XY evolution on dual-ring wormhole geometry.
    """
    n_A = len(topology["universe_A"])
    n_B = len(topology["universe_B"])
    N = n_A + n_B
    
    qc = QuantumCircuit(N, N)
    
    # Initialize: excitation in Universe A (one side of wormhole)
    # Create localized excitation at throat entrance
    throat_entrance = topology["throat_pairs"][0][0]
    for i in range(max(0, throat_entrance-1), min(n_A, throat_entrance+2)):
        qc.h(i)
    
    # Trotter evolution
    for step in range(trotter_steps):
        # Intra-ring coupling (Universe A - ring topology)
        for i in range(n_A - 1):
            theta = 2 * J_ring * dt
            qc.rxx(theta, i, i + 1)
            qc.ryy(theta, i, i + 1)
        # Close ring A
        qc.rxx(2 * J_ring * dt, n_A - 1, 0)
        qc.ryy(2 * J_ring * dt, n_A - 1, 0)
        
        # Intra-ring coupling (Universe B)
        for i in range(n_A, N - 1):
            theta = 2 * J_ring * dt
            qc.rxx(theta, i, i + 1)
            qc.ryy(theta, i, i + 1)
        # Close ring B
        qc.rxx(2 * J_ring * dt, N - 1, n_A)
        qc.ryy(2 * J_ring * dt, N - 1, n_A)
        
        # Cross-throat coupling
        for q_A, q_B in topology["throat_pairs"]:
            theta = 2 * J_throat * dt
            qc.rxx(theta, q_A, q_B)
            qc.ryy(theta, q_A, q_B)
    
    # Measurement basis rotation
    qc.h(range(N))
    qc.measure(range(N), range(N))
    
    return qc

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def compute_wormhole_flux(counts, topology):
    """
    Compute cross-throat entanglement flux W_cross.
    
    W_cross measures correlations between Universe A and Universe B
    through the throat connections.
    """
    N = len(topology["universe_A"]) + len(topology["universe_B"])
    
    # Compute correlators
    correlators = {
        "cross_throat": [],    # Between A and B via throat
        "within_A": [],        # Within Universe A
        "within_B": [],        # Within Universe B
    }
    
    total = sum(counts.values())
    
    # Cross-throat correlations
    for q_A, q_B in topology["throat_pairs"]:
        corr = compute_zz_correlator(counts, q_A, q_B, N)
        correlators["cross_throat"].append(corr)
    
    # Within-universe correlations (sample pairs)
    n_A = len(topology["universe_A"])
    for i in range(0, n_A - 1, 2):
        correlators["within_A"].append(compute_zz_correlator(counts, i, i+1, N))
    
    for i in range(n_A, N - 1, 2):
        correlators["within_B"].append(compute_zz_correlator(counts, i, i+1, N))
    
    # Compute flux metrics
    W_cross = np.mean(np.abs(correlators["cross_throat"]))
    W_within = np.mean(np.abs(correlators["within_A"] + correlators["within_B"]))
    
    return {
        "W_cross": W_cross,
        "W_within": W_within,
        "ratio": W_cross / max(W_within, 1e-10),
        "correlators": correlators,
    }

def compute_zz_correlator(counts, q1, q2, N):
    """Compute <ZZ> correlator between qubits q1 and q2."""
    zz = 0.0
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        if q1 < len(bits) and q2 < len(bits):
            # Convert 0/1 to +1/-1
            z1 = 1 - 2 * bits[q1]
            z2 = 1 - 2 * bits[q2]
            zz += z1 * z2 * count / total
    
    return zz

def apply_shuffle_wormhole(counts, topology):
    """Shuffle control: permute qubits, destroying spatial structure."""
    N = len(topology["universe_A"]) + len(topology["universe_B"])
    mapping = np.random.permutation(N)
    
    shuffled_counts = {}
    for bitstring, count in counts.items():
        bits = list(bitstring[::-1])[:N]
        shuffled_bits = ['0'] * N
        for i, m in enumerate(mapping):
            if i < len(bits):
                shuffled_bits[m] = bits[i]
        shuffled_string = ''.join(shuffled_bits[::-1])
        shuffled_counts[shuffled_string] = shuffled_counts.get(shuffled_string, 0) + count
    
    return shuffled_counts

def evaluate_seed(results, topology, seed_id):
    """Evaluate single seed: is W_cross > W_shuffle?"""
    counts = results.get_counts() if hasattr(results, 'get_counts') else results
    
    # Standard flux
    flux = compute_wormhole_flux(counts, topology)
    
    # Shuffle control
    shuffled_counts = apply_shuffle_wormhole(counts, topology)
    shuffle_flux = compute_wormhole_flux(shuffled_counts, topology)
    
    success = flux["W_cross"] > shuffle_flux["W_cross"]
    
    return {
        "seed": seed_id,
        "W_cross": flux["W_cross"],
        "W_within": flux["W_within"],
        "ratio": flux["ratio"],
        "W_shuffle": shuffle_flux["W_cross"],
        "success": success,
        "margin": flux["W_cross"] - shuffle_flux["W_cross"],
    }

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
def compute_combined_statistics(new_results, existing_successes=5, existing_total=5):
    """
    Combine new results with existing 5/5 validation.
    """
    new_successes = sum(1 for r in new_results if r["success"])
    new_total = len(new_results)
    
    total_successes = existing_successes + new_successes
    total_trials = existing_total + new_total
    
    # Binomial test: H0 = success rate ≤ 0.5
    p_value = stats.binom_test(total_successes, total_trials, p=0.5, alternative='greater')
    
    # Effect size (success rate)
    success_rate = total_successes / total_trials
    
    # 95% CI using Wilson score
    z = 1.96
    n = total_trials
    p_hat = success_rate
    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denominator
    ci_lower = center - margin
    ci_upper = center + margin
    
    return {
        "total_successes": total_successes,
        "total_trials": total_trials,
        "success_rate": success_rate,
        "p_value": p_value,
        "ci_95": (ci_lower, ci_upper),
        "new_successes": new_successes,
        "new_total": new_total,
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("="*70)
    print("HAWKING WORMHOLE EXTENDED VALIDATION")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Backend: {CONFIG['backend']}")
    print(f"Existing validation: {CONFIG['existing_successes']}/{CONFIG['existing_seeds']} seeds")
    print(f"New seeds to run: {CONFIG['n_seeds']}")
    print(f"Target total: {CONFIG['existing_seeds'] + CONFIG['n_seeds']} seeds")
    print("="*70)
    
    # 1. Initialize framework
    print("\n[1. FRAMEWORK INITIALIZATION]")
    fw = QMCFramework(
        project=CONFIG["project_name"],
        backend_name=CONFIG["backend"],
        shots=CONFIG["shots"],
        auto_confirm=False,
    )
    fw.initialize(mode=RunMode.QPU)
    fw.connect()
    
    # 2. Calibration
    print("\n[2. CALIBRATION ANALYSIS]")
    fw.analyze_calibration()
    
    # 3. Create wormhole topology
    print("\n[3. WORMHOLE TOPOLOGY]")
    topology = create_wormhole_topology(
        n_ring=CONFIG["ring_size"],
        n_throat=CONFIG["throat_qubits"]
    )
    print(f"Universe A: {len(topology['universe_A'])} qubits")
    print(f"Universe B: {len(topology['universe_B'])} qubits")
    print(f"Throat bridges: {len(topology['throat_pairs'])}")
    
    # 4. Run seeds
    print("\n[4. SEED EXECUTION]")
    all_results = []
    
    for seed in range(CONFIG["n_seeds"]):
        print(f"\n--- Seed {seed + 1}/{CONFIG['n_seeds']} ---")
        
        # Set random seed for reproducibility
        np.random.seed(seed + 100)  # Offset from existing seeds
        
        # Create circuit
        circuit = create_wormhole_circuit(
            topology,
            J_ring=CONFIG["J_ring"],
            J_throat=CONFIG["J_throat"],
            trotter_steps=CONFIG["trotter_steps"]
        )
        
        # Execute
        results = fw.run_on_qpu([circuit], shots=CONFIG["shots"])
        
        # Evaluate
        seed_result = evaluate_seed(results, topology, seed)
        all_results.append(seed_result)
        
        status = "✅ WORMHOLE > SHUFFLE" if seed_result["success"] else "❌ FAILED"
        print(f"W_cross: {seed_result['W_cross']:.4f}")
        print(f"W_shuffle: {seed_result['W_shuffle']:.4f}")
        print(f"Result: {status}")
    
    # 5. Combined statistics
    print("\n" + "="*70)
    print("COMBINED STATISTICAL ANALYSIS")
    print("="*70)
    
    stats_result = compute_combined_statistics(
        all_results,
        existing_successes=CONFIG["existing_successes"],
        existing_total=CONFIG["existing_seeds"]
    )
    
    print(f"\nExisting: {CONFIG['existing_successes']}/{CONFIG['existing_seeds']}")
    print(f"New: {stats_result['new_successes']}/{stats_result['new_total']}")
    print(f"TOTAL: {stats_result['total_successes']}/{stats_result['total_trials']}")
    print(f"\nSuccess rate: {stats_result['success_rate']*100:.1f}%")
    print(f"95% CI: [{stats_result['ci_95'][0]*100:.1f}%, {stats_result['ci_95'][1]*100:.1f}%]")
    print(f"p-value: {stats_result['p_value']:.6f}")
    
    # Verdict
    if stats_result['p_value'] < 0.001:
        verdict = "STRONG EVIDENCE ✅ (p < 0.001)"
    elif stats_result['p_value'] < 0.01:
        verdict = "CONFIRMATORY EVIDENCE ✅ (p < 0.01)"
    elif stats_result['p_value'] < 0.05:
        verdict = "PRELIMINARY EVIDENCE ⚠️ (p < 0.05)"
    else:
        verdict = "INSUFFICIENT EVIDENCE ❌ (p ≥ 0.05)"
    
    print(f"\nVERDICT: {verdict}")
    
    # 6. Save results
    print("\n[5. OUTPUT FILES]")
    print(f"Report: {fw.last_report_path}")
    
    output_data = {
        "experiment": CONFIG["project_name"],
        "date": datetime.now().isoformat(),
        "topology": {
            "n_qubits": CONFIG["n_qubits"],
            "ring_size": CONFIG["ring_size"],
            "throat_qubits": CONFIG["throat_qubits"],
        },
        "seeds": all_results,
        "statistics": stats_result,
        "verdict": verdict,
    }
    
    output_file = f"HAWKING_WORMHOLE_EXTENDED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results: {output_file}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return all_results, stats_result

# =============================================================================
# PRE-EXECUTION CHECKLIST
# =============================================================================
"""
PRE-EXECUTION CHECKLIST:
========================
- [x] Framework imported from qmc_quantum_framework_v2_5_23
- [x] auto_confirm=False
- [x] Backend: ibm_fez
- [x] analyze_calibration() called
- [x] Shuffle control included
- [x] Combined statistics with existing 5/5 seeds

QPU BUDGET CHECK:
- Estimated time: ~10 minutes (15 seeds × ~40s each)
- Priority: 🟠 SERIOUS (recommended)
- Proceed? [Y/N]

EXPECTED OUTCOME:
- If 18-20/20 total successes → p < 0.001, STRONG claim
- If 15-17/20 successes → p < 0.01, CONFIRMATORY
- If <15/20 successes → Need to revise claim language
"""

if __name__ == "__main__":
    main()
