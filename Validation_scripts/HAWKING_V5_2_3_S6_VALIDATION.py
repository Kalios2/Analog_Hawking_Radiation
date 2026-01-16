#!/usr/bin/env python3
"""
HAWKING V5.2.3 - S=6 MULTI-SCALE VALIDATION
============================================
Objective: Resolve ratio discrepancy between Palier 9 (83.2×) and V5.2.2 (1.14-1.61×)
           by running S=6 Trotter steps on V5.2.2 configurations

Author: QMC Research Lab
Date: January 2026
Framework: qmc_quantum_framework v2.5.23

RATIONALE:
- V5.2.2 used S=1 Trotter step → weak signal (1.14-1.61× ratio)
- Palier 9 used S=6 Trotter steps → strong signal (83.2× ratio)
- This run tests S=6 on V5.2.2 scales to isolate Trotter depth effect

EXPECTED OUTCOME:
- If S=6 shows ~50-80× in V5.2.2 zones → Depth is the driver
- If S=6 shows ~5-10× → Qubit selection/calibration is the driver
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
    "project_name": "HAWKING_V5_2_3_S6_VALIDATION",
    "backend": "ibm_fez",
    "shots": 16384,
    "auto_confirm": False,  # ⚠️ NEVER True in production
    
    # Multi-scale configurations (same as V5.2.2)
    "configurations": [
        {"name": "Mini_S6", "N": 20, "x_horizon": 10, "trotter_steps": 6},
        {"name": "Medium_S6", "N": 40, "x_horizon": 20, "trotter_steps": 6},
        {"name": "Large_S6", "N": 80, "x_horizon": 40, "trotter_steps": 6},
    ],
    
    # Clean zones (validated in V5.2.2)
    "clean_zones": {
        "zone_1": list(range(0, 31)),      # q0-q30
        "zone_2": list(range(120, 156)),   # q120-q155
    },
    
    # XY Hamiltonian parameters
    "J_max": 1.0,       # Strong coupling
    "J_min": 0.1,       # Weak coupling at horizon (10% of max)
    "sigma": 2.0,       # Gaussian width
    
    # Validation settings
    "include_shuffle": True,
    "seeds": 3,         # 3 seeds per configuration
}

# Success criteria
SUCCESS_CRITERIA = {
    "min_ratio": 1.8,           # GO threshold
    "min_peak_accuracy": 1.0,   # 100% peak at horizon
    "max_shuffle_retention": 0.2,  # <20% signal after shuffle
}

# =============================================================================
# CIRCUIT GENERATION
# =============================================================================
def gaussian_coupling(x, x_h, J_max, J_min, sigma):
    """Gaussian coupling profile with dip at horizon."""
    return J_max * (1 - (1 - J_min/J_max) * np.exp(-(x - x_h)**2 / (2 * sigma**2)))

def create_hawking_circuit(N, x_horizon, trotter_steps, J_max=1.0, J_min=0.1, sigma=2.0, dt=0.1):
    """
    Create XY spin chain circuit with Trotter decomposition.
    
    Parameters:
    -----------
    N : int
        Number of qubits (chain length)
    x_horizon : int
        Position of the analog horizon
    trotter_steps : int
        Number of Trotter steps S
    J_max : float
        Maximum coupling strength
    J_min : float
        Minimum coupling at horizon
    sigma : float
        Gaussian width parameter
    dt : float
        Trotter time step
    """
    qc = QuantumCircuit(N, N)
    
    # Calculate coupling profile
    J = np.array([gaussian_coupling(i, x_horizon, J_max, J_min, sigma) for i in range(N-1)])
    
    # Initialization: Apply "kick" at horizon (H gates)
    # This creates the initial excitation
    kick_range = range(max(0, x_horizon-2), min(N, x_horizon+3))
    for i in kick_range:
        qc.h(i)
    
    # Trotter evolution: XY Hamiltonian
    for step in range(trotter_steps):
        # Odd links (0-1, 2-3, 4-5, ...)
        for i in range(0, N-1, 2):
            theta = 2 * J[i] * dt
            qc.rxx(theta, i, i+1)
            qc.ryy(theta, i, i+1)
        
        # Even links (1-2, 3-4, 5-6, ...)
        for i in range(1, N-1, 2):
            theta = 2 * J[i] * dt
            qc.rxx(theta, i, i+1)
            qc.ryy(theta, i, i+1)
    
    # Basis rotation for measurement
    for i in range(N):
        qc.h(i)
    
    # Measurement
    qc.measure(range(N), range(N))
    
    return qc

def create_shuffle_mapping(N, seed=None):
    """Create random permutation for shuffle control."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(N)

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def compute_excitation_density(counts, N):
    """Compute excitation density n(x) at each site."""
    n = np.zeros(N)
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Reverse bitstring (qiskit convention)
        bits = [int(b) for b in bitstring[::-1]]
        for i, b in enumerate(bits[:N]):
            n[i] += b * count / total
    
    return n

def compute_localization_ratio(n, x_horizon, near_range=3):
    """
    Compute localization ratio F_h / F_far.
    
    near_range: sites around horizon considered "near"
    """
    N = len(n)
    
    # Near-horizon: within near_range of horizon
    near_indices = range(max(0, x_horizon - near_range), min(N, x_horizon + near_range + 1))
    n_near = np.mean([n[i] for i in near_indices])
    
    # Far-field: outside near_range from horizon
    far_indices = [i for i in range(N) if abs(i - x_horizon) > near_range]
    n_far = np.mean([n[i] for i in far_indices]) if far_indices else 1e-10
    
    ratio = n_near / max(n_far, 1e-10)
    
    return {
        "n_near": n_near,
        "n_far": n_far,
        "ratio": ratio,
        "max_site": int(np.argmax(n)),
    }

def apply_shuffle(counts, mapping, N):
    """Apply shuffle permutation to measurement results."""
    shuffled_counts = {}
    
    for bitstring, count in counts.items():
        bits = list(bitstring[::-1])[:N]  # Reverse and take N bits
        shuffled_bits = ['0'] * N
        for i, m in enumerate(mapping):
            if i < len(bits):
                shuffled_bits[m] = bits[i]
        shuffled_string = ''.join(shuffled_bits[::-1])
        shuffled_counts[shuffled_string] = shuffled_counts.get(shuffled_string, 0) + count
    
    return shuffled_counts

def analyze_configuration(results, config, include_shuffle=True):
    """Full analysis for one configuration."""
    N = config["N"]
    x_horizon = config["x_horizon"]
    
    counts = results.get_counts() if hasattr(results, 'get_counts') else results
    
    # Standard analysis
    n = compute_excitation_density(counts, N)
    metrics = compute_localization_ratio(n, x_horizon)
    metrics["config"] = config["name"]
    metrics["N"] = N
    metrics["x_horizon"] = x_horizon
    metrics["peak_at_horizon"] = (metrics["max_site"] == x_horizon)
    
    # Shuffle control
    if include_shuffle:
        shuffle_mapping = create_shuffle_mapping(N, seed=42)
        shuffled_counts = apply_shuffle(counts, shuffle_mapping, N)
        n_shuffled = compute_excitation_density(shuffled_counts, N)
        shuffle_metrics = compute_localization_ratio(n_shuffled, x_horizon)
        
        metrics["shuffle_ratio"] = shuffle_metrics["ratio"]
        metrics["shuffle_max_site"] = shuffle_metrics["max_site"]
        metrics["shuffle_degradation"] = 1 - (shuffle_metrics["ratio"] / max(metrics["ratio"], 1e-10))
        metrics["peak_moved"] = (shuffle_metrics["max_site"] != metrics["max_site"])
    
    return metrics

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def evaluate_success(all_metrics):
    """Evaluate if experiment meets success criteria."""
    print("\n" + "="*70)
    print("VALIDATION RESULTS - S=6 MULTI-SCALE")
    print("="*70)
    
    checks_passed = 0
    total_checks = 0
    
    for m in all_metrics:
        print(f"\n{m['config']} (N={m['N']}, x_h={m['x_horizon']}):")
        print(f"  Ratio: {m['ratio']:.2f}× (threshold: {SUCCESS_CRITERIA['min_ratio']}×)")
        print(f"  Peak at horizon: {m['max_site']} == {m['x_horizon']} → {'✅' if m['peak_at_horizon'] else '❌'}")
        
        if m.get('shuffle_degradation') is not None:
            print(f"  Shuffle degradation: {m['shuffle_degradation']*100:.1f}%")
            print(f"  Peak moved: {m['shuffle_max_site']} → {'✅ YES' if m['peak_moved'] else '❌ NO'}")
        
        # Count checks
        total_checks += 3
        if m['ratio'] >= SUCCESS_CRITERIA['min_ratio']:
            checks_passed += 1
        if m['peak_at_horizon']:
            checks_passed += 1
        if m.get('peak_moved', False):
            checks_passed += 1
    
    print("\n" + "-"*70)
    score = checks_passed / total_checks * 100
    print(f"GLOBAL SCORE: {checks_passed}/{total_checks} ({score:.1f}%)")
    
    if score >= 80:
        verdict = "GO ✅ - S=6 validation successful"
    elif score >= 60:
        verdict = "GO_MARGINAL ⚠️ - Partial validation"
    else:
        verdict = "NO-GO ❌ - Validation failed"
    
    print(f"VERDICT: {verdict}")
    
    return {
        "checks_passed": checks_passed,
        "total_checks": total_checks,
        "score": score,
        "verdict": verdict,
    }

def compare_with_v522(all_metrics):
    """Compare S=6 results with V5.2.2 S=1 results."""
    print("\n" + "="*70)
    print("COMPARISON: S=6 (this run) vs S=1 (V5.2.2)")
    print("="*70)
    
    # V5.2.2 reference values (S=1)
    v522_ref = {
        "Mini": {"ratio": 1.61, "peak_at_horizon": True},
        "Medium": {"ratio": 1.37, "peak_at_horizon": True},
        "Large": {"ratio": 1.14, "peak_at_horizon": True},
    }
    
    print(f"{'Config':<15} {'S=1 Ratio':<12} {'S=6 Ratio':<12} {'Amplification':<15}")
    print("-"*55)
    
    for m in all_metrics:
        base_name = m['config'].replace("_S6", "")
        if base_name in v522_ref:
            s1_ratio = v522_ref[base_name]["ratio"]
            s6_ratio = m["ratio"]
            amplification = s6_ratio / s1_ratio
            print(f"{m['config']:<15} {s1_ratio:<12.2f} {s6_ratio:<12.2f} {amplification:<15.1f}×")
    
    print("\nINTERPRETATION:")
    avg_amp = np.mean([m["ratio"] / v522_ref[m['config'].replace("_S6", "")]["ratio"] 
                       for m in all_metrics if m['config'].replace("_S6", "") in v522_ref])
    
    if avg_amp > 30:
        print(f"  Average amplification: {avg_amp:.1f}× → CONFIRMS Trotter depth drives ratio")
        print("  → The ratio discrepancy is explained by S=6 vs S=1")
    elif avg_amp > 5:
        print(f"  Average amplification: {avg_amp:.1f}× → PARTIAL confirmation")
        print("  → Both depth AND calibration contribute")
    else:
        print(f"  Average amplification: {avg_amp:.1f}× → Calibration is the main driver")
        print("  → Need to investigate qubit selection criteria")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("="*70)
    print("HAWKING V5.2.3 - S=6 MULTI-SCALE VALIDATION")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Backend: {CONFIG['backend']}")
    print(f"Shots: {CONFIG['shots']}")
    print(f"Trotter steps: 6 (same as Palier 9)")
    print("="*70)
    
    # 1. Initialize framework
    print("\n[1. FRAMEWORK INITIALIZATION]")
    fw = QMCFramework(
        project=CONFIG["project_name"],
        backend_name=CONFIG["backend"],
        shots=CONFIG["shots"],
        auto_confirm=False,
    )
    
    # 2. Initialize mode
    fw.initialize(mode=RunMode.QPU)
    
    # 3. Connect to backend
    print("\n[2. BACKEND CONNECTION]")
    fw.connect()
    
    # 4. Analyze calibration
    print("\n[3. CALIBRATION ANALYSIS]")
    topology = fw.analyze_calibration()
    
    # Select qubits from clean zones
    available_qubits = CONFIG["clean_zones"]["zone_1"] + CONFIG["clean_zones"]["zone_2"]
    print(f"Clean zones: q0-30, q120-155")
    print(f"Total available qubits: {len(available_qubits)}")
    
    # 5. Generate and run circuits
    print("\n[4. CIRCUIT GENERATION & EXECUTION]")
    all_metrics = []
    
    for config in CONFIG["configurations"]:
        print(f"\n--- {config['name']} ---")
        print(f"N={config['N']}, x_horizon={config['x_horizon']}, S={config['trotter_steps']}")
        
        # Create circuit
        circuit = create_hawking_circuit(
            N=config["N"],
            x_horizon=config["x_horizon"],
            trotter_steps=config["trotter_steps"],
            J_max=CONFIG["J_max"],
            J_min=CONFIG["J_min"],
            sigma=CONFIG["sigma"],
        )
        
        print(f"Circuit depth: {circuit.depth()} layers")
        
        # Estimate cost
        estimate = fw.estimate_cost([circuit], shots=CONFIG["shots"])
        print(f"Estimated time: {estimate}")
        
        # Execute
        print("Executing on QPU...")
        results = fw.run_on_qpu([circuit], shots=CONFIG["shots"])
        
        # Analyze
        metrics = analyze_configuration(
            results, 
            config, 
            include_shuffle=CONFIG["include_shuffle"]
        )
        all_metrics.append(metrics)
        
        print(f"Ratio: {metrics['ratio']:.2f}×")
        print(f"Peak at site: {metrics['max_site']} (expected: {config['x_horizon']})")
    
    # 6. Validation
    validation_result = evaluate_success(all_metrics)
    
    # 7. Comparison with V5.2.2
    compare_with_v522(all_metrics)
    
    # 8. Save results
    print("\n[5. OUTPUT FILES]")
    print(f"Report: {fw.last_report_path}")
    print(f"Archive: {fw.last_archive_path}")
    
    # Save detailed JSON
    output_data = {
        "experiment": CONFIG["project_name"],
        "date": datetime.now().isoformat(),
        "backend": CONFIG["backend"],
        "configurations": CONFIG["configurations"],
        "metrics": all_metrics,
        "validation": validation_result,
    }
    
    output_file = f"HAWKING_V5_2_3_S6_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Detailed results: {output_file}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return all_metrics, validation_result

# =============================================================================
# PRE-EXECUTION CHECKLIST
# =============================================================================
"""
PRE-EXECUTION CHECKLIST:
========================
- [x] Framework imported from qmc_quantum_framework_v2_5_23
- [x] auto_confirm=False (NEVER True)
- [x] Backend specified: ibm_fez
- [x] RunMode.QPU set for real execution
- [x] analyze_calibration() called
- [x] estimate_cost() reviewed
- [x] Success criteria defined (min_ratio=1.8)
- [x] Output paths documented

QPU BUDGET CHECK:
- Estimated time: ~15-20 minutes total (3 configs × ~5 min each)
- Available budget: ~50 min/month
- Proceed? [Y/N]

EXPECTED OUTCOME:
- Resolves ratio discrepancy question (83.2× vs 1.14×)
- Provides data for Figure A (Ratio vs Trotter depth)
- Validates S=6 on V5.2.2 qubit selections
"""

if __name__ == "__main__":
    main()
