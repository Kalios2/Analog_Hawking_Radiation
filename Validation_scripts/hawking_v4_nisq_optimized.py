#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    HAWKING CROSS-PLATFORM VALIDATION v4 - NISQ OPTIMIZED                     ║
║                    QMC Research Lab - January 2026                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  POST-MORTEM v3:                                                             ║
║  - Ratio F_h/F_far = 0.77× (INVERSÉ vs attendu >1.8×)                        ║
║  - Depth 137 + 348 2Q gates → fidélité ~17% → signal noyé                    ║
║  - Corrélations ≈ 0 (thermalisé, pas XX ≈ -YY)                               ║
║                                                                              ║
║  CORRECTIONS v4:                                                             ║
║  1. Chaînes plus courtes (N=16 au lieu de 30)                                ║
║  2. Moins de Trotter steps (3 au lieu de 6)                                  ║
║  3. σ plus petit (1.5 au lieu de 3.0) → horizon plus localisé                ║
║  4. dt plus petit (0.8 au lieu de 1.5) → évolution plus courte               ║
║  5. kick_strength augmenté (0.3 au lieu de 0.15)                             ║
║  6. Profondeur cible : <50 après transpilation                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    # Test shallow circuits first
    python hawking_v4_nisq_optimized.py --backend ibm_fez --shots 8192 --dry-run
    
    # QPU execution with optimized parameters
    python hawking_v4_nisq_optimized.py --backend ibm_fez --shots 16384

PARAMETER COMPARISON:
    Parameter     | v3 (failed)  | v4 (optimized) | Rationale
    --------------|--------------|----------------|---------------------------
    chain_length  | 30           | 16             | Reduce 2Q gate count
    trotter_steps | 6            | 3              | Reduce depth by 50%
    sigma         | 3.0          | 1.5            | Sharper horizon profile
    dt            | 1.5          | 0.8            | Shorter evolution time
    kick_strength | 0.15         | 0.30           | Stronger initial signal
    
    Expected depth reduction: 137 → ~45 (3x improvement)
    Expected fidelity: 17% → ~60% (preserves quantum signal)
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import argparse
import warnings

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

# ============================================================================
# BACKEND SPECS
# ============================================================================

BACKEND_SPECS = {
    "ibm_fez": {"qubits": 156, "generation": "Heron r2", "topology": "heavy-hex", "available": True},
    "ibm_torino": {"qubits": 133, "generation": "Heron r1", "topology": "heavy-hex", "available": True},
}

# ============================================================================
# OPTIMIZED PARAMETERS FOR NISQ
# ============================================================================

# v3 parameters (FAILED on QPU)
V3_PARAMS = {
    "chain_length": 30,
    "trotter_steps": 6,
    "j_max": 1.0,
    "j_min": 0.1,
    "sigma": 3.0,
    "dt": 1.5,
    "kick_strength": 0.15,
    "expected_depth": 137,
    "expected_2q_gates": 348,
}

# v4 parameters (OPTIMIZED for NISQ)
V4_PARAMS = {
    "chain_length": 16,      # Reduced from 30
    "trotter_steps": 3,      # Reduced from 6 (halved!)
    "j_max": 1.0,
    "j_min": 0.1,
    "sigma": 1.5,            # Reduced from 3.0 (sharper horizon)
    "dt": 0.8,               # Reduced from 1.5 (shorter evolution)
    "kick_strength": 0.30,   # Increased from 0.15 (stronger signal)
    "expected_depth": 45,    # Target: <50
    "expected_2q_gates": 90, # 3 steps × 15 bonds × 2 = 90
}

# Ultra-shallow variant for extreme NISQ constraints
V4_ULTRA_SHALLOW = {
    "chain_length": 10,
    "trotter_steps": 2,
    "j_max": 1.0,
    "j_min": 0.2,
    "sigma": 1.0,
    "dt": 0.6,
    "kick_strength": 0.4,
    "expected_depth": 25,
    "expected_2q_gates": 36,
}


# ============================================================================
# CHAIN FINDER (from v3)
# ============================================================================

class HeavyHexChainFinder:
    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map
        self.n_qubits = coupling_map.size()
        self.adjacency = {i: set() for i in range(self.n_qubits)}
        for edge in coupling_map.get_edges():
            self.adjacency[edge[0]].add(edge[1])
            self.adjacency[edge[1]].add(edge[0])
    
    def find_chain(self, length: int, excluded: Set[int] = None) -> Optional[List[int]]:
        excluded = excluded or set()
        available = set(range(self.n_qubits)) - excluded
        
        for start in available:
            chain = [start]
            used = {start}
            while len(chain) < length:
                current = chain[-1]
                neighbors = self.adjacency[current] & available - used
                if not neighbors:
                    if len(chain) > 1:
                        chain = chain[::-1]
                        neighbors = self.adjacency[chain[-1]] & available - used
                    if not neighbors:
                        break
                if neighbors:
                    next_q = min(neighbors, key=lambda q: len(self.adjacency[q] & available - used))
                    chain.append(next_q)
                    used.add(next_q)
            if len(chain) >= length:
                return chain[:length]
        return None
    
    def find_disjoint_chains(self, n_chains: int, length: int) -> List[List[int]]:
        chains, excluded = [], set()
        for _ in range(n_chains):
            chain = self.find_chain(length, excluded)
            if not chain:
                break
            chains.append(chain)
            excluded.update(chain)
            for q in chain:
                excluded.update(self.adjacency[q])
        return chains
    
    def verify_connectivity(self, chain: List[int]) -> bool:
        return all(chain[i+1] in self.adjacency[chain[i]] for i in range(len(chain)-1))


# ============================================================================
# SWAP VERIFIER (from v3)
# ============================================================================

class SwapVerifier:
    @staticmethod
    def verify(circuit: QuantumCircuit, initial_layout: List[int]) -> Dict:
        result = {"swap_count": 0, "verdict": "UNKNOWN"}
        for inst in circuit.data:
            if "swap" in inst.operation.name.lower():
                result["swap_count"] += 1
        
        if hasattr(circuit, 'layout') and circuit.layout:
            try:
                final = circuit.layout.final_index_layout()
                if list(final)[:len(initial_layout)] != initial_layout:
                    result["routing_detected"] = True
            except:
                pass
        
        result["verdict"] = "0-SWAP VERIFIED" if result["swap_count"] == 0 else "ROUTING DETECTED"
        return result


# ============================================================================
# OPTIMIZED HAWKING CIRCUIT (v4)
# ============================================================================

class OptimizedHawkingCircuit:
    """
    NISQ-optimized Hawking simulation circuit.
    
    Key differences from v3:
    - Configurable parameters for depth/fidelity tradeoff
    - Explicit depth tracking
    - Fidelity estimation
    """
    
    def __init__(self, physical_qubits: List[int], params: Dict):
        self.physical_qubits = physical_qubits
        self.n_qubits = len(physical_qubits)
        self.params = params
        
        # Extract parameters
        self.j_max = params.get("j_max", 1.0)
        self.j_min = params.get("j_min", 0.1)
        self.sigma = params.get("sigma", 1.5)
        self.dt = params.get("dt", 0.8)
        self.trotter_steps = params.get("trotter_steps", 3)
        self.kick_strength = params.get("kick_strength", 0.3)
        
        self.x_h = self.n_qubits // 2  # Horizon at center
        self.coupling = self._compute_coupling()
    
    def _compute_coupling(self) -> np.ndarray:
        """Gaussian coupling profile J(x) with sharper horizon."""
        x = np.arange(self.n_qubits - 1)
        r = self.j_min / self.j_max
        # Sharper Gaussian with smaller sigma
        gaussian = np.exp(-np.abs(x - self.x_h)**2 / (2 * self.sigma**2))
        return self.j_max * (1 - (1 - r) * gaussian)
    
    def build_circuit(self, basis: str = 'Z', seed: int = 42) -> QuantumCircuit:
        """Build circuit on logical qubits 0 to n-1."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.name = f"Hawking_v4_N{self.n_qubits}_T{self.trotter_steps}_{basis}"
        
        # Initial kicks (stronger than v3)
        np.random.seed(seed)
        for i in range(self.n_qubits):
            kick = self.kick_strength * np.random.uniform(-1, 1) * np.pi
            qc.ry(kick, i)
        qc.barrier()
        
        # Trotter evolution (fewer steps than v3)
        for _ in range(self.trotter_steps):
            # Even bonds
            for i in range(0, self.n_qubits - 1, 2):
                theta = self.coupling[i] * self.dt
                qc.rxx(theta, i, i+1)
                qc.ryy(theta, i, i+1)
            # Odd bonds
            for i in range(1, self.n_qubits - 1, 2):
                theta = self.coupling[i] * self.dt
                qc.rxx(theta, i, i+1)
                qc.ryy(theta, i, i+1)
            qc.barrier()
        
        # Basis rotation
        if basis == 'X':
            for i in range(self.n_qubits):
                qc.h(i)
        elif basis == 'Y':
            for i in range(self.n_qubits):
                qc.sdg(i)
                qc.h(i)
        
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc
    
    def get_horizon_qubits(self) -> List[int]:
        """Qubits near horizon (logical indices)."""
        c = self.x_h
        # With smaller sigma, use fewer horizon qubits
        return [i for i in [c-1, c, c+1] if 0 <= i < self.n_qubits]
    
    def get_far_field_qubits(self) -> List[int]:
        """Edge qubits (logical indices)."""
        return [0, 1, self.n_qubits-2, self.n_qubits-1]
    
    def estimate_fidelity(self, gate_fidelity: float = 0.995) -> float:
        """Estimate circuit fidelity based on 2Q gate count."""
        n_2q = 2 * (self.n_qubits - 1) * self.trotter_steps  # RXX + RYY per bond
        return gate_fidelity ** n_2q
    
    def get_metrics(self) -> Dict:
        """Return circuit metrics for comparison."""
        n_2q = 2 * (self.n_qubits - 1) * self.trotter_steps
        return {
            "n_qubits": self.n_qubits,
            "trotter_steps": self.trotter_steps,
            "expected_2q_gates": n_2q,
            "estimated_fidelity": self.estimate_fidelity(),
            "horizon_width": self.sigma,
            "evolution_time": self.dt * self.trotter_steps,
        }


# ============================================================================
# STATISTICS (from v3, corrected)
# ============================================================================

def compute_flux(counts: Dict, target_indices: List[int], n_qubits: int) -> Tuple[float, float]:
    total_shots = sum(counts.values())
    n_targets = len(target_indices)
    if not target_indices or total_shots == 0:
        return 0.0, 0.0
    
    excitations = 0
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        for idx in target_indices:
            if idx < len(bits) and bits[idx] == '1':
                excitations += count
    
    total_obs = total_shots * n_targets
    flux = excitations / total_obs
    std_err = np.sqrt(flux * (1 - flux) / total_obs)
    return flux, std_err


def compute_ratio_bayesian(F_h, F_h_err, F_far, F_far_err, n_h, n_far, shots):
    alpha, beta = 1, 1
    obs_h = shots * n_h
    obs_far = shots * n_far
    
    exc_h = F_h * obs_h
    exc_far = F_far * obs_far
    
    F_h_bayes = (exc_h + alpha) / (obs_h + alpha + beta)
    F_far_bayes = (exc_far + alpha) / (obs_far + alpha + beta)
    
    if F_far_bayes <= 0:
        return float('inf'), float('inf')
    
    ratio = F_h_bayes / F_far_bayes
    
    def beta_var(exc, n):
        a, b = exc + alpha, (n - exc) + beta
        return (a * b) / ((a + b)**2 * (a + b + 1))
    
    var_h = beta_var(exc_h, obs_h)
    var_far = beta_var(exc_far, obs_far)
    rel_var = var_h / (F_h_bayes**2) + var_far / (F_far_bayes**2)
    ratio_err = ratio * np.sqrt(rel_var)
    
    return ratio, ratio_err


def compute_correlator(counts: Dict, q1: int, q2: int, n_qubits: int) -> Tuple[float, float]:
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0, 0.0
    
    same, diff = 0, 0
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        if q1 < len(bits) and q2 < len(bits):
            if bits[q1] == bits[q2]:
                same += count
            else:
                diff += count
    
    corr = (same - diff) / total_shots
    err = np.sqrt((1 - corr**2) / total_shots)
    return corr, err


# ============================================================================
# MOCK FUNCTIONS
# ============================================================================

def create_mock_coupling_map(n_qubits: int) -> CouplingMap:
    edges = [(i, i+1) for i in range(n_qubits-1)]
    for i in range(0, n_qubits-12, 12):
        if i + 6 < n_qubits:
            edges.append((i, i+6))
    return CouplingMap(couplinglist=edges)


def generate_mock_results(meta_list: List[Dict], shots: int) -> List[Dict]:
    np.random.seed(42)
    results = []
    for meta in meta_list:
        n = meta["n_qubits"]
        horizon = meta["horizon_qubits"]
        counts = {}
        for _ in range(shots):
            bits = ['0'] * n
            for i in range(n):
                # Higher probability at horizon (simulate expected signal)
                prob = 0.35 if i in horizon else 0.05
                if np.random.random() < prob:
                    bits[i] = '1'
            bs = ''.join(bits[::-1])
            counts[bs] = counts.get(bs, 0) + 1
        results.append(counts)
    return results


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def run_validation(backend_name: str = "ibm_fez",
                   shots: int = 8192,
                   n_chains: int = 4,
                   variant: str = "optimized",
                   dry_run: bool = False):
    """
    Run validation with NISQ-optimized parameters.
    
    Variants:
    - "optimized": V4_PARAMS (default, balanced)
    - "ultra_shallow": V4_ULTRA_SHALLOW (extreme NISQ)
    - "v3": V3_PARAMS (original, for comparison)
    """
    
    print("=" * 80)
    print("HAWKING v4 - NISQ OPTIMIZED VALIDATION")
    print("=" * 80)
    
    # Select parameters
    if variant == "ultra_shallow":
        params = V4_ULTRA_SHALLOW
        print(f"\n📉 Using ULTRA-SHALLOW parameters (extreme NISQ)")
    elif variant == "v3":
        params = V3_PARAMS
        print(f"\n⚠️  Using V3 parameters (for comparison - may fail)")
    else:
        params = V4_PARAMS
        print(f"\n✓ Using OPTIMIZED v4 parameters")
    
    chain_length = params["chain_length"]
    
    print(f"\n📊 Parameter comparison vs v3:")
    print(f"   {'Parameter':<15} {'v3':<10} {'v4':<10} {'Change':<15}")
    print(f"   {'-'*50}")
    print(f"   {'chain_length':<15} {V3_PARAMS['chain_length']:<10} {params['chain_length']:<10} {params['chain_length']/V3_PARAMS['chain_length']*100-100:+.0f}%")
    print(f"   {'trotter_steps':<15} {V3_PARAMS['trotter_steps']:<10} {params['trotter_steps']:<10} {params['trotter_steps']/V3_PARAMS['trotter_steps']*100-100:+.0f}%")
    print(f"   {'sigma':<15} {V3_PARAMS['sigma']:<10} {params['sigma']:<10} {params['sigma']/V3_PARAMS['sigma']*100-100:+.0f}%")
    print(f"   {'dt':<15} {V3_PARAMS['dt']:<10} {params['dt']:<10} {params['dt']/V3_PARAMS['dt']*100-100:+.0f}%")
    print(f"   {'kick_strength':<15} {V3_PARAMS['kick_strength']:<10} {params['kick_strength']:<10} {params['kick_strength']/V3_PARAMS['kick_strength']*100-100:+.0f}%")
    print(f"   {'expected_2q':<15} {V3_PARAMS['expected_2q_gates']:<10} {params['expected_2q_gates']:<10} {params['expected_2q_gates']/V3_PARAMS['expected_2q_gates']*100-100:+.0f}%")
    
    # Estimate fidelity improvement
    fid_v3 = 0.995 ** V3_PARAMS["expected_2q_gates"]
    fid_v4 = 0.995 ** params["expected_2q_gates"]
    print(f"\n   📈 Estimated fidelity: {fid_v3:.1%} (v3) → {fid_v4:.1%} (v4) = {fid_v4/fid_v3:.1f}x improvement")
    
    # Backend
    specs = BACKEND_SPECS.get(backend_name, {})
    print(f"\n🖥️  Backend: {backend_name} ({specs.get('generation', '?')})")
    print(f"   Mode: {'DRY-RUN' if dry_run else 'LIVE QPU'}")
    
    # =========================================================================
    # Step 1: Chain discovery
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 1: Chain Discovery")
    print("-" * 40)
    
    if dry_run:
        coupling_map = create_mock_coupling_map(specs.get("qubits", 156))
        print("  [DRY-RUN] Mock coupling map")
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        coupling_map = backend.coupling_map
        print(f"  [LIVE] Got coupling map ({coupling_map.size()} qubits)")
    
    finder = HeavyHexChainFinder(coupling_map)
    chains = finder.find_disjoint_chains(n_chains, chain_length)
    
    print(f"\n  Found {len(chains)} chains of length {chain_length}:")
    for i, ch in enumerate(chains):
        conn = "✓" if finder.verify_connectivity(ch) else "✗"
        print(f"    Chain {i}: {ch[0]}→{ch[-1]} [{conn}]")
    
    if not chains:
        print("  ERROR: No chains found")
        return None
    
    # =========================================================================
    # Step 2: Build circuits
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 2: Circuit Construction")
    print("-" * 40)
    
    all_circuits = []
    all_layouts = []
    circuit_meta = []
    
    for chain_id, chain in enumerate(chains):
        builder = OptimizedHawkingCircuit(chain, params)
        metrics = builder.get_metrics()
        
        for basis in ['Z', 'X', 'Y']:
            qc = builder.build_circuit(basis=basis)
            all_circuits.append(qc)
            all_layouts.append(chain)
            circuit_meta.append({
                "chain_id": chain_id,
                "basis": basis,
                "n_qubits": len(chain),
                "horizon_qubits": builder.get_horizon_qubits(),
                "far_field_qubits": builder.get_far_field_qubits(),
            })
        
        print(f"  Chain {chain_id}: depth={all_circuits[-1].depth()}, est_fidelity={metrics['estimated_fidelity']:.1%}")
    
    # =========================================================================
    # Step 3: Transpile and verify
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 3: Transpilation & SWAP Check")
    print("-" * 40)
    
    transpiled = []
    swap_report = []
    
    for i, (qc, layout) in enumerate(zip(all_circuits, all_layouts)):
        tc = transpile(qc, coupling_map=coupling_map, initial_layout=layout, 
                       optimization_level=3, seed_transpiler=42)
        transpiled.append(tc)
        
        swap_info = SwapVerifier.verify(tc, layout)
        swap_report.append(swap_info)
        
        status = "✓" if "VERIFIED" in swap_info["verdict"] else "✗"
        print(f"  {qc.name}: depth={tc.depth()}, {status}")
    
    total_swaps = sum(r["swap_count"] for r in swap_report)
    print(f"\n  Total SWAPs: {total_swaps}")
    print(f"  {'✓ 0-SWAP VERIFIED' if total_swaps == 0 else '✗ ROUTING DETECTED'}")
    
    # =========================================================================
    # Step 4: Execute
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 4: Execution")
    print("-" * 40)
    
    if dry_run:
        print("  [DRY-RUN] Mock results")
        all_counts = generate_mock_results(circuit_meta, shots)
        exec_meta = {"real_qpu": False}
    else:
        print("  [LIVE] Submitting to QPU...")
        # Use QMC Framework or direct runtime
        try:
            sys.path.insert(0, '/mnt/project')
            from qmc_quantum_framework_v2_5_23 import QMCFrameworkV2_4 as QMCFramework, RunMode
            
            fw = QMCFramework(project="HAWKING_V4", backend_name=backend_name, 
                            shots=shots, auto_confirm=False)
            fw.initialize(mode=RunMode.QPU)
            fw.connect()
            
            results = fw.run_on_qpu(all_circuits, shots=shots)
            all_counts = [r.get('counts', r) if isinstance(r, dict) else {} for r in results]
            exec_meta = {"real_qpu": True, "framework": "2.5.23"}
            
        except Exception as e:
            print(f"  Framework error: {e}")
            print("  Using direct Runtime...")
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            
            service = QiskitRuntimeService()
            backend = service.backend(backend_name)
            sampler = SamplerV2(backend)
            job = sampler.run(transpiled, shots=shots)
            result = job.result()
            
            all_counts = [pub.data.c.get_counts() for pub in result]
            exec_meta = {"real_qpu": True, "job_id": job.job_id()}
    
    # =========================================================================
    # Step 5: Analysis
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 5: Analysis")
    print("-" * 40)
    
    analysis = []
    
    for chain_id in range(len(chains)):
        meta = [m for m in circuit_meta if m["chain_id"] == chain_id]
        counts = all_counts[chain_id * 3 : chain_id * 3 + 3]  # Z, X, Y
        
        z_counts = counts[0]
        n_q = meta[0]["n_qubits"]
        h_qubits = meta[0]["horizon_qubits"]
        f_qubits = meta[0]["far_field_qubits"]
        
        F_h, F_h_err = compute_flux(z_counts, h_qubits, n_q)
        F_far, F_far_err = compute_flux(z_counts, f_qubits, n_q)
        
        ratio, ratio_err = compute_ratio_bayesian(
            F_h, F_h_err, F_far, F_far_err,
            len(h_qubits), len(f_qubits), shots
        )
        
        # Correlators
        x_counts, y_counts = counts[1], counts[2]
        c = n_q // 2
        q1, q2 = max(0, c-2), min(n_q-1, c+2)
        
        xx, xx_err = compute_correlator(x_counts, q1, q2, n_q)
        yy, yy_err = compute_correlator(y_counts, q1, q2, n_q)
        
        partner_sum = xx + yy
        partner_check = abs(partner_sum) < 0.2
        
        analysis.append({
            "chain_id": chain_id,
            "F_h": F_h, "F_h_err": F_h_err,
            "F_far": F_far, "F_far_err": F_far_err,
            "ratio": ratio, "ratio_err": ratio_err,
            "XX": xx, "YY": yy,
            "partner_sum": partner_sum,
            "partner_check": partner_check,
        })
        
        print(f"\n  Chain {chain_id}:")
        print(f"    F_h = {F_h:.4f} ± {F_h_err:.4f}")
        print(f"    F_far = {F_far:.4f} ± {F_far_err:.4f}")
        print(f"    Ratio = {ratio:.2f} ± {ratio_err:.2f}× {'✓' if ratio > 1.8 else '✗'}")
        print(f"    XX+YY = {partner_sum:.3f} {'✓' if partner_check else '✗'}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    mean_ratio = np.mean([a["ratio"] for a in analysis])
    all_pass = all(a["ratio"] > 1.8 for a in analysis)
    
    print(f"  Variant: {variant}")
    print(f"  Backend: {backend_name}")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'REAL QPU'}")
    print(f"  Chains: {len(chains)} × {chain_length} qubits")
    print(f"  Mean ratio: {mean_ratio:.2f}×")
    print(f"  Above 1.8×: {'✓ ALL' if all_pass else '✗ SOME FAILED'}")
    print(f"  0-SWAP: {'✓' if total_swaps == 0 else '✗'}")
    
    pub_ready = not dry_run and all_pass and total_swaps == 0
    print(f"\n  {'✅ PUBLICATION READY' if pub_ready else '❌ NOT PUBLICATION READY'}")
    print("=" * 80)
    
    # Save report
    report = {
        "version": "4.0 (NISQ-optimized)",
        "variant": variant,
        "params": params,
        "backend": backend_name,
        "dry_run": dry_run,
        "chains": [{"id": i, "qubits": chains[i]} for i in range(len(chains))],
        "analysis": analysis,
        "summary": {
            "mean_ratio": mean_ratio,
            "all_above_threshold": all_pass,
            "swap_count": total_swaps,
            "publication_ready": pub_ready,
        }
    }
    
    os.makedirs("../analyses", exist_ok=True)
    mode_tag = "DRYRUN" if dry_run else "QPU"
    out_path = f"../analyses/hawking_v4_{variant}_{backend_name}_{mode_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n  Report: {out_path}")
    
    return report


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAWKING v4 - NISQ Optimized")
    parser.add_argument("--backend", default="ibm_fez", choices=["ibm_fez", "ibm_torino"])
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--variant", default="optimized", 
                        choices=["optimized", "ultra_shallow", "v3"])
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    run_validation(
        backend_name=args.backend,
        shots=args.shots,
        n_chains=args.n_chains,
        variant=args.variant,
        dry_run=args.dry_run
    )
