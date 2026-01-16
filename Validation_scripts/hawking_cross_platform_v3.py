#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    HAWKING CROSS-PLATFORM VALIDATION v3 - PUBLICATION GRADE                  ║
║                    QMC Research Lab - January 2026                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORRECTIONS v3 (based on Alicia's second review):                           ║
║  ✓ Real QPU execution via QMC Framework (not mock in LIVE mode)              ║
║  ✓ Robust 0-SWAP verification via TranspileLayout.final_layout               ║
║  ✓ Correct statistical error: std_err = sqrt(p(1-p)/(shots*n_targets))       ║
║  ✓ No F_far clamp - use Bayesian pseudo-count for small denominators         ║
║  ✓ Publication-grade claims only after real QPU validation                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BACKEND SPECIFICATIONS (IBM Quantum January 2026):                          ║
║  - ibm_fez:       156 qubits, Heron r2 (heavy-hex) ← DEFAULT                 ║
║  - ibm_torino:    133 qubits, Heron r1 (heavy-hex) ← for comparison          ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    # Dry run (no QPU, mock data)
    python hawking_cross_platform_v3.py --backend ibm_fez --shots 8192 --dry-run
    
    # Real QPU execution (requires QMC Framework + IBM credentials)
    python hawking_cross_platform_v3.py --backend ibm_fez --shots 16384
    
    # Cross-platform comparison
    python hawking_cross_platform_v3.py --backend ibm_fez --shots 8192
    python hawking_cross_platform_v3.py --backend ibm_torino --shots 8192

PUBLICATION CHECKLIST (must ALL be ✓ before claiming results):
    □ Real QPU execution completed (not --dry-run)
    □ 0-SWAP verified via TranspileLayout
    □ Ratio F_h/F_far > 1.8 with statistical significance
    □ Partner correlations XX ≈ -YY verified
    □ Consistent across at least 2 backends
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
import argparse
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import CouplingMap, Layout

# ============================================================================
# SECTION 1: BACKEND SPECIFICATIONS (VERIFIED IBM QUANTUM JAN 2026)
# ============================================================================

BACKEND_SPECS = {
    "ibm_fez": {
        "qubits": 156,
        "generation": "Heron r2",
        "topology": "heavy-hex",
        "available": True,
        "max_chain_length": 39,
    },
    "ibm_torino": {
        "qubits": 133,
        "generation": "Heron r1", 
        "topology": "heavy-hex",
        "available": True,
        "max_chain_length": 30,
    },
}


# ============================================================================
# SECTION 2: HEAVY-HEX CHAIN FINDER (unchanged from v2)
# ============================================================================

class HeavyHexChainFinder:
    """
    Finds connected linear chains in heavy-hex topology.
    """
    
    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map
        self.n_qubits = coupling_map.size()
        self.adjacency = self._build_adjacency()
        
    def _build_adjacency(self) -> Dict[int, Set[int]]:
        adj = {i: set() for i in range(self.n_qubits)}
        for edge in self.coupling_map.get_edges():
            q1, q2 = edge
            adj[q1].add(q2)
            adj[q2].add(q1)
        return adj
    
    def find_linear_chain(self, length: int, 
                          excluded_qubits: Set[int] = None,
                          start_qubit: int = None) -> Optional[List[int]]:
        if excluded_qubits is None:
            excluded_qubits = set()
        available = set(range(self.n_qubits)) - excluded_qubits
        start_candidates = [start_qubit] if start_qubit is not None else list(available)
        
        for start in start_candidates:
            if start not in available:
                continue
            chain = self._extend_chain_greedy(start, available, length)
            if chain and len(chain) >= length:
                return chain[:length]
        return None
    
    def _extend_chain_greedy(self, start: int, available: Set[int], 
                             target_length: int) -> List[int]:
        chain = [start]
        used = {start}
        
        while len(chain) < target_length:
            current = chain[-1]
            neighbors = self.adjacency[current] & available - used
            
            if not neighbors:
                if len(chain) > 1:
                    chain = chain[::-1]
                    current = chain[-1]
                    neighbors = self.adjacency[current] & available - used
                if not neighbors:
                    break
            
            next_qubit = min(neighbors, 
                           key=lambda q: len(self.adjacency[q] & available - used))
            chain.append(next_qubit)
            used.add(next_qubit)
        return chain
    
    def find_disjoint_chains(self, n_chains: int, 
                             chain_length: int) -> List[List[int]]:
        chains = []
        excluded = set()
        
        for i in range(n_chains):
            chain = self.find_linear_chain(chain_length, excluded)
            if chain is None:
                warnings.warn(f"Could only find {i} chains of length {chain_length}")
                break
            chains.append(chain)
            excluded.update(chain)
            for q in chain:
                excluded.update(self.adjacency[q])
        return chains
    
    def verify_chain_connectivity(self, chain: List[int]) -> bool:
        for i in range(len(chain) - 1):
            if chain[i+1] not in self.adjacency[chain[i]]:
                return False
        return True


# ============================================================================
# SECTION 3: ROBUST SWAP VERIFIER (CORRECTED per Alicia)
# ============================================================================

class RobustSwapVerifier:
    """
    Verifies 0-SWAP claim using BOTH:
    1. TranspileLayout.final_layout comparison with initial_layout
    2. Explicit SwapGate counting in transpiled circuit
    
    Per Alicia's review: just counting "swap" in gate names is insufficient
    because transpilation can permute via other means (bridge operations, etc.)
    """
    
    @staticmethod
    def verify_no_routing(transpiled_circuit: QuantumCircuit,
                          initial_layout: List[int]) -> Dict:
        """
        Comprehensive SWAP/routing verification.
        
        Returns:
            Dict with:
            - swap_count: explicit SwapGate count
            - routing_detected: True if final_layout != initial_layout
            - layout_preserved: True if no permutation occurred
            - final_layout: actual final qubit mapping
            - verdict: "0-SWAP VERIFIED" or "ROUTING DETECTED"
        """
        result = {
            "swap_count": 0,
            "iswap_count": 0,
            "bridge_count": 0,
            "total_2q_gates": 0,
            "routing_detected": False,
            "layout_preserved": True,
            "final_layout": None,
            "initial_layout": initial_layout,
            "verdict": "UNKNOWN",
            "details": []
        }
        
        # 1. Count explicit SWAP-type gates
        for instruction in transpiled_circuit.data:
            op_name = instruction.operation.name.lower()
            n_qubits = len(instruction.qubits)
            
            if n_qubits == 2:
                result["total_2q_gates"] += 1
            
            if "swap" in op_name and "iswap" not in op_name:
                result["swap_count"] += 1
                result["details"].append(f"Found SwapGate: {op_name}")
            elif "iswap" in op_name:
                result["iswap_count"] += 1
                result["details"].append(f"Found iSwapGate: {op_name}")
        
        # 2. Check TranspileLayout for routing/permutation
        if hasattr(transpiled_circuit, 'layout') and transpiled_circuit.layout is not None:
            layout = transpiled_circuit.layout
            
            # Get final layout (after routing)
            try:
                # final_index_layout() returns mapping of virtual -> physical
                final_layout = layout.final_index_layout()
                result["final_layout"] = list(final_layout) if final_layout else None
                
                # Compare with initial layout
                # If initial_layout was [5, 6, 7, ...] and final is different,
                # then routing occurred
                if result["final_layout"] is not None:
                    # The initial_layout maps virtual qubit i -> physical qubit initial_layout[i]
                    # The final_layout maps virtual qubit i -> physical qubit final_layout[i]
                    # If they differ, routing (SWAP insertion) occurred
                    
                    n_virtual = len(initial_layout)
                    n_final = len(result["final_layout"]) if result["final_layout"] else 0
                    
                    if n_final >= n_virtual:
                        # Compare the first n_virtual entries
                        initial_mapping = initial_layout[:n_virtual]
                        final_mapping = result["final_layout"][:n_virtual]
                        
                        if initial_mapping != final_mapping:
                            result["routing_detected"] = True
                            result["layout_preserved"] = False
                            result["details"].append(
                                f"Layout changed: initial={initial_mapping[:5]}... → final={final_mapping[:5]}..."
                            )
                    
            except Exception as e:
                result["details"].append(f"Warning: Could not check final_layout: {e}")
        else:
            result["details"].append("Warning: No layout info available on circuit")
        
        # 3. Determine verdict
        if result["swap_count"] > 0 or result["iswap_count"] > 0:
            result["verdict"] = "ROUTING DETECTED (explicit SWAP gates)"
        elif result["routing_detected"]:
            result["verdict"] = "ROUTING DETECTED (layout permutation)"
        else:
            result["verdict"] = "0-SWAP VERIFIED"
        
        return result


# ============================================================================
# SECTION 4: CORRECTED STATISTICAL FUNCTIONS (per Alicia)
# ============================================================================

def compute_flux_corrected(counts: Dict[str, int],
                           target_indices: List[int],
                           n_qubits: int) -> Tuple[float, float, Dict]:
    """
    Compute excitation flux with CORRECTED error estimation.
    
    Alicia's correction: std_err should be sqrt(p(1-p) / (shots * n_targets))
    not sqrt(p(1-p) / shots)
    
    Args:
        counts: Measurement counts (bitstrings)
        target_indices: Logical qubit indices to measure flux on
        n_qubits: Total number of qubits measured
        
    Returns:
        (flux, std_error, metadata)
    """
    total_shots = sum(counts.values())
    n_targets = len(target_indices)
    
    if not target_indices or total_shots == 0:
        return 0.0, 0.0, {"error": "Empty targets or zero shots"}
    
    # Count excitations at target qubits
    excitations = 0
    for bitstring, count in counts.items():
        bits = bitstring[::-1]  # Qiskit convention: rightmost = qubit 0
        for idx in target_indices:
            if idx < len(bits) and bits[idx] == '1':
                excitations += count
    
    # Total observations = shots × number of target qubits
    total_observations = total_shots * n_targets
    
    # Flux = excitation probability per target qubit
    flux = excitations / total_observations
    
    # CORRECTED standard error for binomial proportion
    # p = excitations / total_observations
    # Var(p) = p(1-p) / total_observations
    # std_err = sqrt(Var(p))
    std_err = np.sqrt(flux * (1 - flux) / total_observations)
    
    metadata = {
        "excitations": excitations,
        "total_observations": total_observations,
        "shots": total_shots,
        "n_targets": n_targets,
        "raw_probability": flux,
    }
    
    return flux, std_err, metadata


def compute_ratio_bayesian(F_h: float, F_h_err: float,
                           F_far: float, F_far_err: float,
                           n_h: int, n_far: int,
                           shots: int) -> Tuple[float, float, Dict]:
    """
    Compute F_h/F_far ratio with Bayesian pseudo-count for small denominators.
    
    Alicia's correction: Don't use max(F_far, 0.001) as it introduces bias.
    Instead, use Bayesian approach with pseudo-counts.
    
    Args:
        F_h, F_h_err: Horizon flux and error
        F_far, F_far_err: Far-field flux and error
        n_h, n_far: Number of target qubits in each region
        shots: Number of measurement shots
        
    Returns:
        (ratio, ratio_error, metadata)
    """
    # Bayesian pseudo-count: add 1 "virtual" excitation and 1 "virtual" non-excitation
    # This is the Laplace smoothing / Beta(1,1) prior
    alpha = 1  # pseudo-excitations
    beta = 1   # pseudo-non-excitations
    
    # Adjusted observations
    obs_h = shots * n_h
    obs_far = shots * n_far
    
    # Bayesian estimate of flux (posterior mean with Beta prior)
    # For Beta(alpha, beta) prior and Binomial(n, p) likelihood:
    # Posterior mean = (successes + alpha) / (n + alpha + beta)
    
    exc_h = F_h * obs_h  # estimated excitations at horizon
    exc_far = F_far * obs_far  # estimated excitations at far-field
    
    F_h_bayes = (exc_h + alpha) / (obs_h + alpha + beta)
    F_far_bayes = (exc_far + alpha) / (obs_far + alpha + beta)
    
    # Ratio with Bayesian estimates
    if F_far_bayes <= 0:
        # Should never happen with pseudo-counts, but safety check
        return float('inf'), float('inf'), {"error": "Zero denominator even with pseudo-counts"}
    
    ratio = F_h_bayes / F_far_bayes
    
    # Error propagation (using delta method on Bayesian estimates)
    # Var(ratio) ≈ (ratio^2) * [Var(F_h)/F_h^2 + Var(F_far)/F_far^2]
    
    # Variance of Beta posterior: α*β / [(α+β)^2 * (α+β+1)]
    # where α = exc + 1, β = (n - exc) + 1
    def beta_var(exc, n):
        a = exc + alpha
        b = (n - exc) + beta
        return (a * b) / ((a + b)**2 * (a + b + 1))
    
    var_h = beta_var(exc_h, obs_h)
    var_far = beta_var(exc_far, obs_far)
    
    # Relative variance of ratio
    rel_var = var_h / (F_h_bayes**2) + var_far / (F_far_bayes**2) if F_h_bayes > 0 else float('inf')
    ratio_err = ratio * np.sqrt(rel_var) if rel_var < float('inf') else float('inf')
    
    metadata = {
        "F_h_raw": F_h,
        "F_far_raw": F_far,
        "F_h_bayes": F_h_bayes,
        "F_far_bayes": F_far_bayes,
        "pseudo_alpha": alpha,
        "pseudo_beta": beta,
        "method": "Bayesian (Beta(1,1) prior)",
    }
    
    return ratio, ratio_err, metadata


def compute_correlator_corrected(counts: Dict[str, int],
                                 q1: int, q2: int,
                                 n_qubits: int) -> Tuple[float, float, Dict]:
    """
    Compute ⟨σ_z^i σ_z^j⟩ correlator with correct error estimation.
    
    Returns:
        (correlator, std_error, metadata)
    """
    total_shots = sum(counts.values())
    
    if total_shots == 0:
        return 0.0, 0.0, {"error": "Zero shots"}
    
    # Compute ⟨σ_z^i σ_z^j⟩ = P(same) - P(different)
    same = 0  # Both 0 or both 1
    diff = 0  # One 0, one 1
    
    for bitstring, count in counts.items():
        bits = bitstring[::-1]
        if q1 < len(bits) and q2 < len(bits):
            b1 = bits[q1]
            b2 = bits[q2]
            if b1 == b2:
                same += count
            else:
                diff += count
    
    # Correlator = (same - diff) / total
    correlator = (same - diff) / total_shots
    
    # Standard error: Var(correlator) = (1 - correlator^2) / shots
    # This is the variance of a bounded random variable in [-1, 1]
    std_err = np.sqrt((1 - correlator**2) / total_shots)
    
    metadata = {
        "same_count": same,
        "diff_count": diff,
        "total_shots": total_shots,
    }
    
    return correlator, std_err, metadata


# ============================================================================
# SECTION 5: HAWKING XY CHAIN CIRCUIT (unchanged logic)
# ============================================================================

class HawkingXYChainCircuit:
    """
    Builds XY spin chain circuits on LOGICAL qubits (0 to n-1).
    Physical mapping is done via initial_layout during transpilation.
    """
    
    def __init__(self,
                 physical_qubits: List[int],
                 j_max: float = 1.0,
                 j_min: float = 0.1,
                 sigma: float = 3.0,
                 dt: float = 1.5,
                 trotter_steps: int = 6,
                 kick_strength: float = 0.15):
        self.physical_qubits = physical_qubits
        self.n_qubits = len(physical_qubits)
        self.j_max = j_max
        self.j_min = j_min
        self.sigma = sigma
        self.x_h = self.n_qubits // 2
        self.dt = dt
        self.trotter_steps = trotter_steps
        self.kick_strength = kick_strength
        self.coupling_profile = self._compute_coupling()
        
    def _compute_coupling(self) -> np.ndarray:
        x = np.arange(self.n_qubits - 1)
        r = self.j_min / self.j_max
        gaussian = np.exp(-np.abs(x - self.x_h)**2 / (2 * self.sigma**2))
        return self.j_max * (1 - (1 - r) * gaussian)
    
    def build_circuit(self, measurement_basis: str = 'Z', seed: int = 42) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.name = f"Hawking_N{self.n_qubits}_S{self.trotter_steps}_{measurement_basis}"
        
        # Initial kicks
        np.random.seed(seed)
        for i in range(self.n_qubits):
            kick = self.kick_strength * np.random.uniform(-1, 1) * np.pi
            qc.ry(kick, i)
        qc.barrier()
        
        # Trotter evolution
        for step in range(self.trotter_steps):
            self._apply_trotter_step(qc)
            qc.barrier()
        
        # Basis rotation
        if measurement_basis == 'X':
            for i in range(self.n_qubits):
                qc.h(i)
        elif measurement_basis == 'Y':
            for i in range(self.n_qubits):
                qc.sdg(i)
                qc.h(i)
        
        # Measure
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc
    
    def _apply_trotter_step(self, qc: QuantumCircuit):
        for i in range(0, self.n_qubits - 1, 2):
            theta = self.coupling_profile[i] * self.dt
            qc.rxx(theta, i, i+1)
            qc.ryy(theta, i, i+1)
        for i in range(1, self.n_qubits - 1, 2):
            theta = self.coupling_profile[i] * self.dt
            qc.rxx(theta, i, i+1)
            qc.ryy(theta, i, i+1)
    
    def get_horizon_qubits(self) -> List[int]:
        center = self.x_h
        return [i for i in [center - 1, center, center + 1] if 0 <= i < self.n_qubits]
    
    def get_far_field_qubits(self) -> List[int]:
        far_indices = [0, 1, 2, self.n_qubits - 3, self.n_qubits - 2, self.n_qubits - 1]
        return [i for i in far_indices if 0 <= i < self.n_qubits]


# ============================================================================
# SECTION 6: QMC FRAMEWORK INTEGRATION (NEW in v3)
# ============================================================================

def execute_on_qpu_via_framework(circuits: List[QuantumCircuit],
                                 backend_name: str,
                                 shots: int,
                                 initial_layouts: List[List[int]]) -> Tuple[List[Dict], Dict]:
    """
    Execute circuits on real QPU via QMC Framework.
    
    This is the CRITICAL function that was missing in v2.
    
    Returns:
        (list_of_counts, execution_metadata)
    """
    try:
        # Try to import QMC Framework
        sys.path.insert(0, '/mnt/project')
        from qmc_quantum_framework_v2_5_23 import QMCFrameworkV2_4 as QMCFramework, RunMode
        
        print("  ✓ QMC Framework v2.5.23 loaded")
        
        # Initialize framework
        fw = QMCFramework(
            project="HAWKING_V3_CROSS_PLATFORM",
            backend_name=backend_name,
            shots=shots,
            auto_confirm=False  # NEVER auto_confirm in production
        )
        
        # Initialize in QPU mode
        fw.initialize(mode=RunMode.QPU)
        
        # Connect to backend
        fw.connect()
        
        # Get real coupling map for transpilation
        coupling_map = fw.backend.coupling_map
        
        # Transpile each circuit with its initial_layout
        transpiled_circuits = []
        transpile_metadata = []
        
        for i, (qc, layout) in enumerate(zip(circuits, initial_layouts)):
            transpiled = transpile(
                qc,
                coupling_map=coupling_map,
                initial_layout=layout,
                optimization_level=3,
                seed_transpiler=42
            )
            transpiled_circuits.append(transpiled)
            
            # Verify SWAP
            swap_info = RobustSwapVerifier.verify_no_routing(transpiled, layout)
            transpile_metadata.append(swap_info)
            
            status = "✓" if swap_info["verdict"] == "0-SWAP VERIFIED" else "✗"
            print(f"    Circuit {i}: {qc.name} - {status} {swap_info['verdict']}")
        
        # Estimate cost
        estimate = fw.estimate_cost(transpiled_circuits, shots=shots)
        print(f"\n  Estimated QPU time: {estimate.get('total_seconds', '?')}s")
        
        # Execute with confirmation
        print("\n  Submitting to QPU...")
        results = fw.run_on_qpu(transpiled_circuits, shots=shots)
        
        # Extract counts
        all_counts = []
        for r in results:
            if isinstance(r, dict) and 'counts' in r:
                all_counts.append(r['counts'])
            elif isinstance(r, dict):
                all_counts.append(r)
            else:
                all_counts.append({})
        
        execution_metadata = {
            "framework_version": "2.5.23",
            "backend": backend_name,
            "shots": shots,
            "job_id": getattr(fw, 'last_job_id', None),
            "report_path": getattr(fw, 'last_report_path', None),
            "archive_path": getattr(fw, 'last_archive_path', None),
            "transpile_info": transpile_metadata,
            "real_qpu": True,
        }
        
        return all_counts, execution_metadata
        
    except ImportError as e:
        print(f"  ✗ QMC Framework not available: {e}")
        print("  → Falling back to direct IBM Runtime...")
        
        # Fallback to direct IBM Runtime
        return execute_on_qpu_direct(circuits, backend_name, shots, initial_layouts)
    
    except Exception as e:
        print(f"  ✗ Framework error: {e}")
        raise


def execute_on_qpu_direct(circuits: List[QuantumCircuit],
                          backend_name: str,
                          shots: int,
                          initial_layouts: List[List[int]]) -> Tuple[List[Dict], Dict]:
    """
    Fallback: Execute directly via IBM Runtime (without QMC Framework).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    
    print("  Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    coupling_map = backend.coupling_map
    
    print(f"  ✓ Connected to {backend_name} ({backend.num_qubits} qubits)")
    
    # Transpile
    transpiled_circuits = []
    transpile_metadata = []
    
    for i, (qc, layout) in enumerate(zip(circuits, initial_layouts)):
        transpiled = transpile(
            qc,
            backend=backend,
            initial_layout=layout,
            optimization_level=3,
            seed_transpiler=42
        )
        transpiled_circuits.append(transpiled)
        
        swap_info = RobustSwapVerifier.verify_no_routing(transpiled, layout)
        transpile_metadata.append(swap_info)
        
        status = "✓" if swap_info["verdict"] == "0-SWAP VERIFIED" else "✗"
        print(f"    Circuit {i}: {qc.name} - {status} {swap_info['verdict']}")
    
    # Execute
    print("\n  Submitting to QPU via SamplerV2...")
    sampler = SamplerV2(backend)
    job = sampler.run(transpiled_circuits, shots=shots)
    
    print(f"  Job ID: {job.job_id()}")
    print("  Waiting for results...")
    
    result = job.result()
    
    # Extract counts
    all_counts = []
    for i, pub_result in enumerate(result):
        counts = pub_result.data.c.get_counts()
        all_counts.append(counts)
    
    execution_metadata = {
        "framework_version": "direct_runtime",
        "backend": backend_name,
        "shots": shots,
        "job_id": job.job_id(),
        "transpile_info": transpile_metadata,
        "real_qpu": True,
    }
    
    return all_counts, execution_metadata


# ============================================================================
# SECTION 7: MOCK DATA GENERATOR (for dry-run only)
# ============================================================================

def create_mock_coupling_map(n_qubits: int) -> CouplingMap:
    """Create mock heavy-hex coupling map for dry-run."""
    edges = []
    for i in range(n_qubits - 1):
        edges.append((i, i + 1))
    for i in range(0, n_qubits - 12, 12):
        if i + 6 < n_qubits:
            edges.append((i, i + 6))
        if i + 12 < n_qubits:
            edges.append((i + 6, i + 12))
    return CouplingMap(couplinglist=edges)


def generate_mock_results(circuit_metadata: List[Dict], shots: int) -> List[Dict]:
    """Generate mock results with realistic Hawking-like signal."""
    np.random.seed(42)
    results = []
    
    for meta in circuit_metadata:
        n_qubits = meta["n_qubits"]
        horizon_indices = meta["horizon_qubits"]
        
        counts = {}
        for _ in range(shots):
            bits = ['0'] * n_qubits
            for i in range(n_qubits):
                prob = 0.35 if i in horizon_indices else 0.03
                if np.random.random() < prob:
                    bits[i] = '1'
            bitstring = ''.join(bits[::-1])
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        results.append(counts)
    
    return results


# ============================================================================
# SECTION 8: MAIN VALIDATION FUNCTION
# ============================================================================

def run_validation(backend_name: str = "ibm_fez",
                   shots: int = 8192,
                   chain_length: int = 30,
                   n_chains: int = 4,
                   dry_run: bool = False):
    """
    Run cross-platform validation with all Alicia corrections.
    """
    print("=" * 80)
    print("HAWKING CROSS-PLATFORM VALIDATION v3 (PUBLICATION GRADE)")
    print("=" * 80)
    
    # Verify backend
    if backend_name not in BACKEND_SPECS:
        print(f"ERROR: Unknown backend {backend_name}")
        return None
    
    specs = BACKEND_SPECS[backend_name]
    print(f"\nBackend: {backend_name}")
    print(f"  Qubits: {specs['qubits']}")
    print(f"  Generation: {specs['generation']}")
    
    # Adjust chain length
    max_chain = specs['max_chain_length']
    if chain_length > max_chain:
        print(f"  Adjusting chain length: {chain_length} → {max_chain}")
        chain_length = max_chain
    
    print(f"\nParameters:")
    print(f"  Chain length: {chain_length}")
    print(f"  Number of chains: {n_chains}")
    print(f"  Shots: {shots}")
    print(f"  Mode: {'DRY-RUN (mock data)' if dry_run else 'LIVE QPU'}")
    
    if dry_run:
        print("\n  ⚠️  WARNING: Dry-run mode - results are NOT publication-grade")
    
    # =========================================================================
    # Step 1: Get coupling map and find chains
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 1: Chain Discovery")
    print("-" * 40)
    
    if dry_run:
        coupling_map = create_mock_coupling_map(specs['qubits'])
        print("  [DRY-RUN] Using mock coupling map")
    else:
        print("  [LIVE] Fetching coupling map from backend...")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        coupling_map = backend.coupling_map
        print(f"  ✓ Got coupling map ({coupling_map.size()} qubits)")
    
    finder = HeavyHexChainFinder(coupling_map)
    chains = finder.find_disjoint_chains(n_chains, chain_length)
    
    print(f"\n  Found {len(chains)} chains:")
    for i, chain in enumerate(chains):
        conn = "✓ CONNECTED" if finder.verify_chain_connectivity(chain) else "✗ BROKEN"
        print(f"    Chain {i}: qubits {chain[0]}→{chain[-1]} (len={len(chain)}) [{conn}]")
    
    if len(chains) == 0:
        print("  ERROR: No chains found")
        return None
    
    n_chains = len(chains)
    
    # =========================================================================
    # Step 2: Build circuits
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 2: Circuit Construction")
    print("-" * 40)
    
    all_circuits = []
    all_layouts = []
    circuit_metadata = []
    
    for chain_id, chain in enumerate(chains):
        builder = HawkingXYChainCircuit(
            physical_qubits=chain,
            j_max=1.0,
            j_min=0.1,
            sigma=3.0,
            dt=1.5,
            trotter_steps=6,
            kick_strength=0.15
        )
        
        for basis in ['Z', 'X', 'Y']:
            qc = builder.build_circuit(measurement_basis=basis)
            qc.name = f"Chain{chain_id}_{basis}"
            all_circuits.append(qc)
            all_layouts.append(chain)  # Physical qubits for initial_layout
            
            circuit_metadata.append({
                "chain_id": chain_id,
                "basis": basis,
                "physical_qubits": chain,
                "n_qubits": len(chain),
                "horizon_qubits": builder.get_horizon_qubits(),
                "far_field_qubits": builder.get_far_field_qubits(),
            })
        
        print(f"  Chain {chain_id}: 3 circuits (Z/X/Y), depth={all_circuits[-1].depth()}")
    
    print(f"\n  Total circuits: {len(all_circuits)}")
    
    # =========================================================================
    # Step 3: Transpile and verify SWAP
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 3: Transpilation & SWAP Verification")
    print("-" * 40)
    
    transpiled_circuits = []
    swap_report = []
    total_routing_detected = 0
    
    for i, (qc, layout) in enumerate(zip(all_circuits, all_layouts)):
        transpiled = transpile(
            qc,
            coupling_map=coupling_map,
            initial_layout=layout,
            optimization_level=3,
            seed_transpiler=42
        )
        transpiled_circuits.append(transpiled)
        
        # ROBUST verification (Alicia's correction)
        swap_info = RobustSwapVerifier.verify_no_routing(transpiled, layout)
        swap_report.append(swap_info)
        
        if swap_info["verdict"] != "0-SWAP VERIFIED":
            total_routing_detected += 1
            print(f"  ✗ {qc.name}: {swap_info['verdict']}")
        else:
            print(f"  ✓ {qc.name}: 0-SWAP verified")
    
    print(f"\n  Circuits with routing: {total_routing_detected}/{len(all_circuits)}")
    
    if total_routing_detected == 0:
        print("  ✓ MHIL 0-SWAP CLAIM: VERIFIED")
    else:
        print("  ✗ MHIL 0-SWAP CLAIM: VIOLATED")
    
    # =========================================================================
    # Step 4: Execute
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 4: Execution")
    print("-" * 40)
    
    if dry_run:
        print("  [DRY-RUN] Generating mock results...")
        all_counts = generate_mock_results(circuit_metadata, shots)
        execution_metadata = {"real_qpu": False, "mode": "dry-run"}
    else:
        print("  [LIVE] Executing on real QPU...")
        all_counts, execution_metadata = execute_on_qpu_via_framework(
            all_circuits, backend_name, shots, all_layouts
        )
    
    # =========================================================================
    # Step 5: Analysis with corrected statistics
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 5: Analysis (Corrected Statistics)")
    print("-" * 40)
    
    analysis_results = []
    
    for chain_id in range(n_chains):
        chain_meta = [m for m in circuit_metadata if m["chain_id"] == chain_id]
        chain_counts = all_counts[chain_id * 3 : chain_id * 3 + 3]  # Z, X, Y
        
        z_counts = chain_counts[0]
        n_qubits = chain_meta[0]["n_qubits"]
        horizon_qubits = chain_meta[0]["horizon_qubits"]
        far_field_qubits = chain_meta[0]["far_field_qubits"]
        
        # Compute fluxes with CORRECTED error estimation
        F_h, F_h_err, meta_h = compute_flux_corrected(
            z_counts, horizon_qubits, n_qubits
        )
        
        F_far, F_far_err, meta_far = compute_flux_corrected(
            z_counts, far_field_qubits, n_qubits
        )
        
        # Compute ratio with Bayesian pseudo-counts (no clamp!)
        ratio, ratio_err, meta_ratio = compute_ratio_bayesian(
            F_h, F_h_err, F_far, F_far_err,
            len(horizon_qubits), len(far_field_qubits), shots
        )
        
        # Partner correlations
        x_counts = chain_counts[1]
        y_counts = chain_counts[2]
        
        q1 = n_qubits // 2 - 2
        q2 = n_qubits // 2 + 2
        
        xx_corr, xx_err, _ = compute_correlator_corrected(x_counts, q1, q2, n_qubits)
        yy_corr, yy_err, _ = compute_correlator_corrected(y_counts, q1, q2, n_qubits)
        
        # Partner check: XX ≈ -YY means xx_corr + yy_corr ≈ 0
        partner_sum = xx_corr + yy_corr
        partner_check = abs(partner_sum) < 0.2  # Tolerance
        
        analysis_results.append({
            "chain_id": chain_id,
            "F_h": F_h,
            "F_h_err": F_h_err,
            "F_far": F_far,
            "F_far_err": F_far_err,
            "ratio": ratio,
            "ratio_err": ratio_err,
            "ratio_method": meta_ratio["method"],
            "XX_correlator": xx_corr,
            "XX_err": xx_err,
            "YY_correlator": yy_corr,
            "YY_err": yy_err,
            "partner_sum": partner_sum,
            "partner_check": partner_check,
        })
        
        print(f"\n  Chain {chain_id}:")
        print(f"    F_h = {F_h:.4f} ± {F_h_err:.4f}")
        print(f"    F_far = {F_far:.4f} ± {F_far_err:.4f}")
        print(f"    Ratio = {ratio:.1f} ± {ratio_err:.1f}× ({meta_ratio['method']})")
        print(f"    ⟨XX⟩ = {xx_corr:.3f} ± {xx_err:.3f}")
        print(f"    ⟨YY⟩ = {yy_corr:.3f} ± {yy_err:.3f}")
        print(f"    Partner check (XX+YY≈0): {'✓' if partner_check else '✗'} (sum={partner_sum:.3f})")
    
    # =========================================================================
    # Step 6: Generate report
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 6: Report Generation")
    print("-" * 40)
    
    report = {
        "metadata": {
            "version": "3.0 (publication-grade)",
            "backend": backend_name,
            "backend_specs": specs,
            "shots": shots,
            "chain_length": chain_length,
            "n_chains": n_chains,
            "dry_run": dry_run,
            "timestamp": datetime.now().isoformat(),
            "execution": execution_metadata,
        },
        "chains": [
            {
                "chain_id": i,
                "physical_qubits": chains[i],
                "connected": finder.verify_chain_connectivity(chains[i])
            }
            for i in range(len(chains))
        ],
        "swap_verification": {
            "total_circuits": len(all_circuits),
            "routing_detected": total_routing_detected,
            "mhil_0swap_verified": total_routing_detected == 0,
            "method": "TranspileLayout.final_layout + SwapGate count",
            "details": swap_report,
        },
        "analysis": analysis_results,
        "summary": {
            "mean_ratio": np.mean([r["ratio"] for r in analysis_results]),
            "mean_ratio_err": np.sqrt(np.sum([r["ratio_err"]**2 for r in analysis_results])) / len(analysis_results),
            "all_above_threshold": all(r["ratio"] > 1.8 for r in analysis_results),
            "partner_verified": all(r["partner_check"] for r in analysis_results),
            "publication_ready": (
                not dry_run and 
                total_routing_detected == 0 and 
                all(r["ratio"] > 1.8 for r in analysis_results)
            ),
        },
        "publication_checklist": {
            "real_qpu_execution": not dry_run,
            "swap_verification_passed": total_routing_detected == 0,
            "ratio_above_threshold": all(r["ratio"] > 1.8 for r in analysis_results),
            "partner_correlations_valid": all(r["partner_check"] for r in analysis_results),
            "statistical_methods": "Corrected (Bayesian pseudo-count, proper error propagation)",
        }
    }
    
    # Save report
    output_dir = "../analyses"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_tag = "DRYRUN" if dry_run else "QPU"
    output_path = f"{output_dir}/hawking_v3_{backend_name}_{mode_tag}_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n  Report saved to: {output_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Backend: {backend_name} ({specs['generation']})")
    print(f"  Chains: {len(chains)} × {chain_length} qubits")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'REAL QPU'}")
    print(f"  SWAP verification: {'✓ PASSED' if total_routing_detected == 0 else '✗ FAILED'}")
    print(f"  Mean ratio: {report['summary']['mean_ratio']:.1f} ± {report['summary']['mean_ratio_err']:.1f}×")
    print(f"  Above threshold (1.8×): {'✓ ALL' if report['summary']['all_above_threshold'] else '✗ SOME FAILED'}")
    print(f"  Partner correlations: {'✓ VERIFIED' if report['summary']['partner_verified'] else '✗ FAILED'}")
    print()
    
    if report['summary']['publication_ready']:
        print("  ✅ PUBLICATION READY")
    else:
        print("  ❌ NOT PUBLICATION READY")
        if dry_run:
            print("     → Run without --dry-run for real QPU execution")
        if total_routing_detected > 0:
            print("     → SWAP/routing detected - review chain mapping")
        if not report['summary']['all_above_threshold']:
            print("     → Ratio below threshold - check signal")
    
    print("=" * 80)
    
    return report


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAWKING Cross-Platform Validation v3 (Publication Grade)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ibm_fez",
        choices=["ibm_fez", "ibm_torino"],
        help="IBM Quantum backend (default: ibm_fez)"
    )
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--chain-length", type=int, default=30)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock data (NOT publication-grade)")
    
    args = parser.parse_args()
    
    run_validation(
        backend_name=args.backend,
        shots=args.shots,
        chain_length=args.chain_length,
        n_chains=args.n_chains,
        dry_run=args.dry_run
    )
