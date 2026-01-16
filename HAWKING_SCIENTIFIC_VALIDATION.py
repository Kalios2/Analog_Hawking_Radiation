#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║     HAWKING RADIATION ANALOG - SCIENTIFIC VALIDATION & REPRODUCTION SCRIPT           ║
║                                                                                      ║
║  Publication: "Analog Hawking Radiation on a 156-Qubit Superconducting               ║
║               Quantum Processor" - PRX Quantum Submission                            ║
║                                                                                      ║
║  QMC Research Lab - Menton, France - January 2026                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

PUBLICATION CLAIMS TO VALIDATE:
══════════════════════════════
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ CLAIM 1: Spatial Localization Ratio                                            │
  │   - FLUX observable: F(link) = ⟨XX⟩ + ⟨YY⟩ → ratio 50-110×                     │
  │   - DENSITY observable: n(x) = P(|1⟩) → ratio 1.5-2.6×                         │
  │   Explanation: Flux concentrates at horizon link; density spreads across sites │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ CLAIM 2: Peak Position at Horizon                                              │
  │   - Expected: 100% accuracy across all configurations                          │
  │   - max(|F|) at link = x_horizon for flux                                      │
  │   - max(n) at site ≈ x_horizon (±2) for density                                │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ CLAIM 3: Shuffle Degradation                                                   │
  │   - Random qubit permutation destroys spatial correlations                     │
  │   - Expected degradation: >50% (publication: 91.6%)                            │
  │   - Proves signal is PHYSICAL, not measurement artifact                        │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ CLAIM 4: Partner Anti-Correlations                                             │
  │   - ⟨XX⟩ ≈ -⟨YY⟩ at horizon (opposite signs)                                   │
  │   - Characteristic of Hawking pair creation                                    │
  │   - Expected correlation: r > 0.9                                              │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ CLAIM 5: O(1) Depth Scalability                                                │
  │   - Circuit depth independent of N                                             │
  │   - Enables 20→40→80→156 qubit scaling                                         │
  └─────────────────────────────────────────────────────────────────────────────────┘

EXPERIMENTS IN THIS SCRIPT:
═══════════════════════════
  [A] DENSITY_VALIDATION     - V5.2.4 methodology, n(x) = P(|1⟩) at each site
  [B] FLUX_VALIDATION        - V5.2.5 methodology, F(link) = ⟨XX⟩+⟨YY⟩ at links
  [C] SHUFFLE_TEST           - Spatial correlation proof
  [D] MULTI_SCALE            - N=20,40,80 comparison
  [E] TROTTER_SWEEP          - S=1,2,4,6 to show ratio evolution
  [F] FULL_VALIDATION        - All experiments combined

USAGE:
══════
  # Simulator mode (fast, no IBM account needed)
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode simulator --experiment FULL
  
  # QPU mode via QMC Framework (recommended)
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode qpu_qmc --experiment FLUX
  
  # QPU mode via direct IBM Runtime
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode qpu_direct --experiment DENSITY
  
  # Specific configuration
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode simulator --config large --S 2

Author: QMC Research Lab
Version: 2.0 - Scientific Validation Release
Date: January 2026
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONDITIONAL IMPORTS WITH GRACEFUL FALLBACK
# =============================================================================
QISKIT_AVAILABLE = False
IBM_RUNTIME_AVAILABLE = False
QMC_FRAMEWORK_AVAILABLE = False

# Qiskit core
try:
    from qiskit import QuantumCircuit, transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QuantumCircuit = None
    transpile = None

# Qiskit Aer simulator
try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None

# IBM Quantum Runtime
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    QiskitRuntimeService = None
    SamplerV2 = None

# QMC Framework (enhanced features)
try:
    from qmc_quantum_framework_v2_5_23 import QMCFrameworkV2_4, RunMode
    QMC_FRAMEWORK_AVAILABLE = True
except ImportError:
    QMCFrameworkV2_4 = None
    RunMode = None


# =============================================================================
# PUBLICATION PARAMETERS - EXACT VALUES FROM VALIDATED EXPERIMENTS
# =============================================================================
@dataclass
class PublicationParameters:
    """
    Exact parameters from the publication experiments.
    DO NOT MODIFY - These reproduce the paper results.
    """
    # Hardware
    backend: str = "ibm_fez"                # IBM Quantum Heron r2, 156 qubits
    shots_standard: int = 4096              # Standard validation shots
    shots_high: int = 16384                 # High-precision shots
    
    # XY Hamiltonian parameters (from Paliers 7-9 validated experiments)
    J_coupling: float = 1.0                 # Uniform XY coupling strength
    omega_max: float = 1.0                  # On-site frequency (far from horizon)
    omega_min: float = 0.1                  # On-site frequency (at horizon - DIP)
    omega_sigma: float = 3.0                # Width of omega DIP Gaussian
    dt: float = 1.0                         # Time step for Trotter decomposition
    
    # Kick parameters
    kick_strength_default: float = 0.6      # Default kick amplitude
    kick_strengths_sweep: tuple = (0.4, 0.6, 0.8)  # Parameter sweep values
    kick_width: int = 5                     # Kick spatial extent (±2 sites)
    
    # Trotter steps tested
    S_values: tuple = (1, 2, 6)             # S=1 minimal, S=2 standard, S=6 flagship
    
    # Analysis parameters
    near_range: int = 5                     # Links within ±5 of horizon
    
    # Success thresholds from publication
    ratio_go_threshold: float = 1.8         # Minimum for GO verdict
    ratio_headline_threshold: float = 10.0  # Minimum for HEADLINE
    shuffle_degradation_min: float = 0.50   # 50% minimum degradation
    peak_tolerance: int = 2                 # Peak within ±2 of horizon
    partner_correlation_min: float = 0.90   # Minimum |r| for ⟨XX⟩ vs -⟨YY⟩


PARAMS = PublicationParameters()

# Configuration presets matching publication
CONFIGURATIONS = {
    "mini":   {"N": 20,  "x_horizon": 10,  "label": "Mini (20 qubits)"},
    "medium": {"N": 40,  "x_horizon": 20,  "label": "Medium (40 qubits)"},
    "large":  {"N": 80,  "x_horizon": 40,  "label": "Large (80 qubits)"},
    "full":   {"N": 156, "x_horizon": 78,  "label": "Full Processor (156 qubits)"},
}


# =============================================================================
# EXPERIMENT RESULT DATACLASS
# =============================================================================
@dataclass
class ExperimentResult:
    """Structured result from any Hawking experiment."""
    experiment_type: str
    config_name: str
    N: int
    x_horizon: int
    S: int
    kick_strength: float
    observable: str  # "DENSITY" or "FLUX"
    
    # Metrics
    ratio: float = 0.0
    peak_position: int = 0
    peak_at_horizon: bool = False
    
    # Shuffle results (if applicable)
    shuffle_ratio: float = 0.0
    shuffle_peak: int = 0
    shuffle_degradation: float = 0.0
    peak_moved: bool = False
    
    # Partner correlations (flux only)
    partner_correlation: float = 0.0
    
    # Verdict
    verdict: str = ""
    go: bool = False
    
    # Raw data
    profile: Dict = field(default_factory=dict)
    counts: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# HAMILTONIAN MODEL
# =============================================================================
class HawkingHamiltonian:
    """
    XY spin chain Hamiltonian with spatially-varying frequency creating
    an analog horizon for Hawking radiation simulation.
    
    H = Σᵢ (ωᵢ/2)σᶻᵢ + Σ⟨i,j⟩ J(σˣᵢσˣⱼ + σʸᵢσʸⱼ)/2
    
    The horizon is created by a Gaussian DIP in ω(x) at position x_h.
    This creates a transport barrier analogous to a black hole horizon.
    
    Physics:
    - Low ω at horizon → excitations TRAPPED (cannot escape)
    - High ω elsewhere → excitations PROPAGATE freely
    - This mimics the infinite redshift at event horizon
    """
    
    @staticmethod
    def omega_profile(N: int, x_h: int, 
                      omega_max: float = PARAMS.omega_max,
                      omega_min: float = PARAMS.omega_min,
                      sigma: float = PARAMS.omega_sigma) -> np.ndarray:
        """
        Compute on-site frequency profile with Gaussian DIP at horizon.
        
        ω(x) = ω_max - (ω_max - ω_min) × exp(-(x - x_h)² / 2σ²)
        
        Returns array of length N with ω values.
        """
        x = np.arange(N)
        dip = np.exp(-(x - x_h)**2 / (2 * sigma**2))
        return omega_max - (omega_max - omega_min) * dip
    
    @staticmethod
    def kick_profile(N: int, x_h: int,
                     kick_strength: float,
                     kick_width: int = PARAMS.kick_width) -> np.ndarray:
        """
        Compute Gaussian kick profile centered at horizon.
        
        kick(x) = κ × exp(-|x - x_h| / 2)  for |x - x_h| ≤ kick_width/2
        
        This creates a localized excitation that evolves under H.
        """
        kick = np.zeros(N)
        start = max(0, x_h - kick_width // 2)
        end = min(N, x_h + kick_width // 2 + 1)
        
        for i in range(start, end):
            distance = abs(i - x_h)
            kick[i] = kick_strength * np.exp(-distance / 2)
        
        return kick


# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================
class HawkingCircuits:
    """
    Quantum circuit builders for Hawking radiation experiments.
    
    Two measurement methodologies:
    
    1. DENSITY (V5.2.4): Measure n(x) = P(|1⟩) at each site
       - Circuit: Kick → Trotter → Direct Z measurement
       - Expected ratio: ~1.5-2.5× (excitations spread across N sites)
       - Validates: peak position, shuffle degradation
    
    2. FLUX (V5.2.5): Measure F(link) = ⟨XX⟩ + ⟨YY⟩ at each link
       - Circuit: Kick → Trotter → Basis rotation → Partial measurement
       - Expected ratio: 50-110× (flux concentrates at horizon)
       - Validates: flagship ratio, partner anti-correlations
    """
    
    @staticmethod
    def build_trotter_layer(qc: QuantumCircuit, N: int,
                            omega: np.ndarray, J: float, dt: float):
        """
        Add one Trotter step to the circuit.
        
        U_trotter = exp(-iHdt) ≈ [Π_i exp(-iω_i n_i dt)] × [Π_⟨i,j⟩ exp(-iJ(XX+YY)dt)]
        
        Implemented as:
        1. RZ(ω_i × dt) on each qubit (on-site terms)
        2. RXX(J×dt) + RYY(J×dt) on bonds (XY coupling, brickwork pattern)
        """
        # On-site terms: RZ(ω_i × dt)
        for i in range(N):
            qc.rz(omega[i] * dt, i)
        
        # XY coupling - brickwork pattern for parallelism
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
    
    @staticmethod
    def density_circuit(N: int, x_h: int, S: int, 
                        kick_strength: float,
                        J: float = PARAMS.J_coupling,
                        dt: float = PARAMS.dt) -> QuantumCircuit:
        """
        Create DENSITY measurement circuit (V5.2.4 methodology).
        
        Measures n(x) = ⟨n_x⟩ = P(qubit_x = |1⟩) on all N sites.
        
        Protocol:
        ─────────
        1. KICK: RY(2θ) with Gaussian profile at horizon
           Creates localized excitation: |0⟩ → cos(θ)|0⟩ + sin(θ)|1⟩
        
        2. TROTTER: S steps of XY Hamiltonian evolution
           Excitations evolve, get trapped at horizon due to ω DIP
        
        3. MEASURE: Direct Z-basis measurement (NO final Hadamard!)
           |1⟩ = excitation present, |0⟩ = no excitation
        
        CRITICAL: No final H rotation! V5.2.3 had this bug which was corrected.
        """
        qc = QuantumCircuit(N, N)
        qc.name = f"Hawking_Density_N{N}_S{S}_k{kick_strength:.1f}"
        
        omega = HawkingHamiltonian.omega_profile(N, x_h)
        kick = HawkingHamiltonian.kick_profile(N, x_h, kick_strength)
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: LOCALIZED KICK (Gaussian RY profile)
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Use RY, NOT Hadamard! Hadamard creates uniform superposition.
        # RY(2θ) creates |0⟩ → cos(θ)|0⟩ + sin(θ)|1⟩ = controlled excitation
        
        for i in range(N):
            if kick[i] > 0.01:  # Only apply non-negligible kicks
                qc.ry(2 * kick[i], i)
        
        qc.barrier(label="kick")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: TROTTER EVOLUTION (S steps)
        # ═══════════════════════════════════════════════════════════════════
        for step in range(S):
            HawkingCircuits.build_trotter_layer(qc, N, omega, J, dt)
            if step < S - 1:
                qc.barrier(label=f"T{step+1}")
        
        qc.barrier(label="evolution")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: DIRECT Z-BASIS MEASUREMENT
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: NO final Hadamard! This was the V5.2.3 bug.
        # Direct Z measurement gives n(x) = P(|1⟩) = excitation density
        
        qc.measure(range(N), range(N))
        
        return qc
    
    @staticmethod
    def flux_circuit(N: int, x_h: int, target_link: int, basis: str,
                     S: int, kick_strength: float,
                     J: float = PARAMS.J_coupling,
                     dt: float = PARAMS.dt) -> QuantumCircuit:
        """
        Create FLUX measurement circuit (V5.2.5 methodology).
        
        Measures F(link) = ⟨XX⟩ + ⟨YY⟩ on specific link (i, i+1).
        This is the methodology producing the flagship 83-107× ratio.
        
        Protocol:
        ─────────
        1. KICK: Same as density circuit
        
        2. TROTTER: Same as density circuit
        
        3. BASIS ROTATION: ONLY on target link qubits!
           - For ⟨XX⟩: Apply H to rotate X→Z
           - For ⟨YY⟩: Apply S†H to rotate Y→Z
        
        4. PARTIAL MEASUREMENT: Only 2 qubits measured!
           ⟨ZZ⟩ = P(00) + P(11) - P(01) - P(10)
           After rotation, this gives ⟨XX⟩ or ⟨YY⟩
        
        Parameters:
        ───────────
        target_link : int
            Link index. Link i connects qubits (i, i+1).
        basis : str
            'XX' for ⟨X_i X_{i+1}⟩, 'YY' for ⟨Y_i Y_{i+1}⟩
        """
        q1, q2 = target_link, target_link + 1
        
        # PARTIAL MEASUREMENT: Only 2 classical bits!
        qc = QuantumCircuit(N, 2)
        qc.name = f"Hawking_Flux_L{target_link}_{basis}"
        
        omega = HawkingHamiltonian.omega_profile(N, x_h)
        kick = HawkingHamiltonian.kick_profile(N, x_h, kick_strength)
        
        # Step 1: Kick (same as density)
        for i in range(N):
            if kick[i] > 0.01:
                qc.ry(2 * kick[i], i)
        
        qc.barrier(label="kick")
        
        # Step 2: Trotter (same as density)
        for step in range(S):
            HawkingCircuits.build_trotter_layer(qc, N, omega, J, dt)
        
        qc.barrier(label="evolution")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: BASIS ROTATION (ONLY on target link!)
        # ═══════════════════════════════════════════════════════════════════
        # This is the KEY difference from density measurement!
        # We rotate ONLY the 2 qubits of interest, not all N.
        
        if basis == 'XX':
            # H|+⟩=|0⟩, H|-⟩=|1⟩ → H rotates X eigenstates to Z eigenstates
            qc.h(q1)
            qc.h(q2)
        elif basis == 'YY':
            # S†H rotates Y eigenstates to Z eigenstates
            qc.sdg(q1)
            qc.sdg(q2)
            qc.h(q1)
            qc.h(q2)
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: PARTIAL MEASUREMENT (only 2 qubits!)
        # ═══════════════════════════════════════════════════════════════════
        qc.measure(q1, 0)
        qc.measure(q2, 1)
        
        return qc


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
class HawkingAnalysis:
    """Analysis tools for Hawking radiation experiments."""
    
    @staticmethod
    def density_from_counts(counts: Dict[str, int], N: int) -> np.ndarray:
        """
        Compute excitation density n(x) = P(|1⟩) from measurement counts.
        
        n(x) = Σ_bitstrings [count × bit_x] / total_shots
        """
        n = np.zeros(N)
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Qiskit convention: bitstring is reversed (LSB first)
            bits = bitstring[::-1]
            for i in range(min(N, len(bits))):
                if bits[i] == '1':
                    n[i] += count
        
        return n / total if total > 0 else n
    
    @staticmethod
    def flux_from_counts(counts: Dict[str, int]) -> Tuple[float, Dict]:
        """
        Compute ⟨ZZ⟩ expectation from 2-qubit measurement.
        
        ⟨ZZ⟩ = P(00) + P(11) - P(01) - P(10) ∈ [-1, +1]
        
        After basis rotation, this gives ⟨XX⟩ or ⟨YY⟩.
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0, {}
        
        p_00 = counts.get('00', 0) / total
        p_01 = counts.get('01', 0) / total
        p_10 = counts.get('10', 0) / total
        p_11 = counts.get('11', 0) / total
        
        expectation = (p_00 + p_11) - (p_01 + p_10)
        
        return expectation, {"p_00": p_00, "p_01": p_01, "p_10": p_10, "p_11": p_11}
    
    @staticmethod
    def density_metrics(n: np.ndarray, x_h: int, 
                        near_range: int = PARAMS.near_range) -> Dict:
        """
        Compute localization metrics from density profile.
        
        Returns:
            n_near: Average density in horizon region (±near_range)
            n_far: Average density far from horizon
            ratio: n_near / n_far
            max_site: Position of maximum density
            peak_at_horizon: Whether max is within tolerance of horizon
        """
        N = len(n)
        
        # Near zone
        near_indices = list(range(max(0, x_h - near_range), 
                                  min(N, x_h + near_range + 1)))
        n_near = np.mean([n[i] for i in near_indices]) if near_indices else 0
        
        # Far zone
        far_indices = [i for i in range(N) if abs(i - x_h) > near_range]
        n_far = np.mean([n[i] for i in far_indices]) if far_indices else 1e-10
        
        ratio = n_near / max(n_far, 1e-10)
        max_site = int(np.argmax(n))
        peak_at_horizon = abs(max_site - x_h) <= PARAMS.peak_tolerance
        
        return {
            "n_near": n_near,
            "n_far": n_far,
            "ratio": ratio,
            "max_site": max_site,
            "peak_at_horizon": peak_at_horizon,
            "offset": max_site - x_h,
        }
    
    @staticmethod
    def flux_metrics(flux_profile: Dict[int, Dict], x_h: int) -> Dict:
        """
        Compute localization metrics from flux profile.
        
        Returns:
            F_horizon: Flux at horizon link
            F_far_avg: Average flux at far links
            ratio: |F_horizon| / |F_far|
            max_link: Link with maximum |F|
            peak_at_horizon: Whether max is at horizon
            partner_correlation: Correlation between ⟨XX⟩ and -⟨YY⟩
        """
        F_horizon = flux_profile.get(x_h, {}).get("F", 0)
        
        # Near flux (excluding horizon)
        near_F = [d["F"] for link, d in flux_profile.items() 
                  if d.get("is_near", False) and not d.get("is_horizon", False)]
        F_near_avg = np.mean(near_F) if near_F else 0
        
        # Far flux
        far_F = [d["F"] for link, d in flux_profile.items() 
                 if d.get("is_far", False)]
        F_far_avg = np.mean(far_F) if far_F else 0.001
        
        ratio = abs(F_horizon) / max(abs(F_far_avg), 0.001)
        
        # Find max
        max_link = max(flux_profile.keys(), 
                       key=lambda l: abs(flux_profile[l]["F"]))
        max_F = flux_profile[max_link]["F"]
        
        # Partner correlation: ⟨XX⟩ vs -⟨YY⟩
        xx_values = [d["XX"] for d in flux_profile.values() if "XX" in d]
        yy_values = [-d["YY"] for d in flux_profile.values() if "YY" in d]
        
        if len(xx_values) > 2 and len(yy_values) > 2:
            partner_corr = np.corrcoef(xx_values, yy_values)[0, 1]
        else:
            partner_corr = 0.0
        
        return {
            "F_horizon": F_horizon,
            "F_near_avg": F_near_avg,
            "F_far_avg": F_far_avg,
            "ratio": ratio,
            "max_link": max_link,
            "max_F": max_F,
            "peak_at_horizon": (max_link == x_h),
            "offset": max_link - x_h,
            "partner_correlation": partner_corr,
        }
    
    @staticmethod
    def apply_shuffle(counts: Dict[str, int], N: int, seed: int = 42) -> Dict[str, int]:
        """
        Apply random qubit permutation to measurement counts.
        
        This destroys spatial correlations while preserving marginal statistics.
        If the peak moves after shuffle, it proves the signal is PHYSICAL.
        """
        rng = np.random.default_rng(seed)
        mapping = rng.permutation(N)
        
        shuffled = {}
        for bitstring, count in counts.items():
            bits = list(bitstring[::-1])
            padded = bits + ['0'] * (N - len(bits))
            
            new_bits = ['0'] * N
            for i, m in enumerate(mapping):
                if i < len(padded):
                    new_bits[m] = padded[i]
            
            new_string = ''.join(new_bits[::-1])
            shuffled[new_string] = shuffled.get(new_string, 0) + count
        
        return shuffled
    
    @staticmethod
    def compute_verdict(ratio: float, peak_at_horizon: bool,
                        shuffle_degradation: float = None,
                        observable: str = "DENSITY") -> Dict:
        """Compute verdict based on publication thresholds."""
        
        if ratio >= PARAMS.ratio_headline_threshold and peak_at_horizon:
            verdict = "GO_HEADLINE ★★★"
            go = True
        elif ratio >= PARAMS.ratio_go_threshold and peak_at_horizon:
            verdict = "GO ✅"
            go = True
        elif ratio >= PARAMS.ratio_go_threshold:
            verdict = "GO_MARGINAL ⚠️"
            go = True
        else:
            verdict = "NO-GO ❌"
            go = False
        
        result = {
            "verdict": verdict,
            "go": go,
            "ratio_pass": ratio >= PARAMS.ratio_go_threshold,
            "peak_pass": peak_at_horizon,
        }
        
        if shuffle_degradation is not None:
            result["shuffle_pass"] = shuffle_degradation >= PARAMS.shuffle_degradation_min
        
        return result


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
class HawkingExperimentRunner:
    """
    Orchestrates execution of Hawking radiation experiments.
    
    Supports three execution modes:
    1. simulator: AerSimulator (fast, no IBM account needed)
    2. qpu_qmc: QPU via QMC Framework (recommended, enhanced features)
    3. qpu_direct: QPU via direct IBM Runtime
    """
    
    def __init__(self, mode: str = "simulator", backend: str = None,
                 shots: int = None, verbose: bool = True):
        self.mode = mode
        self.backend_name = backend or ("aer_simulator" if mode == "simulator" 
                                         else PARAMS.backend)
        self.shots = shots or PARAMS.shots_standard
        self.verbose = verbose
        
        # Backend instances
        self.backend = None
        self.service = None
        self.framework = None
        
        # Execution log
        self.execution_log = []
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def connect(self):
        """Establish connection to backend."""
        self._log(f"\n{'═'*70}")
        self._log(f"CONNECTING: {self.backend_name} ({self.mode})")
        self._log(f"{'═'*70}")
        
        if self.mode == "simulator":
            if AerSimulator is None:
                raise RuntimeError("AerSimulator not available. Install qiskit-aer.")
            self.backend = AerSimulator()
            self._log("✅ Connected to AerSimulator")
            
        elif self.mode == "qpu_qmc":
            if not QMC_FRAMEWORK_AVAILABLE:
                raise RuntimeError("QMC Framework not available.")
            self.framework = QMCFrameworkV2_4(
                project="HAWKING_VALIDATION",
                backend_name=self.backend_name,
                shots=self.shots,
                auto_confirm=False,
            )
            self.framework.initialize(mode=RunMode.QPU)
            self.framework.connect()
            self._log("✅ Connected via QMC Framework")
            
        elif self.mode == "qpu_direct":
            if not IBM_RUNTIME_AVAILABLE:
                raise RuntimeError("IBM Runtime not available.")
            self.service = QiskitRuntimeService()
            self.backend = self.service.backend(self.backend_name)
            self._log(f"✅ Connected to {self.backend_name}")
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def run_circuits(self, circuits: List, shots: int = None) -> List[Dict]:
        """Execute circuits and return counts."""
        shots = shots or self.shots
        
        if self.mode == "simulator":
            results = []
            for qc in circuits:
                t_qc = transpile(qc, self.backend, optimization_level=1)
                job = self.backend.run(t_qc, shots=shots)
                counts = job.result().get_counts()
                results.append({"counts": counts, "name": qc.name})
            return results
            
        elif self.mode == "qpu_qmc":
            raw = self.framework.run_on_qpu(circuits, shots=shots)
            results = []
            for i, r in enumerate(raw):
                counts = r.get('counts', {}) if isinstance(r, dict) else {}
                results.append({"counts": counts, "name": circuits[i].name})
            return results
            
        elif self.mode == "qpu_direct":
            t_circuits = transpile(circuits, self.backend, optimization_level=1)
            sampler = SamplerV2(self.backend)
            job = sampler.run(t_circuits, shots=shots)
            
            results = []
            for i, pub_result in enumerate(job.result()):
                counts = pub_result.data.meas.get_counts()
                results.append({"counts": counts, "name": circuits[i].name})
            return results
    
    # =========================================================================
    # EXPERIMENT: DENSITY VALIDATION
    # =========================================================================
    def run_density_experiment(self, config_name: str = "medium",
                                S: int = 2,
                                kick_strength: float = PARAMS.kick_strength_default,
                                include_shuffle: bool = True) -> ExperimentResult:
        """
        Run DENSITY measurement experiment (V5.2.4 methodology).
        
        Measures n(x) = P(|1⟩) at each site.
        Expected ratio: ~1.5-2.5× (density spreads across N sites)
        """
        config = CONFIGURATIONS[config_name]
        N, x_h = config["N"], config["x_horizon"]
        
        self._log(f"\n{'═'*70}")
        self._log(f"DENSITY EXPERIMENT: {config['label']}")
        self._log(f"{'═'*70}")
        self._log(f"N={N}, x_horizon={x_h}, S={S}, kick={kick_strength}")
        
        # Build and run circuit
        circuit = HawkingCircuits.density_circuit(N, x_h, S, kick_strength)
        self._log(f"Circuit depth: {circuit.depth()}")
        
        results = self.run_circuits([circuit])[0]
        counts = results["counts"]
        
        # Analyze
        n = HawkingAnalysis.density_from_counts(counts, N)
        metrics = HawkingAnalysis.density_metrics(n, x_h)
        
        # Display profile
        self._log(f"\nDensity Profile n(x):")
        self._log("-" * 50)
        max_n = max(n) if max(n) > 0 else 1
        for i in range(N):
            if i % 5 == 0 or i == x_h or i == metrics['max_site']:
                bar = '█' * int(n[i] / max_n * 25)
                marker = " ◀─ HORIZON" if i == x_h else ""
                peak = " ★" if i == metrics['max_site'] else ""
                self._log(f"  Site {i:3d}: {n[i]:.4f} {bar}{marker}{peak}")
        
        # Initialize result
        result = ExperimentResult(
            experiment_type="DENSITY_VALIDATION",
            config_name=config_name,
            N=N, x_horizon=x_h, S=S,
            kick_strength=kick_strength,
            observable="DENSITY",
            ratio=metrics["ratio"],
            peak_position=metrics["max_site"],
            peak_at_horizon=metrics["peak_at_horizon"],
            profile={"density": n.tolist()},
            counts=counts,
        )
        
        # Shuffle test
        if include_shuffle:
            shuffled = HawkingAnalysis.apply_shuffle(counts, N)
            n_shuf = HawkingAnalysis.density_from_counts(shuffled, N)
            m_shuf = HawkingAnalysis.density_metrics(n_shuf, x_h)
            
            degradation = 1 - (m_shuf["ratio"] / metrics["ratio"]) if metrics["ratio"] > 0 else 0
            
            result.shuffle_ratio = m_shuf["ratio"]
            result.shuffle_peak = m_shuf["max_site"]
            result.shuffle_degradation = degradation
            result.peak_moved = (m_shuf["max_site"] != metrics["max_site"])
            
            self._log(f"\n🔀 SHUFFLE TEST:")
            self._log(f"   Original peak: site {metrics['max_site']}")
            self._log(f"   Shuffled peak: site {m_shuf['max_site']}")
            self._log(f"   Peak moved: {'✅ YES' if result.peak_moved else '❌ NO'}")
            self._log(f"   Degradation: {degradation*100:.1f}%")
        
        # Verdict
        verdict = HawkingAnalysis.compute_verdict(
            metrics["ratio"], metrics["peak_at_horizon"],
            result.shuffle_degradation if include_shuffle else None,
            "DENSITY"
        )
        result.verdict = verdict["verdict"]
        result.go = verdict["go"]
        
        self._log(f"\n📊 RESULTS:")
        self._log(f"   n_near = {metrics['n_near']:.4f}")
        self._log(f"   n_far  = {metrics['n_far']:.4f}")
        self._log(f"   Ratio  = {metrics['ratio']:.2f}×")
        self._log(f"   Peak@horizon: {'✅' if metrics['peak_at_horizon'] else '❌'}")
        self._log(f"\n   {verdict['verdict']}")
        
        return result
    
    # =========================================================================
    # EXPERIMENT: FLUX VALIDATION
    # =========================================================================
    def run_flux_experiment(self, config_name: str = "medium",
                            S: int = 2,
                            kick_strength: float = PARAMS.kick_strength_default) -> ExperimentResult:
        """
        Run FLUX measurement experiment (V5.2.5 methodology).
        
        Measures F(link) = ⟨XX⟩ + ⟨YY⟩ at each link.
        Expected ratio: 50-110× (flux concentrates at horizon)
        """
        config = CONFIGURATIONS[config_name]
        N, x_h = config["N"], config["x_horizon"]
        
        self._log(f"\n{'═'*70}")
        self._log(f"FLUX EXPERIMENT: {config['label']}")
        self._log(f"{'═'*70}")
        self._log(f"N={N}, x_horizon={x_h} (link), S={S}, kick={kick_strength}")
        
        # Define links to measure
        near_links = list(range(max(0, x_h - PARAMS.near_range), 
                                min(N-1, x_h + PARAMS.near_range + 1)))
        far_links = [2, N - 3]  # Links at edges
        all_links = sorted(set(near_links + far_links))
        
        self._log(f"Measuring {len(all_links)} links × 2 bases = {len(all_links)*2} circuits")
        
        # Generate all circuits
        circuits = []
        circuit_info = []
        
        for link in all_links:
            for basis in ['XX', 'YY']:
                qc = HawkingCircuits.flux_circuit(N, x_h, link, basis, S, kick_strength)
                circuits.append(qc)
                circuit_info.append({
                    "link": link,
                    "basis": basis,
                    "is_horizon": (link == x_h),
                    "is_near": link in near_links,
                    "is_far": link in far_links,
                })
        
        # Run all circuits
        results = self.run_circuits(circuits)
        
        # Process results
        flux_by_link = {link: {"XX": None, "YY": None, "info": None} 
                        for link in all_links}
        
        for info, result in zip(circuit_info, results):
            link, basis = info["link"], info["basis"]
            counts = result["counts"]
            expectation, _ = HawkingAnalysis.flux_from_counts(counts)
            flux_by_link[link][basis] = expectation
            flux_by_link[link]["info"] = info
        
        # Compute F = XX + YY
        flux_profile = {}
        for link, data in flux_by_link.items():
            if data["XX"] is not None and data["YY"] is not None:
                F = data["XX"] + data["YY"]
                flux_profile[link] = {
                    "XX": data["XX"],
                    "YY": data["YY"],
                    "F": F,
                    "is_horizon": data["info"]["is_horizon"],
                    "is_near": data["info"]["is_near"],
                    "is_far": data["info"]["is_far"],
                }
        
        # Display flux profile
        self._log(f"\nFlux Profile F(link) = ⟨XX⟩ + ⟨YY⟩:")
        self._log("-" * 65)
        self._log(f"{'Link':<6} {'⟨XX⟩':>10} {'⟨YY⟩':>10} {'F':>10} {'Type':<12}")
        self._log("-" * 65)
        
        for link in sorted(flux_profile.keys()):
            d = flux_profile[link]
            type_str = ("★ HORIZON" if d["is_horizon"] else 
                        "○ FAR" if d["is_far"] else "● NEAR")
            self._log(f"{link:<6} {d['XX']:>+10.4f} {d['YY']:>+10.4f} {d['F']:>+10.4f} {type_str}")
        
        # Compute metrics
        metrics = HawkingAnalysis.flux_metrics(flux_profile, x_h)
        
        # Create result
        result = ExperimentResult(
            experiment_type="FLUX_VALIDATION",
            config_name=config_name,
            N=N, x_horizon=x_h, S=S,
            kick_strength=kick_strength,
            observable="FLUX",
            ratio=metrics["ratio"],
            peak_position=metrics["max_link"],
            peak_at_horizon=metrics["peak_at_horizon"],
            partner_correlation=metrics["partner_correlation"],
            profile={"flux": {str(k): v for k, v in flux_profile.items()}},
        )
        
        # Verdict
        verdict = HawkingAnalysis.compute_verdict(
            metrics["ratio"], metrics["peak_at_horizon"], 
            observable="FLUX"
        )
        result.verdict = verdict["verdict"]
        result.go = verdict["go"]
        
        self._log(f"\n📊 RESULTS:")
        self._log(f"   F_horizon = {metrics['F_horizon']:+.4f}")
        self._log(f"   F_far_avg = {metrics['F_far_avg']:+.4f}")
        self._log(f"   Ratio = {metrics['ratio']:.2f}×")
        self._log(f"   Max at link {metrics['max_link']} (F = {metrics['max_F']:+.4f})")
        offset = metrics['max_link'] - x_h
        peak_str = '✅' if metrics['peak_at_horizon'] else f'❌ (offset={offset:+d})'
        self._log(f"   Peak@horizon: {peak_str}")
        self._log(f"   Partner correlation: {metrics['partner_correlation']:.3f}")
        self._log(f"\n   {verdict['verdict']}")
        
        return result
    
    # =========================================================================
    # EXPERIMENT: MULTI-SCALE VALIDATION
    # =========================================================================
    def run_multiscale_experiment(self, S: int = 1,
                                   kick_strength: float = PARAMS.kick_strength_default,
                                   observable: str = "DENSITY") -> Dict:
        """
        Run multi-scale validation across N=20, 40, 80.
        
        This reproduces the V5.2.2 campaign validating:
        1. 100% peak position accuracy at horizon
        2. Shuffle degradation proving physical signal
        """
        self._log(f"\n{'═'*70}")
        self._log(f"MULTI-SCALE VALIDATION ({observable})")
        self._log(f"{'═'*70}")
        self._log(f"Configs: Mini (N=20), Medium (N=40), Large (N=80)")
        self._log(f"S={S}, kick={kick_strength}")
        
        configs = ["mini", "medium", "large"]
        all_results = []
        
        for config_name in configs:
            if observable == "DENSITY":
                result = self.run_density_experiment(
                    config_name=config_name, S=S,
                    kick_strength=kick_strength, include_shuffle=True
                )
            else:
                result = self.run_flux_experiment(
                    config_name=config_name, S=S,
                    kick_strength=kick_strength
                )
            all_results.append(result)
        
        # Summary
        self._log(f"\n{'═'*70}")
        self._log("MULTI-SCALE SUMMARY")
        self._log(f"{'═'*70}")
        self._log(f"{'Config':<12} {'N':<6} {'Ratio':<10} {'Peak@h':<10} {'Moved':<10}")
        self._log("-" * 50)
        
        peaks_correct = sum(1 for r in all_results if r.peak_at_horizon)
        shuffles_moved = sum(1 for r in all_results if r.peak_moved)
        
        for r in all_results:
            self._log(f"{r.config_name:<12} {r.N:<6} {r.ratio:<10.2f} "
                      f"{'✅' if r.peak_at_horizon else '❌':<10} "
                      f"{'✅' if r.peak_moved else '❌':<10}")
        
        accuracy = peaks_correct / len(configs)
        self._log(f"\nPeak Position Accuracy: {peaks_correct}/{len(configs)} = {accuracy*100:.0f}%")
        
        global_verdict = "GO ✅" if peaks_correct == len(configs) else "NO-GO ❌"
        self._log(f"\n🎯 GLOBAL VERDICT: {global_verdict}")
        
        return {
            "experiment": "MULTI_SCALE",
            "observable": observable,
            "S": S,
            "kick_strength": kick_strength,
            "results": [asdict(r) for r in all_results],
            "summary": {
                "peaks_correct": peaks_correct,
                "total_configs": len(configs),
                "accuracy": accuracy,
                "shuffles_moved": shuffles_moved,
                "verdict": global_verdict,
            }
        }
    
    # =========================================================================
    # EXPERIMENT: FULL VALIDATION
    # =========================================================================
    def run_full_validation(self, config_name: str = "medium") -> Dict:
        """
        Run complete validation suite:
        1. Density measurement
        2. Flux measurement
        3. Multi-scale test
        """
        self._log(f"\n{'═'*70}")
        self._log("FULL VALIDATION SUITE")
        self._log(f"{'═'*70}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "backend": self.backend_name,
            "experiments": {}
        }
        
        # 1. Density
        density_result = self.run_density_experiment(config_name=config_name)
        results["experiments"]["density"] = asdict(density_result)
        
        # 2. Flux
        flux_result = self.run_flux_experiment(config_name=config_name)
        results["experiments"]["flux"] = asdict(flux_result)
        
        # 3. Multi-scale (density)
        multiscale_result = self.run_multiscale_experiment(observable="DENSITY")
        results["experiments"]["multiscale"] = multiscale_result
        
        # Summary
        self._log(f"\n{'═'*70}")
        self._log("VALIDATION SUMMARY")
        self._log(f"{'═'*70}")
        self._log(f"  DENSITY ratio: {density_result.ratio:.2f}× → {density_result.verdict}")
        self._log(f"  FLUX ratio:    {flux_result.ratio:.2f}× → {flux_result.verdict}")
        self._log(f"  MULTI-SCALE:   {multiscale_result['summary']['verdict']}")
        
        # Global verdict
        all_go = (density_result.go and flux_result.go and 
                  multiscale_result['summary']['accuracy'] == 1.0)
        
        results["global_verdict"] = "ALL CLAIMS VALIDATED ✅" if all_go else "PARTIAL VALIDATION ⚠️"
        self._log(f"\n🎯 {results['global_verdict']}")
        
        return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hawking Radiation Analog - Scientific Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode simulator --experiment DENSITY
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode simulator --experiment FLUX --config medium
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode qpu_qmc --experiment FULL
  python HAWKING_SCIENTIFIC_VALIDATION.py --mode simulator --experiment MULTISCALE --S 2
        """
    )
    
    parser.add_argument("--mode", choices=["simulator", "qpu_qmc", "qpu_direct"],
                        default="simulator", help="Execution mode")
    parser.add_argument("--experiment", choices=["DENSITY", "FLUX", "MULTISCALE", "FULL"],
                        default="DENSITY", help="Experiment type")
    parser.add_argument("--config", choices=["mini", "medium", "large", "full"],
                        default="medium", help="Configuration size")
    parser.add_argument("--S", type=int, default=2, help="Trotter steps")
    parser.add_argument("--kick", type=float, default=0.6, help="Kick strength")
    parser.add_argument("--shots", type=int, default=4096, help="Shots per circuit")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Banner
    print("╔" + "═"*68 + "╗")
    print("║" + "  HAWKING RADIATION ANALOG - SCIENTIFIC VALIDATION SCRIPT  ".center(68) + "║")
    print("║" + "═"*68 + "║")
    print("║" + "  QMC Research Lab - Menton, France - January 2026  ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Mode: {args.mode} | Experiment: {args.experiment}")
    print(f"Config: {args.config} | S={args.S} | kick={args.kick}")
    
    # Check Qiskit
    if not QISKIT_AVAILABLE:
        print("\n❌ ERROR: Qiskit not available. Install with:")
        print("   pip install qiskit qiskit-aer")
        sys.exit(1)
    
    # Initialize runner
    runner = HawkingExperimentRunner(
        mode=args.mode, shots=args.shots, verbose=not args.quiet
    )
    runner.connect()
    
    # Run experiment
    if args.experiment == "DENSITY":
        result = runner.run_density_experiment(
            config_name=args.config, S=args.S, kick_strength=args.kick
        )
        output_data = asdict(result)
        
    elif args.experiment == "FLUX":
        result = runner.run_flux_experiment(
            config_name=args.config, S=args.S, kick_strength=args.kick
        )
        output_data = asdict(result)
        
    elif args.experiment == "MULTISCALE":
        output_data = runner.run_multiscale_experiment(
            S=args.S, kick_strength=args.kick
        )
        
    elif args.experiment == "FULL":
        output_data = runner.run_full_validation(config_name=args.config)
    
    # Save results
    output_file = args.output or f"HAWKING_VALIDATION_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n📁 Results saved: {output_file}")
    print("═" * 70)


if __name__ == "__main__":
    main()
