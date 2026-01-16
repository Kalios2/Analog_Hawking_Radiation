# 🌌 Analog Hawking Radiation Validation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-6929C4.svg)](https://qiskit.org/)

Experimental validation of analog Hawking radiation signatures on IBM Quantum superconducting processors. This repository contains reproduction scripts for the paper:

> **"Analog Hawking Radiation on a 156-Qubit Superconducting Processor: Spatial Localization, Temporal Dynamics, and Multi-Universe Validation"**

## 🎯 Key Results

| Metric | Value | Significance |
|--------|-------|--------------|
| **Qubits** | 156 | Largest Hawking analog simulation |
| **Flux Ratio** | 83.2× (optimized) / 44.3× (standard) | 10× threshold exceeded |
| **Shuffle Validation** | 91.6% degradation | p < 0.001 |
| **Peak Accuracy** | 100% at horizon | All configurations |

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Kalios2/Analog_Hawking_Radiation.git

```

### 2. Configure IBM Quantum API Key

Create a `.env` file in the project root:

```bash
# .env
IBM_API_KEY_ACTIVE_BOB=your_ibm_quantum_api_key_here
IBM_API_KEY_JOHN=other_ibm_quantum_api_key_here

```

> 📝 Get your API key at [quantum.ibm.com](https://quantum.ibm.com/) → Account Settings → API Token

### 3. Run Validation

```bash
# Simulator mode (no API key needed)
python hawking_validation.py --mode simulator

# Real QPU execution
python hawking_validation.py --mode qpu --backend ibm_fez
```


## ⚡ Using the QMC Framework

For advanced experiments, we recommend the **QMC Framework** which provides automated error mitigation, multi-backend support, and result archiving.

### Configuration

Create `.env` in your project:

```bash
# .env - QMC Framework Configuration

# IBM Quantum (required for QPU execution ACTIVE for selectr good api key)
IBM_API_KEY_ACTIVE_NAME=your_token_here
```

### Usage

```python
from qmc_framework import QMCExperiment

# Initialize experiment
exp = QMCExperiment(
    name="hawking_validation",
    backend="ibm_fez",          # or "simulator"
    shots=4096
)

# Run Hawking circuit
result = exp.run_hawking(
    n_qubits=40,
    x_horizon=20,
    kick_strength=0.6,
    trotter_steps=1
)

# Results auto-archived with Job ID
print(f"Job ID: {result.job_id}")
print(f"Flux Ratio: {result.flux_ratio:.1f}×")
print(f"Peak Position: {result.peak_site}")
```

### Advanced: Full Validation Suite

```python
from qmc_framework import HAWKINGValidation

# Run complete validation campaign
validation = HAWKINGValidation(backend="ibm_fez")

# Multi-scale test (20, 40, 80 qubits)
results = validation.run_multi_scale(
    scales=[20, 40, 80],
    kicks=[0.4, 0.6, 0.8],
    shots=16384
)

# Shuffle control
shuffle_result = validation.run_shuffle_control(
    n_qubits=40,
    n_permutations=1000
)

# Generate report
validation.generate_report("hawking_validation_report.html")
```

## 📊 Reproducing Paper Results

### Figure 3: Spatial Flux Profile

```bash
python scripts/HAWKING_SCIENTIFIC_VALIDATION.py \
    --experiment spatial_profile \
    --backend ibm_fez \
    --qubits 40
```

### Figure 5: Shuffle Validation

```bash
python scripts/HAWKING_SCIENTIFIC_VALIDATION.py \
    --experiment shuffle_control \
    --backend ibm_fez \
    --permutations 1000
```

### Figure 25: Ratio vs Trotter Steps

```bash
python scripts/HAWKING_RATIO_S_SWEEP.py \
    --backend ibm_fez \
    --trotter-range 1 3
```

## 🔬 Experimental Details

### Circuit Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N` | 20-156 | Number of qubits |
| `x_horizon` | N/2 | Horizon position |
| `J₀` | 1.0 | Base coupling strength |
| `κ` | 0.06-0.14 | Surface gravity parameter |
| `w` | 3.0 | Horizon width |
| `S` | 1-6 | Trotter steps |

### Observable Definitions

**Bond Correlator (flux proxy):**
```
F(link) = ⟨X_i X_{i+1}⟩ + ⟨Y_i Y_{i+1}⟩
```

**Site Density:**
```
n(x) = (1 - ⟨Z_x⟩) / 2
```

### Error Mitigation

- **Standard:** Dynamical Decoupling (XY4) + Pauli Twirling (32 randomizations)
- **Optimized:** DD-XY4 + PT-64 + Zero-Noise Extrapolation (3 scale factors)

## 📄 Citation

```bibtex
@article{hawking2026analog,
  title={Analog Hawking Radiation on a 156-Qubit Superconducting Processor: 
         Spatial Localization, Temporal Dynamics, and Multi-Universe Validation},
  author={[AUTHORS]},
  journal={PRX Quantum},
  year={2026},
  note={In review}
}
```

## 🔗 Related Resources

- [IBM Quantum Documentation](https://quantum.cloud.ibm.com/docs)
- [Qiskit Tutorials](https://qiskit.org/learn)
- [QMC Framework Documentation](https://qmc-framework.readthedocs.io)

## 📧 Contact

- **Lab:** QMC Research Lab, 
- **Issues:** Please use GitHub Issues for bug reports

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

*This research was conducted on IBM Quantum processors via the IBM Quantum Network.*
