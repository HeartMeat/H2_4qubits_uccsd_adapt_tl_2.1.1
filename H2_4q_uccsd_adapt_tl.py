import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scipy.optimize import minimize

# TkAgg 백엔드로 회로/그래프를 OS 창에 표시
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit
# Qiskit 2.1+ 권장: TwoLocal/NLocal 클래스 대신 함수형 빌더 n_local 사용
from qiskit.circuit.library.n_local import n_local

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.units import DistanceUnit

# =============================
# 실행/최적화/시각화 설정
# =============================
H2_BOND_LENGTH = 0.7414           # H2 평형 결합 길이(Å)
GLOBAL_SEED = 42                   # 재현성 시드
PENALTY_BETA = 3.0                 # VQD 오버랩 패널티 계수(클수록 직교화 강함)

OPTIMIZATION_METHOD = "COBYLA"     # scipy.optimize.minimize 메서드
MAXIMUM_ITERATION = 15000          # 최적화 최대 반복
COBYLA_RHOBEG = 0.2                # COBYLA 초기 스텝 크기
ATTEMPT_COUNT = 2                  # 서로 다른 초기값 재시도 횟수

TL_ENTANGLEMENT = "linear"         # n_local entanglement 패턴 (linear/sca/circular 등)
TL_REPS = 2                        # n_local 반복 횟수

INIT_EPS = 1e-4                    # Stage 2 near-zero 초기값(jitter) 스케일

PRINT_H_TERMS = False              # 해밀토니안 개별 항 상세 출력 여부
DRAW_CIRCUITS = True               # 회로 그림 생성 여부
SHOW_FIGURES = True                # 그림을 즉시 표시할지 여부

FIGSIZE = (25, 8)                  # 회로 그림 크기
FONTSIZE = 8                       # 회로 내부 폰트 크기
FOLD = 30                          # 회로 도식 줄바꿈 폭

STAGE1_NUM_STATES = 6              # Stage 1에서 변분으로 구할 상태 수

# =============================
# 그림 레지스트리(한 번에 show/close)
# =============================
_OPEN_FIGS: List[plt.Figure] = []

def _register_fig(fig: Optional[plt.Figure]):
    """생성된 Figure를 레지스트리에 등록한다(추후 일괄 show/close용)."""
    if fig is not None:
        _OPEN_FIGS.append(fig)

def show_all_figures_blocking():
    """등록된 모든 Figure를 한 번만 표시하고 닫아 Tk 자원을 정리한다."""
    if not _OPEN_FIGS:
        return
    if SHOW_FIGURES:
        plt.show()
    plt.close('all')
    _OPEN_FIGS.clear()

# =============================
# Problem/Hamiltonian 유틸
# =============================
def print_hamiltonian_summary(hamiltonian, verbose: bool = False):
    """
    Qubit Hamiltonian 요약 출력.
    - num_qubits, 파울리 항 개수 등 핵심 정보만 기본 출력
    - verbose=True면 항별 계수까지 표시(성능상 주의)
    """
    print("\n[Hamiltonian]")
    try:
        num_qubits = hamiltonian.num_qubits
        num_terms = len(hamiltonian.paulis)
        print(f"- Qubits: {num_qubits}")
        print(f"- #Pauli terms: {num_terms}")
        if verbose:
            coeffs = hamiltonian.coeffs
            paulis = hamiltonian.paulis
            print("  Terms:")
            for c, p in zip(coeffs, paulis):
                c_val = complex(c)
                c_print = float(c_val.real) if abs(c_val.imag) < 1e-12 else c_val
                sign = "+" if (c_print if isinstance(c_print, float) else c_print.real) >= 0 else "-"
                mag = abs(c_print if isinstance(c_print, float) else c_print)
                print(f"  {sign} {mag:.16f} * {p}")
    except Exception:
        print("- (Operator 요약 형식을 지원하지 않는 타입)")

def build_h2_problem(bond_length: float = H2_BOND_LENGTH):
    """
    H2 전자구조 문제를 생성하고 (problem, e_nuc)를 반환한다.
    - problem: Qiskit Nature Second-quantized 문제(전자부 해밀토니안 포함)
    - e_nuc  : 핵-핵 반발 에너지(총 에너지 상수항)
    """
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    e_nuc = float(problem.hamiltonian.nuclear_repulsion_energy or 0.0)

    print(f"[Molecule] H₂ @ {bond_length:.4f} Å")
    print(f"- #Electrons: {problem.num_particles}")
    print(f"- #Spatial orbitals: {problem.num_spatial_orbitals}")
    return problem, e_nuc

# =============================
# Ansatz Builder (UCCSD / n_local)
# =============================
@dataclass
class AnsatzBuilder:
    """
    평가용/도식용 회로를 생성하는 팩토리.
    - mode='uccsd'  : HartreeFock + UCCSD (Stage 1)
    - mode='nlocal' : Frozen 초기회로 + n_local(ry, rxx) (Stage 2)
    도식용 회로는 게이트 구조 가독성을 위해 decompose 정책을 다르게 유지.
    """
    num_spatial_orbitals: int
    num_particles: Tuple[int, int] | int
    mapper: JordanWignerMapper
    mode: str
    frozen_initial: Optional[QuantumCircuit] = None
    tl_reps: int = TL_REPS
    tl_entanglement: str = TL_ENTANGLEMENT

    num_parameters: Optional[int] = None
    eval_circuit: Optional[QuantumCircuit] = None
    draw_circuit_template: Optional[QuantumCircuit] = None

    def create(self, num_qubits: int) -> QuantumCircuit:
        """모드에 맞는 평가/도식 회로를 생성하고 평가용 회로를 반환."""
        if self.mode == 'uccsd':
            hf = HartreeFock(
                num_spatial_orbitals=self.num_spatial_orbitals,
                num_particles=self.num_particles,
                qubit_mapper=self.mapper,
            )
            ucc = UCCSD(
                num_spatial_orbitals=self.num_spatial_orbitals,
                num_particles=self.num_particles,
                qubit_mapper=self.mapper,
                generalized=True,
                preserve_spin=False,
                include_imaginary=False,
            )

            # 도식용: UCCSD를 1회 decompose(구조 가독성)
            draw = QuantumCircuit(num_qubits)
            draw.compose(hf, inplace=True)
            draw.barrier()
            draw.compose(ucc.decompose(reps=1), inplace=True)
            self.draw_circuit_template = draw

            # 평가용: UCCSD를 2회 decompose(표준 게이트 기반 평가)
            eval_circ = QuantumCircuit(num_qubits)
            eval_circ.compose(hf, inplace=True)
            eval_circ.barrier()
            eval_circ.compose(ucc.decompose(reps=2), inplace=True)
            self.eval_circuit = eval_circ

            self.num_parameters = ucc.num_parameters

        elif self.mode == 'nlocal':
            if self.frozen_initial is None:
                raise ValueError("'nlocal' 모드는 frozen_initial 회로가 필요하다.")

            # 도식용: frozen + n_local(비-decompose)로 원 패턴 그대로 시각화
            tl_draw = n_local(
                num_qubits=num_qubits,
                rotation_blocks='ry',
                entanglement_blocks='rxx',
                entanglement=self.tl_entanglement,
                reps=self.tl_reps,
                skip_final_rotation_layer=False,
            )
            draw = QuantumCircuit(num_qubits)
            draw.compose(self.frozen_initial, inplace=True)
            draw.barrier()
            draw.compose(tl_draw, inplace=True)
            self.draw_circuit_template = draw

            # 평가용: n_local을 2회 decompose하여 표준 게이트로 확장
            tl_eval = n_local(
                num_qubits=num_qubits,
                rotation_blocks='ry',
                entanglement_blocks='rxx',
                entanglement=self.tl_entanglement,
                reps=self.tl_reps,
                skip_final_rotation_layer=False,
            ).decompose(reps=2)
            eval_circ = QuantumCircuit(num_qubits)
            eval_circ.compose(self.frozen_initial, inplace=True)
            eval_circ.barrier()
            eval_circ.compose(tl_eval, inplace=True)
            self.eval_circuit = eval_circ

            self.num_parameters = tl_draw.num_parameters
        else:
            raise ValueError("Unknown ansatz mode: use 'uccsd' or 'nlocal'.")

        return self.eval_circuit  # type: ignore[return-value]

# =============================
# VQD(Variational Quantum Deflation)
# =============================
class VQD:
    """
    내부/외부 오버랩 패널티를 포함한 VQD 구현.
    - 내부 패널티: 이전에 찾은 상태들과의 |⟨ψ_k|ψ_i⟩|² 합 (상호 직교 유도)
    - 외부 패널티: 제공된 기준 회로들과의 |⟨ψ_k|φ_ref⟩|² 합 (상태 보존/분리 유도)
    - 에너지는 전자부 기대값 + 핵-핵 반발 상수(e_nuc)를 더해 총 에너지로 취급
    """
    def __init__(
            self,
            qubit_hamiltonian,
            num_qubits: int,
            energy_shift: float = 0.0,
            beta: float = PENALTY_BETA,
            seed: int = GLOBAL_SEED,
            ansatz: Optional[AnsatzBuilder] = None,
            mapper: Optional[JordanWignerMapper] = None,
            num_spatial_orbitals: Optional[int] = None,
            num_particles: Optional[Tuple[int, int]] = None,
            external_reference_circuits: Optional[List[QuantumCircuit]] = None,
            init_mode: str = 'random',
    ):
        self.H = qubit_hamiltonian
        self.num_qubits = num_qubits
        self.energy_shift = float(energy_shift)
        self.beta = beta
        self.rng = np.random.default_rng(seed)

        self.mapper = mapper
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_particles = num_particles

        if ansatz is None:
            if self.mapper is None or self.num_spatial_orbitals is None or self.num_particles is None:
                raise ValueError("Auto-ansatz에는 mapper/궤도 수/입자 수 정보가 필요하다.")
            ansatz = AnsatzBuilder(
                self.num_spatial_orbitals,
                self.num_particles,
                self.mapper,
                mode='uccsd',
            )
        self.ansatz = ansatz
        self.circuit = self.ansatz.create(self.num_qubits)

        self.energies: List[float] = []          # 총 에너지(Ha): 전자부+e_nuc
        self.params_list: List[np.ndarray] = []   # 상태별 최적 파라미터
        self.external_refs = external_reference_circuits or []
        self.init_mode = init_mode

    # ----- 내부 유틸 -----
    def _init_theta(self, size: int) -> np.ndarray:
        """초기 파라미터 생성: zeros 모드면 near-zero jitter, 아니면 [0,1) 균등."""
        if self.init_mode == 'zeros':
            return INIT_EPS * self.rng.standard_normal(size)
        return self.rng.random(size)

    def _energy_total(self, params: np.ndarray) -> float:
        """상태벡터로 전자부 기대값을 계산하고 핵-핵 반발 상수(e_nuc)를 더해 총 에너지 산출."""
        circ = self.circuit.assign_parameters(params)
        sv = Statevector(circ)
        e_elec = float(np.real(sv.expectation_value(self.H)))
        return e_elec + self.energy_shift

    def _overlap2_params(self, a: np.ndarray, b: np.ndarray) -> float:
        """|⟨ψ(θ_a)|ψ(θ_b)⟩|² (statevector inner product)."""
        sa = Statevector(self.circuit.assign_parameters(a))
        sb = Statevector(self.circuit.assign_parameters(b))
        return float(abs(complex(sa.inner(sb))) ** 2)

    def _overlap2_ext(self, params: np.ndarray, ref: QuantumCircuit) -> float:
        """|⟨ψ(θ)|φ_ref⟩|² (외부 레퍼런스 회로와의 중첩)."""
        s = Statevector(self.circuit.assign_parameters(params))
        r = Statevector(ref)
        return float(abs(complex(s.inner(r))) ** 2)

    # ----- 목적함수 -----
    def _obj_ground(self, params: np.ndarray) -> float:
        pen_ext = 0.0
        for ref in self.external_refs:
            pen_ext += self._overlap2_ext(params, ref)
        return self._energy_total(params) + self.beta * pen_ext

    def _obj_excited(self, params: np.ndarray) -> float:
        pen = 0.0
        for prev in self.params_list:
            pen += self._overlap2_params(params, prev)
        for ref in self.external_refs:
            pen += self._overlap2_ext(params, ref)
        return self._energy_total(params) + self.beta * pen

    # ----- 최적화 -----
    def _minimize(self, fun, x0: np.ndarray):
        """지정된 메서드로 목적함수 최소화."""
        if OPTIMIZATION_METHOD.upper() == 'COBYLA':
            options = {'maxiter': MAXIMUM_ITERATION, 'rhobeg': COBYLA_RHOBEG, 'disp': False}
        else:
            options = {'maxiter': MAXIMUM_ITERATION, 'disp': False}
        return minimize(fun, x0, method=OPTIMIZATION_METHOD, options=options)

    def _run_single(self, objective, label: str) -> Tuple[float, np.ndarray]:
        """여러 초기값을 시도하여 가장 낮은 에너지/파라미터를 선택."""
        best_e = np.inf
        best_x: Optional[np.ndarray] = None
        for _ in range(ATTEMPT_COUNT):
            x0 = self._init_theta(self.ansatz.num_parameters or 0)
            res = self._minimize(objective, x0)
            if not res.success:
                continue
            e = self._energy_total(res.x)
            if e < best_e:
                best_e, best_x = e, np.array(res.x, dtype=float)
        if best_x is None:
            raise RuntimeError(f"Failed to optimize {label} state.")
        return best_e, best_x

    # ----- 공개 메서드 -----
    def find_ground_state(self) -> bool:
        e, x = self._run_single(self._obj_ground, "ground")
        self.energies.append(e)
        self.params_list.append(x)
        print(f"  - State 0: {e:.8f} Ha")
        return True

    def find_excited_state(self) -> bool:
        idx = len(self.energies)
        e, x = self._run_single(self._obj_excited, f"excited {idx}")
        self.energies.append(e)
        self.params_list.append(x)
        print(f"  - State {idx}: {e:.8f} Ha")
        return True

    def draw_circuit(self, state_idx: int = 0, title: Optional[str] = None) -> Optional[plt.Figure]:
        """Builder의 도식 정책을 따라 상태별 회로 그림을 생성(표시/정리는 외부에서)."""
        if state_idx >= len(self.params_list):
            return None
        circ = self.ansatz.draw_circuit_template.assign_parameters(self.params_list[state_idx])
        if title is None:
            title = f"State {state_idx} (E={self.energies[state_idx]:.6f} Ha)"
        fig = circ.draw(
            output='mpl',
            fold=FOLD,
            style={'dpi': 600, 'fontsize': FONTSIZE, 'subfontsize': FONTSIZE - 1},
        )
        fig.set_size_inches(*FIGSIZE)
        fig.suptitle(title, fontsize=FONTSIZE + 4, fontweight='bold')
        _register_fig(fig)
        return fig

    def run(self, num_states: int = 6, verbose: bool = True) -> List[float]:
        """
        VQD로 지정 개수의 상태를 순차적으로 찾는다.
        - Stage 1처럼 num_states>1인 경우에만 개수 안내를 출력
        - Stage 2처럼 num_states=1인 경우엔 조용히 동작하도록 verbose=False로 호출 권장
        """
        if verbose and num_states > 1:
            print(f"- Compute {num_states} states")
        self.find_ground_state()
        for _ in range(1, num_states):
            self.find_excited_state()
        return self.energies

# =============================
# 결과 리포팅/시각화
# =============================
@dataclass
class EvalResults:
    energies: np.ndarray    # 계산된 총 에너지(Ha)
    exact: np.ndarray       # 정확 총 에너지(Ha)
    errors_µHa: np.ndarray  # (Calc-Exact) µHa
    median_µHa: float
    max_µHa: float
    rmse_µHa: float

def analyze_results(vqd_energies: List[float], exact_energies: np.ndarray, header: str = "Results") -> EvalResults:
    """
    결과를 표로 출력하고 통계 지표를 반환.
    - 오차는 µHa(마이크로하트리) 단위만 출력(요구사항)
    """
    calc = np.array(vqd_energies, dtype=float)
    exact = np.array(exact_energies[: len(calc)], dtype=float)
    err_µ = (calc - exact) * 1e6

    print(f"\n[{header}]")
    print(" State |     Calc (Ha) |    Exact (Ha) |  Error (µHa)")
    print("-------|---------------|---------------|------------")
    for i, (c, ex, eµ) in enumerate(zip(calc, exact, err_µ)):
        sign = "+" if eµ >= 0 else "-"
        print(f" {i:>5d} | {c:13.8f} | {ex:13.8f} | {sign}{abs(eµ):12.3f}")
    med = float(np.median(np.abs(err_µ)))
    mx = float(np.max(np.abs(err_µ)))
    rmse = float(np.sqrt(np.mean(err_µ**2)))
    print("-------|---------------|---------------|------------")
    print(f" median |               |               | {med:12.3f}")
    print(f" max    |               |               | {mx:12.3f}")
    print(f" RMSE   |               |               | {rmse:12.3f}")

    return EvalResults(calc, exact, err_µ, med, mx, rmse)

def create_visualization(results: EvalResults, title: str = 'Energy Level Comparison'):
    """
    (i) 에너지 비교 플롯, (ii) µHa 오차 플롯을 생성해 레지스트리에 등록.
    실제 표시/정리는 show_all_figures_blocking()에서 일괄 처리.
    """
    if results is None:
        print("No results to visualize.")
        return
    energies = results.energies
    exact = results.exact
    states = np.arange(len(energies))

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(states, energies, 'o-', markersize=8, linewidth=2.2, label='Calc')
    ax1.plot(states, exact, '--', linewidth=2.0, alpha=0.85, label='Exact')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Energy (Ha)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    _register_fig(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(states, results.errors_µHa, 'o-', markersize=8, linewidth=2.2, label='Error (µHa)')
    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel('State')
    ax2.set_ylabel('Error (µHa)')
    ax2.set_title('Error Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _register_fig(fig2)

def draw_quantum_circuits(vqd: VQD, tag: str = ""):
    """
    상태별 회로 그림을 생성만 하고, 표시/정리는 show_all_figures_blocking()에서 일괄 처리.
    """
    if not DRAW_CIRCUITS:
        return
    print("\n[Circuits] prepare figures" + (f" {tag}" if tag else ""))
    for i in range(len(vqd.energies)):
        vqd.draw_circuit(i)

# =============================
# 메인 파이프라인
# =============================
def main():
    """
    파이프라인 개요:
    1) H2 전자 문제 구축 → 전자부 해밀토니안 H_elec, 핵-핵 반발 e_nuc
    2) 정확 스펙트럼(총 에너지) 계산: eig(H_elec) + e_nuc
    3) Stage 1: HF+UCCSD 앤자츠로 VQD → 여러 상태 탐색
    4) Stage 2: 각 상태를 Frozen 초기회로 + n_local로 미세 보정(오버랩 패널티 포함)
    5) 결과 표/그래프/회로 도식 출력
    """
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 70)
    print("H₂ VQD: Stage 1 (HF+UCCSD) → Stage 2 per-state refine (Frozen + n_local)")
    print(f"- ε-jitter: {INIT_EPS} | TL reps: {TL_REPS} | Optimizer: {OPTIMIZATION_METHOD}")
    print("=" * 70)

    # 문제 생성 및 상수 쉬프트(핵-핵 반발) 취득
    problem, e_nuc = build_h2_problem(bond_length=H2_BOND_LENGTH)

    # 전자부 해밀토니안 맵핑(총 에너지는 전자부 기대값 + e_nuc로 취급)
    mapper = JordanWignerMapper()
    H_elec = mapper.map(problem.hamiltonian.second_q_op())

    print_hamiltonian_summary(H_elec, verbose=PRINT_H_TERMS)

    # 정확 스펙트럼(총 에너지) 계산
    exact_elec = np.linalg.eigvalsh(H_elec.to_matrix())
    exact_total = exact_elec + e_nuc
    print("\n[Exact Eigenvalues] (first 6, total energy)")
    for i, e in enumerate(exact_total[:6]):
        print(f"  - State {i}: {e:.8f} Ha")

    # ----- Stage 1: HF + UCCSD -----
    print("\n[Stage 1] HF + UCCSD (VQD)")
    ansatz_s1 = AnsatzBuilder(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        mode='uccsd',
    )

    vqd1 = VQD(
        qubit_hamiltonian=H_elec,
        num_qubits=H_elec.num_qubits,
        energy_shift=e_nuc,
        beta=PENALTY_BETA,
        seed=GLOBAL_SEED,
        ansatz=ansatz_s1,
        mapper=mapper,
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        init_mode='random',
    )

    e1 = vqd1.run(num_states=STAGE1_NUM_STATES, verbose=True)
    res1 = analyze_results(e1, exact_total, header="Stage 1 (HF+UCCSD)")
    create_visualization(res1, title='Stage 1: H₂ Spectrum (UCCSD)')
    draw_quantum_circuits(vqd1, tag='[Stage 1]')
    show_all_figures_blocking()

    # ----- Stage 2: 상태별 n_local 미세 보정 -----
    print("\n[Stage 2] Per-state refine: Frozen Initial + n_local (ε-jitter)")

    # Stage 1에서 얻은 상태별(파라미터 포함) 동결 회로 준비
    frozen_stage1 = [
        vqd1.ansatz.eval_circuit.assign_parameters(vqd1.params_list[j])
        for j in range(STAGE1_NUM_STATES)
    ]

    e2: List[float] = []
    ref_pool: List[QuantumCircuit] = []

    for j in range(STAGE1_NUM_STATES):
        print(f"  * refine from State {j}")
        ansatz_s2 = AnsatzBuilder(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            mode='nlocal',
            frozen_initial=frozen_stage1[j],
            tl_reps=TL_REPS,
            tl_entanglement=TL_ENTANGLEMENT,
        )

        vqd2 = VQD(
            qubit_hamiltonian=H_elec,
            num_qubits=H_elec.num_qubits,
            energy_shift=e_nuc,
            beta=PENALTY_BETA,
            seed=GLOBAL_SEED,
            ansatz=ansatz_s2,
            mapper=mapper,
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            external_reference_circuits=ref_pool,
            init_mode='zeros',      # near-zero에서 시작해 미세 보정
        )

        # Stage 2는 상태 하나만 최적화하므로 조용히 실행(verbose=False)
        _ = vqd2.run(num_states=1, verbose=False)
        e2.append(vqd2.energies[0])

        # 상태별 회로 도식 생성(표시는 일괄 처리)
        draw_quantum_circuits(vqd2, tag=f'[Stage 2, from State {j}]')

        # 이번 상태의 보정 결과 회로를 외부 레퍼런스로 누적(다음 상태 직교화 강화)
        refined_circ_j = vqd2.circuit.assign_parameters(vqd2.params_list[0])
        ref_pool.append(refined_circ_j)

    print("\n[Stage 2 Summary]")
    print(" State |   Stage1 (Ha) |   Stage2 (Ha) |   Exact (Ha)")
    print("-------|---------------|---------------|--------------")
    for j in range(STAGE1_NUM_STATES):
        print(f" {j:>5d} | {e1[j]:13.8f} | {e2[j]:13.8f} | {exact_total[j]:12.8f}")

    res2 = analyze_results(e2, exact_total, header="Stage 2 (Frozen + n_local)")
    create_visualization(res2, title='Stage 2: Per-State Refinement (n_local)')
    show_all_figures_blocking()

    print("\nDone.")

if __name__ == "__main__":
    main()
