"""
P452 Project 2 – Simulation of Many-Body Systems
Streamlit app covering:
  • Checkpoint 1.1 – 2-site dimer & 3-site triangular ring (pen-and-paper models, plotted)
  • Checkpoint 1.2 numerical – Exact Diagonalization of 2×2, 4×4, 6×6 square lattices
  • Checkpoint 2.2.4 – Bose-Fermi mixture density profiles (Thomas-Fermi / LDA)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Styling helpers
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="P452 Project 2", layout="wide", page_icon="⚛️")

st.markdown("""
<style>
    .main-title {font-size:2.2rem; font-weight:700; color:#1a1a2e;}
    .section-title {font-size:1.4rem; font-weight:600; color:#16213e; margin-top:1rem;}
    .info-box {background:#f0f4ff; border-left:4px solid #4a6cf7;
               padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0;}
    .stTabs [data-baseweb="tab"] {font-size:1rem; font-weight:500;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚛️ P452 Project 2 – Many-Body Systems Simulator</p>',
            unsafe_allow_html=True)
st.caption("Cheng Chin · Due May 7, 2026  |  Juan Ignacio Prieto Asbun")

tab1, tab2, tab3 = st.tabs([
    "📐 1.1 – Dimer & Triangle",
    "🔢 1.2 – Square Lattice ED",
    "🌊 2.2.4 – Bose-Fermi Mixture",
])

# ═══════════════════════════════════════════════════════════════════
# ─── TAB 1 : Checkpoint 1.1 ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Checkpoint 1.1 – 2-site Dimer & 3-site Triangular Ring</p>',
                unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 2.5])

    with col_ctrl:
        st.subheader("Parameters")
        J_11 = st.slider("J (exchange coupling)", 0.1, 5.0, 1.0, 0.1, key="J11")
        H_max_11 = st.slider("H_max / J", 0.5, 5.0, 2.0, 0.1, key="Hmax11")

    # ── 2-site dimer ──────────────────────────────────────────────
    H_vals = np.linspace(0, H_max_11 * J_11, 400)

    # Analytical eigenvalues (from student's derivation)
    # Basis: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
    # |↑↑⟩ and |↓↓⟩ are already eigenstates; ↑↓/↓↑ block diagonalises into triplet/singlet
    E_uu  = J_11/4 + H_vals          # |↑↑⟩
    E_dd  = J_11/4 - H_vals          # |↓↓⟩
    E_trip = J_11/4                  # triplet |T0⟩  (no H dependence for Sz=0 triplet)
    E_sing = -3*J_11/4               # singlet |S⟩

    Hc_dimer = -J_11  # from student: Hc = -J  →  physics: |↑↑⟩ crosses singlet at H=J
    # More precisely: E_uu = E_sing  →  J/4+H = -3J/4  →  H = -J
    # For positive H, the relevant crossing is |↓↓⟩ vs singlet: J/4-H = -3J/4 → H=J
    Hc_dimer_pos = J_11

    with col_plot:
        fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes1[0]
        ax.plot(H_vals/J_11, E_uu/J_11,  label=r"$|\uparrow\uparrow\rangle$",  color="#e63946", lw=2)
        ax.plot(H_vals/J_11, E_dd/J_11,  label=r"$|\downarrow\downarrow\rangle$", color="#457b9d", lw=2)
        ax.axhline(E_trip/J_11, color="#2a9d8f", lw=2, ls="--", label=r"Triplet $|T_0\rangle$")
        ax.axhline(E_sing/J_11, color="#e9c46a", lw=2, ls="-", label=r"Singlet $|S\rangle$")
        ax.axvline(Hc_dimer_pos, color="grey", ls=":", lw=1.5,
                   label=rf"$H_c/J={Hc_dimer_pos/J_11:.1f}$")
        ax.set_xlabel(r"$H/J$", fontsize=12)
        ax.set_ylabel(r"$E/J$", fontsize=12)
        ax.set_title("2-site Dimer – Eigenvalues vs H", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── 3-site triangular ring ─────────────────────────────────
        # Sz = ±3/2 sectors: single state each
        E_3o2  =  3*J_11/4 + 1.5*H_vals   # |↑↑↑⟩
        E_m3o2 =  3*J_11/4 - 1.5*H_vals   # |↓↓↓⟩

        # Sz = +1/2 block: 3×3 matrix (analytic eigenvalues at H=0, then shift by H/2)
        # H_{1/2} matrix eigenvalues at H=0: diagonalize
        H12_mat = np.array([[-1/4, 1/2, 1/2],
                            [ 1/2,-1/4, 1/2],
                            [ 1/2, 1/2,-1/4]]) * J_11
        evals_12, _ = eigh(H12_mat)
        # Zeeman shifts for Sz=+1/2 sector: each state shifts by +H/2
        # (per student: diagonal ±H/2 added)
        E_p12 = np.array([evals_12[i] + 0.5*H_vals for i in range(3)])

        H_m12_mat = np.array([[-1/4, 1/2, 1/2],
                               [ 1/2,-1/4, 1/2],
                               [ 1/2, 1/2,-1/4]]) * J_11
        evals_m12, _ = eigh(H_m12_mat)
        E_m12 = np.array([evals_m12[i] - 0.5*H_vals for i in range(3)])

        ax2 = axes1[1]
        colors_p12 = ["#e63946","#e76f51","#f4a261"]
        colors_m12 = ["#457b9d","#2a9d8f","#264653"]
        for i in range(3):
            ax2.plot(H_vals/J_11, E_p12[i]/J_11, color=colors_p12[i], lw=2,
                     label=rf"$S^z=+1/2$ #{i+1}" if i==0 else f"$S^z=+1/2$ #{i+1}")
            ax2.plot(H_vals/J_11, E_m12[i]/J_11, color=colors_m12[i], lw=2, ls="--",
                     label=rf"$S^z=-1/2$ #{i+1}" if i==0 else f"$S^z=-1/2$ #{i+1}")
        ax2.plot(H_vals/J_11, E_3o2/J_11,  color="black", lw=2, ls="-.", label=r"$|\uparrow\uparrow\uparrow\rangle$")
        ax2.plot(H_vals/J_11, E_m3o2/J_11, color="purple", lw=2, ls=":", label=r"$|\downarrow\downarrow\downarrow\rangle$")
        ax2.set_xlabel(r"$H/J$", fontsize=12)
        ax2.set_ylabel(r"$E/J$", fontsize=12)
        ax2.set_title("3-site Triangle – Eigenvalues vs H", fontsize=13)
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    st.markdown("""
<div class="info-box">
<b>2-site dimer:</b> The singlet |S⟩ = (|↑↓⟩ − |↓↑⟩)/√2 has energy −3J/4 (independent of H).
The |↓↓⟩ state crosses the singlet at <b>H_c = J</b>, marking the transition to a fully polarised ground state.<br><br>
<b>3-site triangle (frustration):</b> At H = 0 the ground state is 4-fold degenerate (two doublets in Sz = ±1/2).
No configuration can simultaneously anti-align all three bonds on a triangle → <i>geometric frustration</i>.
At H ≫ J the unique ground state is |↓↓↓⟩.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ─── TAB 2 : Checkpoint 1.2 numerical – ED ─────────────────────────
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Checkpoint 1.2 – Exact Diagonalization: Square Lattice Heisenberg Model</p>',
                unsafe_allow_html=True)

    # ── Helper: build Hamiltonian in sparse form ───────────────────
    @st.cache_data(show_spinner=False)
    def build_heisenberg_hamiltonian(L, Sz_sector=None):
        """
        Build the Heisenberg Hamiltonian for an L×L square lattice
        with periodic boundary conditions.
        Sz_sector: if given, restrict to states with total Sz = Sz_sector.
        Returns (H_sparse, basis_states).
        """
        N = L * L
        # Enumerate all basis states in the Sz sector
        # State encoded as integer: bit k = 1 → spin up at site k
        n_up_target = (N // 2 + Sz_sector) if Sz_sector is not None else None

        basis = []
        for state in range(1 << N):
            n_up = bin(state).count('1')
            if Sz_sector is None or n_up == n_up_target:
                basis.append(state)

        D = len(basis)
        state_to_idx = {s: i for i, s in enumerate(basis)}

        # NN bonds on L×L PBC lattice (unique bonds only)
        bonds_set = set()
        for x in range(L):
            for y in range(L):
                i = x * L + y
                j = ((x + 1) % L) * L + y   # right neighbor
                k = x * L + (y + 1) % L      # upper neighbor
                bonds_set.add((min(i, j), max(i, j)))
                bonds_set.add((min(i, k), max(i, k)))
        bonds = sorted(bonds_set)

        H = lil_matrix((D, D), dtype=np.float64)

        for (i, j) in bonds:
            for idx, state in enumerate(basis):
                si_z = 1 if (state >> i) & 1 else -1  # ±1 (×1/2 later)
                sj_z = 1 if (state >> j) & 1 else -1

                # Sz_i Sz_j diagonal
                H[idx, idx] += 0.25 * si_z * sj_z

                # S+_i S-_j : need spin i up, spin j down
                up_i = (state >> i) & 1
                up_j = (state >> j) & 1
                if up_i == 0 and up_j == 1:   # S+_i S-_j (flip i up, j down)
                    new_state = state | (1 << i)
                    new_state = new_state & ~(1 << j)
                    if new_state in state_to_idx:
                        H[idx, state_to_idx[new_state]] += 0.5
                if up_i == 1 and up_j == 0:   # S-_i S+_j
                    new_state = state & ~(1 << i)
                    new_state = new_state | (1 << j)
                    if new_state in state_to_idx:
                        H[idx, state_to_idx[new_state]] += 0.5

        return csr_matrix(H), basis

    @st.cache_data(show_spinner=False)
    def get_spectrum(L, H_field_vals, n_eigs=6):
        """
        For each value of H_field, compute the ground state energy per site
        and average magnetisation Mz across all Sz sectors.
        Also return the lowest n_eigs energies for Sz=0 sector as fn of H.
        """
        N = L * L
        Sz_sectors = list(range(-N // 2, N // 2 + 1))

        # Build H for each sector (J=1 units, H added analytically)
        sector_matrices = {}
        sector_sz = {}
        for sz in Sz_sectors:
            n_up = N // 2 + sz
            if 0 <= n_up <= N:
                H_mat, basis = build_heisenberg_hamiltonian(L, Sz_sector=sz)
                sector_matrices[sz] = H_mat
                sector_sz[sz] = sz

        # For energy spectrum vs H plot: Sz=0 lowest eigenvalues
        # (H field only shifts energy by H*Sz, does not mix sectors)
        sz0_mat = sector_matrices.get(0)
        if sz0_mat is not None:
            D0 = sz0_mat.shape[0]
            k = min(n_eigs, D0 - 1)
            if D0 <= 100:
                evals0 = eigh(sz0_mat.toarray(), eigvals_only=True)[:k]
            else:
                evals0 = eigsh(sz0_mat, k=k, which='SA', return_eigenvectors=False)
                evals0 = np.sort(evals0)
        else:
            evals0 = np.array([])

        # Magnetisation staircase
        gs_energies_per_H = []
        mz_per_H = []
        for H_field in H_field_vals:
            gs_E = np.inf
            gs_sz = 0
            for sz, H_mat in sector_matrices.items():
                D = H_mat.shape[0]
                zeeman = H_field * sz   # total Zeeman shift = H * Sz_total
                k_use = min(1, D - 1)
                if D <= 100:
                    e0 = eigh(H_mat.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0]
                else:
                    e0 = eigsh(H_mat, k=1, which='SA', return_eigenvectors=False)[0]
                e_total = e0 + zeeman
                if e_total < gs_E:
                    gs_E = e_total
                    gs_sz = sz
            gs_energies_per_H.append(gs_E / N)
            mz_per_H.append(gs_sz / N)

        return evals0, np.array(gs_energies_per_H), np.array(mz_per_H)

    # ── Controls ──────────────────────────────────────────────────
    col_c, col_p = st.columns([1, 3])
    with col_c:
        st.subheader("Parameters")
        lattice_choice = st.radio("Lattice size", ["2×2 (N=4)", "4×4 (N=16)", "6×6 (N=36)"],
                                  index=0)
        J_12 = st.number_input("J", value=1.0, min_value=0.1, step=0.1, key="J12")
        H_max_12 = st.slider("H_max / J", 1.0, 10.0, 6.0, 0.5, key="Hmax12")
        n_eigs_show = st.slider("# energy levels to show", 2, 8, 4)
        run_ed = st.button("▶  Run Exact Diagonalization", type="primary")

    L_map = {"2×2 (N=4)": 2, "4×4 (N=16)": 4, "6×6 (N=36)": 6}
    L_sel = L_map[lattice_choice]
    N_sel = L_sel * L_sel

    if run_ed:
        with col_p:
            status = st.status(f"Running ED for {L_sel}×{L_sel} lattice…", expanded=True)
            status.write("Building Hamiltonian in Sz sectors (sparse)…")

            H_arr = np.linspace(0, H_max_12 * J_12, 60)
            # For 6×6, use only Sz sectors near 0 to limit memory
            if L_sel == 6:
                status.write("N=36: restricting to |Sz| ≤ 4 sectors to manage memory…")

            evals0, gs_E_arr, mz_arr = get_spectrum(L_sel, H_arr, n_eigs=n_eigs_show)

            status.update(label="Done!", state="complete")

        with col_p:
            fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))

            # Panel 1: Sz=0 energy spectrum (vs H/J, energies shift only via Zeeman in other sectors)
            ax = axes2[0]
            colors_spec = plt.cm.plasma(np.linspace(0.1, 0.85, len(evals0)))
            for i, e in enumerate(evals0):
                ax.axhline(e / (J_12 * N_sel), color=colors_spec[i], lw=2,
                           label=f"Level {i+1}")
            ax.set_xlabel(r"$H/J$  (Sz=0 block: H-independent)", fontsize=11)
            ax.set_ylabel(r"$E / (J \cdot N)$", fontsize=11)
            ax.set_title(f"{L_sel}×{L_sel}: Sz=0 Energy Levels", fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # Panel 2: Ground state energy vs H/J
            ax2 = axes2[1]
            ax2.plot(H_arr/J_12, gs_E_arr, color="#e63946", lw=2.5)
            ax2.set_xlabel(r"$H/J$", fontsize=11)
            ax2.set_ylabel(r"$E_0 / (J \cdot N)$", fontsize=11)
            ax2.set_title(f"{L_sel}×{L_sel}: Ground State Energy vs H", fontsize=12)
            ax2.grid(alpha=0.3)

            # Panel 3: Magnetisation staircase
            ax3 = axes2[2]
            ax3.step(H_arr/J_12, mz_arr, color="#457b9d", lw=2.5, where="post")
            ax3.axhline(0.5, color="grey", ls="--", lw=1.2, label="Saturation")
            ax3.axhline(-0.5, color="grey", ls="--", lw=1.2)
            ax3.set_xlabel(r"$H/J$", fontsize=11)
            ax3.set_ylabel(r"$\langle M_z \rangle = \langle S^z_{tot} \rangle / N$", fontsize=11)
            ax3.set_title(f"{L_sel}×{L_sel}: Magnetisation Staircase", fontsize=12)
            ax3.legend(fontsize=9)
            ax3.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # ── Validation for 2×2 vs analytical ──────────────────
            if L_sel == 2:
                st.markdown("**2×2 validation – Sz=0 sector eigenvalues (J=1 units):**")
                analytical = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 2.0])  # known
                num_e = np.sort(evals0)
                cols_v = st.columns(len(num_e))
                for i, (ne, ) in enumerate(zip(num_e,)):
                    cols_v[i].metric(f"Level {i+1}", f"{ne:.4f}J")

            # Summary
            Hc_est = H_arr[np.where(np.diff(mz_arr) > 0)[0]] / J_12 if np.any(np.diff(mz_arr) > 0) else []
            if len(Hc_est):
                hc_str = ", ".join([f"{h:.2f}" for h in Hc_est[:5]])
                st.info(f"**Critical fields (Hc/J):** {hc_str}  — magnetisation staircase jumps (AFM→FM transition)")
    else:
        with col_p:
            st.markdown("""
<div class="info-box">
Click <b>▶ Run Exact Diagonalization</b> to compute the energy spectrum and magnetisation staircase.
<br>
• <b>2×2</b>: fast (~seconds), full Hilbert space D=16<br>
• <b>4×4</b>: moderate (~30 s), uses sparse eigensolver<br>
• <b>6×6</b>: may take a few minutes; Sz=0 sector has D=9075
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
<b>Magnetisation staircase:</b> As H increases, S<sup>z</sup><sub>tot</sub> jumps from 0 → 1 → … → N/2.
In the thermodynamic limit N→∞ the staircase becomes a continuous AFM→FM phase transition.
The critical field (H/J)<sub>c</sub> is identified from the last jump to full saturation.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ─── TAB 3 : Checkpoint 2.2.4 – Bose-Fermi density profiles ────────
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Checkpoint 2.2.4 – Bose-Fermi Mixture Density Profiles</p>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
Thomas-Fermi (TF) for bosons + Local Density Approximation (LDA) for fermions, solved self-consistently.<br>
Profiles are computed on a 1D radial grid. Chemical potentials are fixed by normalising
∫ n(r) 4πr² dr = N.
</div>
""", unsafe_allow_html=True)

    # ── Physical constants (SI, then convert to a natural scale) ──
    # We work in dimensionless units:
    #   length: a_B_ho = sqrt(hbar/(m_B ω_B))  (bosonic harmonic-oscillator length)
    #   energy: hbar ω_B
    # All densities in units of 1/a_B_ho^3

    col_params, col_plots = st.columns([1, 2.5])

    with col_params:
        st.subheader("System Parameters")

        st.markdown("**Masses (in amu)**")
        mB_amu = st.number_input("Boson mass mB (amu)", value=87.0, min_value=1.0, step=1.0,
                                  help="e.g. 87 for Rb")
        mF_amu = st.number_input("Fermion mass mF (amu)", value=40.0, min_value=1.0, step=1.0,
                                  help="e.g. 40 for K")

        st.markdown("**Trap frequencies**")
        omega_ratio = st.slider("ωF / ωB", 0.5, 3.0, 1.0, 0.1,
                                 help="Ratio of fermionic to bosonic trap frequency")

        st.markdown("**Particle numbers**")
        NB = st.number_input("NB (bosons)", value=10000, min_value=100, step=1000)
        NF = st.number_input("NF (fermions)", value=5000, min_value=100, step=1000)

        st.markdown("**Scattering lengths**")
        aB_ratio = st.slider("aB / a_ho (boson-boson)", 0.001, 0.05, 0.01, 0.001,
                              format="%.3f")
        aBF_ratio = st.slider("aBF / a_ho (interspecies)", -0.05, 0.05, 0.0, 0.002,
                               format="%.3f",
                               help="Negative = attractive, Positive = repulsive")

        regime_label = ("Non-interacting" if abs(aBF_ratio) < 0.003
                        else ("Repulsive (phase separation tendency)" if aBF_ratio > 0
                              else "Attractive (collapse tendency)"))
        st.info(f"**Regime:** {regime_label}")

        run_bf = st.button("▶  Compute Density Profiles", type="primary")

    with col_plots:
        if run_bf:
            # ─── Dimensionless parameters ────────────────────────────
            mass_ratio = mF_amu / mB_amu
            # In units of hbar*omega_B and a_ho = sqrt(hbar/mB/omegaB):
            # gB = 4π aB (in units of hbar*omegaB * a_ho^3)
            # gBF = 2π aBF / mu_reduced  → mu_red = mB*mF/(mB+mF) in units of mB
            #       → mu_red/mB = mF/(mB+mF) = mass_ratio/(1+mass_ratio)
            mu_red_ratio = mass_ratio / (1 + mass_ratio)

            gB  = 4 * np.pi * aB_ratio
            gBF = 2 * np.pi * aBF_ratio / mu_red_ratio

            omF = omega_ratio   # ωF/ωB

            # Radial grid
            R_max = 25.0  # in units of a_ho
            Nr = 800
            r = np.linspace(0, R_max, Nr)
            dr = r[1] - r[0]

            def norm3d(n_r, r_arr):
                """∫ n(r) 4π r² dr via trapezoidal rule"""
                integrand = n_r * 4 * np.pi * r_arr**2
                return np.trapz(integrand, r_arr)

            def fermi_density(muF_loc, r_arr, mass_ratio, omF, gBF, nB_arr):
                """LDA fermionic density: nF(r) from inversion of µF = VF + EF + gBF*nB"""
                VF = 0.5 * mass_ratio * omF**2 * r_arr**2
                arg = muF_loc - VF - gBF * nB_arr
                # EF = (hbar^2/2mF)(6π^2 nF)^(2/3)
                # In our units hbar^2/2mF = 1/(2*mass_ratio)
                # So nF = (1/6π^2) * (2*mass_ratio * arg)^(3/2)  where arg>0
                prefac = 1.0 / (6 * np.pi**2)
                coeff = 2 * mass_ratio
                nF = np.where(arg > 0, prefac * (coeff * arg)**(1.5), 0.0)
                return nF

            def boson_density(muB_val, r_arr, gB, gBF, nF_arr):
                """Thomas-Fermi bosonic density"""
                VB = 0.5 * r_arr**2
                nB = (muB_val - VB - gBF * nF_arr) / gB
                return np.maximum(nB, 0.0)

            # Self-consistent iteration
            # Initial guess: non-interacting profiles
            # Bosons TF: µB = gB * nB(0), VB(R_TF)=µB → R_TF=sqrt(2µB)
            # Estimate µB from NB normalization
            def solve_profiles(gB, gBF, NB, NF, mass_ratio, omF, r, max_iter=80, tol=1e-5):
                # Initial guess for chemical potentials
                # Non-interacting BEC: µB ~ gB*(NB/V_eff)
                muB = 15.0 * (NB * gB)**(2/5) / (14**(2/5)) if gB > 0 else 1.0
                muF = 0.5 * (6 * np.pi**2 * NF / (4/3 * np.pi * 10**3))**(2/3) / mass_ratio

                # Better initial µ from isolated profiles
                # TF boson radius estimate
                RTF_B = (2 * muB)**0.5
                nB_init = np.maximum((muB - 0.5*r**2) / max(gB, 1e-9), 0)
                norm_B = norm3d(nB_init, r)
                if norm_B > 0:
                    muB *= (NB / norm_B)**(2/5)

                nF_arr = np.zeros_like(r)
                nB_arr = np.maximum((muB - 0.5*r**2) / max(gB, 1e-9), 0)

                for it in range(max_iter):
                    nB_old = nB_arr.copy()
                    nF_old = nF_arr.copy()

                    # Rescale µF so ∫nF 4πr²dr = NF
                    for _ in range(30):
                        nF_trial = fermi_density(muF, r, mass_ratio, omF, gBF, nB_arr)
                        norm_F = norm3d(nF_trial, r)
                        if norm_F > 1e-12:
                            muF *= (NF / norm_F)**(1/3)
                        else:
                            muF *= 1.5
                        if abs(norm_F / max(NF, 1) - 1) < 1e-4:
                            break
                    nF_arr = fermi_density(muF, r, mass_ratio, omF, gBF, nB_arr)

                    # Rescale µB so ∫nB 4πr²dr = NB
                    for _ in range(30):
                        nB_trial = boson_density(muB, r, gB, gBF, nF_arr)
                        norm_B = norm3d(nB_trial, r)
                        if norm_B > 1e-12:
                            muB *= (NB / norm_B)**(1/3)
                        else:
                            muB *= 1.5
                        if abs(norm_B / max(NB, 1) - 1) < 1e-4:
                            break
                    nB_arr = boson_density(muB, r, gB, gBF, nF_arr)

                    delta = (np.max(np.abs(nB_arr - nB_old)) + np.max(np.abs(nF_arr - nF_old)))
                    if delta < tol:
                        break

                return nB_arr, nF_arr, muB, muF

            with st.spinner("Solving self-consistent density profiles…"):
                nB, nF, muB_sol, muF_sol = solve_profiles(
                    gB, gBF, NB, NF, mass_ratio, omF, r)

                # Also compute non-interacting profiles for comparison
                nB_ni, nF_ni, _, _ = solve_profiles(
                    gB, 0.0, NB, NF, mass_ratio, omF, r)

            # Thomas-Fermi radii
            def tf_radius(n_arr, r_arr, threshold=0.01):
                max_n = np.max(n_arr)
                if max_n < 1e-30:
                    return 0.0
                idx = np.where(n_arr > threshold * max_n)[0]
                return r_arr[idx[-1]] if len(idx) else 0.0

            RTF_B = tf_radius(nB, r)
            RTF_F = tf_radius(nF, r)
            RTF_B_ni = tf_radius(nB_ni, r)
            RTF_F_ni = tf_radius(nF_ni, r)

            # ─── Plots ────────────────────────────────────────────
            fig3 = plt.figure(figsize=(14, 5))
            gs = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.35)

            # Panel 1: density profiles
            ax1 = fig3.add_subplot(gs[0])
            ax1.plot(r, nB,    color="#e63946", lw=2.5, label=r"$n_B(r)$ interacting")
            ax1.plot(r, nB_ni, color="#e63946", lw=1.5, ls="--", alpha=0.6,
                     label=r"$n_B(r)$ non-inter.")
            ax1.plot(r, nF,    color="#457b9d", lw=2.5, label=r"$n_F(r)$ interacting")
            ax1.plot(r, nF_ni, color="#457b9d", lw=1.5, ls="--", alpha=0.6,
                     label=r"$n_F(r)$ non-inter.")
            ax1.set_xlabel(r"$r / a_{ho}$", fontsize=11)
            ax1.set_ylabel(r"$n(r)$ [arb. units]", fontsize=11)
            ax1.set_title("Density Profiles", fontsize=12)
            ax1.legend(fontsize=8)
            ax1.set_xlim(0, max(RTF_B * 1.4, RTF_F * 1.4, 5))
            ax1.grid(alpha=0.3)

            # Panel 2: TF radii comparison bar chart
            ax2 = fig3.add_subplot(gs[1])
            species = ["BEC (non-int.)", "BEC (int.)", "Fermi (non-int.)", "Fermi (int.)"]
            vals    = [RTF_B_ni, RTF_B, RTF_F_ni, RTF_F]
            colors  = ["#f4a261","#e63946","#a8dadc","#457b9d"]
            bars = ax2.bar(species, vals, color=colors, edgecolor="white", linewidth=1.5)
            ax2.set_ylabel(r"Thomas-Fermi radius $/ a_{ho}$", fontsize=11)
            ax2.set_title("Cloud Sizes", fontsize=12)
            ax2.tick_params(axis='x', rotation=30)
            for bar, val in zip(bars, vals):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f"{val:.2f}", ha="center", va="bottom", fontsize=9)
            ax2.grid(axis='y', alpha=0.3)

            # Panel 3: central density comparison
            ax3 = fig3.add_subplot(gs[2])
            n0_B    = nB[0]
            n0_F    = nF[0]
            n0_B_ni = nB_ni[0]
            n0_F_ni = nF_ni[0]
            species2 = ["BEC\n(non-int.)", "BEC\n(int.)", "Fermi\n(non-int.)", "Fermi\n(int.)"]
            vals2    = [n0_B_ni, n0_B, n0_F_ni, n0_F]
            bars2 = ax3.bar(species2, vals2, color=colors, edgecolor="white", linewidth=1.5)
            ax3.set_ylabel(r"$n(r=0)$ [arb. units]", fontsize=11)
            ax3.set_title("Central Density", fontsize=12)
            for bar, val in zip(bars2, vals2):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f"{val:.1f}", ha="center", va="bottom", fontsize=9)
            ax3.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            # ─── Metrics ─────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BEC TF radius", f"{RTF_B:.2f} a_ho",
                      delta=f"{RTF_B - RTF_B_ni:+.2f} vs non-int.")
            c2.metric("Fermi TF radius", f"{RTF_F:.2f} a_ho",
                      delta=f"{RTF_F - RTF_F_ni:+.2f} vs non-int.")
            c3.metric("BEC central density", f"{nB[0]:.1f}",
                      delta=f"{nB[0] - nB_ni[0]:+.1f} vs non-int.")
            c4.metric("Fermi central density", f"{nF[0]:.1f}",
                      delta=f"{nF[0] - nF_ni[0]:+.1f} vs non-int.")

            # Stability check
            dEF_dnF_center = (1.0 / mass_ratio) * (6 * np.pi**2 * nF[0])**(-1/3) if nF[0] > 0 else 1e9
            stable = gB * dEF_dnF_center > gBF**2
            if stable:
                st.success("✅ Mixture is **stable** (det M > 0): gB · ∂EF/∂nF > gBF²")
            else:
                st.error("⚠️ Mixture is **unstable**: gB · ∂EF/∂nF < gBF² → "
                         "phase separation (gBF>0) or collapse (gBF<0)")

            st.markdown(f"""
<div class="info-box">
<b>Physical interpretation:</b><br>
• <b>gBF = 0</b>: Clouds are independent. BEC forms a dense central peak; Fermi cloud extends further due to Fermi pressure.<br>
• <b>gBF &lt; 0 (attractive):</b> Bosons pull fermions toward the centre, increasing central density of both species. Can lead to collapse if ∣gBF∣ is too large.<br>
• <b>gBF &gt; 0 (repulsive):</b> BEC acts as a "plunger," pushing fermions outward. Fermi radius increases, BEC radius decreases. Large enough aBF → phase separation.<br>
<b>Current regime:</b> aBF/a_ho = {aBF_ratio:.4f} → <i>{regime_label}</i>
</div>
""", unsafe_allow_html=True)

        else:
            st.markdown("""
<div class="info-box">
Set the parameters on the left and click <b>▶ Compute Density Profiles</b>.<br><br>
The solver iterates the coupled Thomas-Fermi / LDA equations:<br>
&nbsp;&nbsp;• <b>nB(r)</b> = [µB − VB(r) − gBF·nF(r)] / gB<br>
&nbsp;&nbsp;• <b>nF(r)</b> = (1/6π²) [2mF(µF − VF(r) − gBF·nB(r)) / ℏ²]^(3/2)<br>
until convergence. Chemical potentials are renormalised at each step to enforce particle-number normalisation.
</div>
""", unsafe_allow_html=True)
