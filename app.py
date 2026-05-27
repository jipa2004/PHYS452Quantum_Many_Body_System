"""
P452 Project 2 – Simulation of Many-Body Systems
Streamlit app: Checkpoints 1.1, 1.2 (numerical ED), 2.2.4 (Bose-Fermi mixtures)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Force matplotlib into a light style regardless of OS theme
# ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#1a1a2e",
    "xtick.color":       "#1a1a2e",
    "ytick.color":       "#1a1a2e",
    "text.color":        "#1a1a2e",
    "legend.facecolor":  "#ffffff",
    "legend.edgecolor":  "#cccccc",
    "legend.labelcolor": "#1a1a2e",
    "grid.color":        "#dddddd",
    "lines.color":       "#1a1a2e",
    "patch.edgecolor":   "#cccccc",
})

# ─────────────────────────────────────────────────────────────────
# Page config & global style
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="P452 Project 2", layout="wide", page_icon="⚛️")

st.markdown("""
<style>
  /* ── Lock the entire app to a white/light theme ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stApp"], [data-testid="block-container"],
  .main, .block-container {
    background-color: #f7f8fc !important;
    color: #1a1a2e !important;
  }

  /* Sidebar (if any) */
  [data-testid="stSidebar"] { background-color: #eef0f8 !important; }

  /* All text nodes */
  *, *::before, *::after { color: #1a1a2e !important; }

  /* Override Streamlit's own dark-mode overrides */
  .stMarkdown, .stMarkdown p, .stMarkdown li,
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
  p, li, span, label, div { color: #1a1a2e !important; }

  /* Inputs, selects, sliders */
    [data-baseweb="input"] input,
    [data-baseweb="select"] div {
    color: #1a1a2e !important;
    background-color: #eef0f8 !important;
    }

    [data-baseweb="input"] {
    background-color: #eef0f8 !important;
    border-radius: 6px !important;
    }
  /* Metric widgets */
  [data-testid="metric-container"] * { color: #1a1a2e !important; }

  /* Tab strip */
  .stTabs [data-baseweb="tab-list"] { background: #eef0f8 !important; }
  .stTabs [data-baseweb="tab"] {
    font-size: 1rem; font-weight: 500;
    color: #1a1a2e !important;
    background: transparent !important;
  }
  .stTabs [aria-selected="true"] {
    background: #ffffff !important;
    border-bottom: 3px solid #4a6cf7 !important;
  }

  /* Buttons */
  .stButton button {
    background-color: #4a6cf7 !important;
    color: #ffffff !important;
    border: none; border-radius: 6px;
  }
  .stButton button:hover { background-color: #3a5ce7 !important; }

  .block-container { padding-top: 1.2rem; }

  .main-title {
    font-size: 2rem; font-weight: 700;
    color: #1a1a2e !important; margin-bottom: 0.2rem;
  }
  .section-hdr {
    font-size: 1.25rem; font-weight: 600;
    color: #1a1a2e !important; border-bottom: 2px solid #4a6cf7;
    padding-bottom: 4px; margin: 0.8rem 0 0.5rem;
  }
  .info-box {
    background: #dde6ff !important;
    border-left: 4px solid #4a6cf7;
    padding: 0.75rem 1rem; border-radius: 6px;
    margin: 0.4rem 0 0.8rem;
    color: #1a1a2e !important; font-size: 0.92rem; line-height: 1.55;
  }
  .warn-box {
    background: #fff3cd !important;
    border-left: 4px solid #e6a817;
    padding: 0.75rem 1rem; border-radius: 6px;
    margin: 0.4rem 0 0.8rem; color: #1a1a2e !important; font-size: 0.92rem;
  }
  .ok-box {
    background: #d4f5e9 !important;
    border-left: 4px solid #10b981;
    padding: 0.75rem 1rem; border-radius: 6px;
    margin: 0.4rem 0; color: #1a1a2e !important; font-size: 0.92rem;
  }
  .err-box {
    background: #ffe0e0 !important;
    border-left: 4px solid #ef4444;
    padding: 0.75rem 1rem; border-radius: 6px;
    margin: 0.4rem 0; color: #1a1a2e !important; font-size: 0.92rem;
  }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚛️ P452 Project 2 – Many-Body Systems Simulator</p>',
            unsafe_allow_html=True)
st.caption("Cheng Chin · P452  |  Juan Ignacio Prieto Asbun · May 2026")

tab1, tab2, tab3 = st.tabs([
    "📐 1.1 – Dimer & Triangle",
    "🔢 1.2 – Square Lattice ED",
    "🌊 2.2.4 – Bose-Fermi Mixture",
])

# ══════════════════════════════════════════════════════════════════
# Shared: vectorized Hamiltonian builder
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_sector_vectorized(L, sz_sector):
    """
    Build the Sz-sector Hamiltonian for L×L PBC Heisenberg model (J=1).
    Returns (csr_matrix, dimension).  Returns (None, 0) if sector is empty.
    Uses vectorised numpy operations — ~10× faster than a Python loop.
    """
    N = L * L
    n_up = N // 2 + sz_sector
    if n_up < 0 or n_up > N:
        return None, 0

    # Enumerate basis states from combinations (faster than scanning 2^N)
    basis = np.array(
        [sum(1 << p for p in pos) for pos in combinations(range(N), n_up)],
        dtype=np.int64,
    )
    D = len(basis)
    state_to_idx = {int(s): i for i, s in enumerate(basis)}

    # Unique NN bonds with PBC
    bonds_set = set()
    for x in range(L):
        for y in range(L):
            i = x * L + y
            j = ((x + 1) % L) * L + y
            k = x * L + (y + 1) % L
            bonds_set.add((min(i, j), max(i, j)))
            bonds_set.add((min(i, k), max(i, k)))
    bonds = sorted(bonds_set)

    row_list, col_list, val_list = [], [], []
    idx_all = np.arange(D, dtype=np.int32)

    for (i, j) in bonds:
        mi = np.int64(1 << i)
        mj = np.int64(1 << j)

        # Diagonal  Sz_i Sz_j / 4
        si = np.where(basis & mi, 1.0, -1.0)
        sj = np.where(basis & mj, 1.0, -1.0)
        row_list.append(idx_all)
        col_list.append(idx_all)
        val_list.append(0.25 * si * sj)

        # S+_i S-_j  (need spin-i=↓, spin-j=↑)
        flip1 = np.where((~(basis & mi).astype(bool)) & (basis & mj).astype(bool))[0]
        for idx in flip1:
            ns = (int(basis[idx]) | int(mi)) & ~int(mj)
            if ns in state_to_idx:
                row_list.append([idx])
                col_list.append([state_to_idx[ns]])
                val_list.append([0.5])

        # S-_i S+_j  (need spin-i=↑, spin-j=↓)
        flip2 = np.where((basis & mi).astype(bool) & (~(basis & mj).astype(bool)))[0]
        for idx in flip2:
            ns = (int(basis[idx]) & ~int(mi)) | int(mj)
            if ns in state_to_idx:
                row_list.append([idx])
                col_list.append([state_to_idx[ns]])
                val_list.append([0.5])

    rows = np.concatenate([np.atleast_1d(r) for r in row_list])
    cols = np.concatenate([np.atleast_1d(c) for c in col_list])
    vals = np.concatenate([np.atleast_1d(v) for v in val_list])

    H = csr_matrix((vals, (rows, cols)), shape=(D, D))
    return H, D


def lowest_eigenvalue(H_csr, D):
    """Return the lowest eigenvalue of a sparse or dense matrix."""
    if D == 1:
        return float(H_csr[0, 0])
    if D <= 400:
        return float(eigh(H_csr.toarray(), eigvals_only=True)[0])
    return float(eigsh(H_csr, k=1, which="SA", return_eigenvectors=False)[0])


def lowest_k_eigenvalues(H_csr, D, k):
    k = min(k, D - 1)
    if D <= 400:
        return eigh(H_csr.toarray(), eigvals_only=True)[:k]
    return np.sort(eigsh(H_csr, k=k, which="SA", return_eigenvectors=False))


# ═══════════════════════════════════════════════════════════════════
# TAB 1 – Checkpoint 1.1
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-hdr">Checkpoint 1.1 – 2-site Dimer &amp; 3-site Triangular Ring</p>',
                unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 2.8])

    with col_ctrl:
        st.subheader("Parameters")
        J_11 = st.slider("J (exchange coupling)", 0.1, 5.0, 1.0, 0.1, key="J11")
        H_max_11 = st.slider("H_max / J", 0.5, 5.0, 2.5, 0.1, key="Hmax11")

    H_arr = np.linspace(0, H_max_11 * J_11, 500)

    # ── 2-site dimer (analytic) ───────────────────────────────────
    E_uu   = J_11 / 4 + H_arr           # |↑↑⟩,  Sz=+1
    E_dd   = J_11 / 4 - H_arr           # |↓↓⟩,  Sz=-1
    E_trip =  J_11 / 4 * np.ones_like(H_arr)   # triplet |T0⟩, Sz=0
    E_sing = -3 * J_11 / 4 * np.ones_like(H_arr)  # singlet, Sz=0
    Hc     = J_11                        # |↓↓⟩ crosses singlet at H=J

    # ── 3-site triangle (analytic block diagonalisation) ─────────
    # Sz = ±3/2 : single state
    E_uuu  =  3*J_11/4 + 1.5*H_arr
    E_ddd  =  3*J_11/4 - 1.5*H_arr

    # Sz = +1/2 block (3×3), H=0 matrix; Zeeman shifts each level by +H/2
    H12 = J_11 * np.array([[-1/4, 1/2, 1/2],
                             [ 1/2,-1/4, 1/2],
                             [ 1/2, 1/2,-1/4]])
    ev12, _ = eigh(H12)
    E_p12 = np.outer(ev12, np.ones_like(H_arr)) + 0.5 * H_arr   # (3, n_H)

    ev_m12, _ = eigh(J_11 * np.array([[-1/4, 1/2, 1/2],
                                       [ 1/2,-1/4, 1/2],
                                       [ 1/2, 1/2,-1/4]]))
    E_m12 = np.outer(ev_m12, np.ones_like(H_arr)) - 0.5 * H_arr

    with col_plot:
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.8))
        fig1.patch.set_facecolor("white")

        # ── Panel A: dimer ────────────────────────────────────────
        ax = axes1[0]
        ax.set_facecolor("white")
        ax.plot(H_arr/J_11, E_uu/J_11,   color="#c0392b", lw=2,   label="|uu>")
        ax.plot(H_arr/J_11, E_dd/J_11,   color="#2980b9", lw=2,   label="|dd>")
        ax.plot(H_arr/J_11, E_trip/J_11, color="#27ae60", lw=2, ls="--", label=r"Triplet $|T_0\rangle$")
        ax.plot(H_arr/J_11, E_sing/J_11, color="#e67e22", lw=2.5, label=r"Singlet $|S\rangle$")
        ax.axvline(Hc/J_11, color="#555555", ls=":", lw=1.5,
                   label=rf"$H_c/J = {Hc/J_11:.1f}$")
        ax.set_xlabel(r"$H/J$", fontsize=12)
        ax.set_ylabel(r"$E/J$", fontsize=12, color="#1a1a2e")
        ax.set_title("2-site Dimer – Eigenvalues vs $H$", fontsize=13)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
                  labelcolor="#1a1a2e")
        ax.grid(alpha=0.3, color="#aaaaaa")

        # ── Panel B: triangle ─────────────────────────────────────
        ax2 = axes1[1]
        ax2.set_facecolor("white")
        reds   = ["#c0392b", "#e74c3c", "#ff7675"]
        blues  = ["#2980b9", "#3498db", "#74b9ff"]
        for i in range(3):
            ax2.plot(H_arr/J_11, E_p12[i]/J_11,  color=reds[i],  lw=2,
                     label=f"Sz=+1/2  #{i+1}")
            ax2.plot(H_arr/J_11, E_m12[i]/J_11,  color=blues[i], lw=2, ls="--",
                     label=f"Sz=-1/2  #{i+1}")
        ax2.plot(H_arr/J_11, E_uuu/J_11, color="#2d3436", lw=2, ls="-.",
                 label="|uuu>")
        ax2.plot(H_arr/J_11, E_ddd/J_11, color="#6c5ce7", lw=2, ls=":",
                 label="|ddd>")
        ax2.set_xlabel(r"$H/J$", fontsize=12)
        ax2.set_ylabel(r"$E/J$", fontsize=12, color="#1a1a2e")
        ax2.set_title("3-site Triangle – Eigenvalues vs $H$", fontsize=13)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#cccccc")
        ax2.legend(fontsize=7.5, ncol=2, facecolor="white",
                   edgecolor="#cccccc", labelcolor="#1a1a2e")
        ax2.grid(alpha=0.3, color="#aaaaaa")

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    st.markdown(f"""
<div class="info-box">
<b>2-site dimer:</b> The singlet |S⟩ = (|↑↓⟩ − |↓↑⟩)/√2 has energy −3J/4,
independent of <i>H</i>. The |↓↓⟩ state crosses the singlet at <b>H<sub>c</sub>/J =
{Hc/J_11:.2f}</b>, marking the onset of the fully polarised phase.<br><br>
<b>3-site triangle (frustration):</b> At H = 0, the ground state is four-fold degenerate
(two doublets inside the S<sup>z</sup> = ±½ sectors). No spin configuration can
simultaneously anti-align all three bonds → <em>geometric frustration</em>. At H ≫ J
the unique ground state is |↓↓↓⟩.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 – Checkpoint 1.2 numerical ED
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-hdr">Checkpoint 1.2 – Exact Diagonalization: Square Lattice Heisenberg Model</p>',
                unsafe_allow_html=True)

    col_c, col_p = st.columns([1, 3])

    with col_c:
        st.subheader("Parameters")
        lattice_choice = st.radio(
            "Lattice", ["2×2  (N=4)", "4×4  (N=16)", "6×6  (N=36)"], index=0
        )
        J_12    = st.number_input("J", value=1.0, min_value=0.1, step=0.1, key="J12")
        H_max_12 = st.slider("H_max / J", 1.0, 12.0, 6.0, 0.5, key="Hmax12")
        n_eigs_show = st.slider("Energy levels to show (Sz=0)", 2, 8, 4)
        run_ed = st.button("▶  Run Exact Diagonalization", type="primary")

    L_map = {"2×2  (N=4)": 2, "4×4  (N=16)": 4, "6×6  (N=36)": 6}
    L_sel = L_map[lattice_choice]
    N_sel = L_sel * L_sel

    # Decide which Sz sectors are feasible
    from math import comb as math_comb
    MAX_DIM = 80_000   # sector dimensions above this are skipped for 6×6

    def feasible_sectors(L):
        N = L * L
        sectors = []
        for sz in range(-N//2, N//2 + 1):
            n_up = N//2 + sz
            if 0 <= n_up <= N:
                D = math_comb(N, n_up)
                if D <= MAX_DIM:
                    sectors.append(sz)
        return sectors

    if run_ed:
        with col_p:
            status = st.status(f"Running ED for {L_sel}×{L_sel} lattice…", expanded=True)

            sectors = feasible_sectors(L_sel)
            skipped = [sz for sz in range(-N_sel//2, N_sel//2+1)
                       if sz not in sectors and 0 <= N_sel//2+sz <= N_sel]

            status.write(f"Feasible sectors: {len(sectors)}  |  "
                         f"Skipped (D > {MAX_DIM:,}): {len(skipped)}")

            # ── Build each sector once, compute ground state energy ──
            sector_e0 = {}  # sz → E0 (J-only, no H)
            sz0_evals = None

            for sz in sectors:
                D_sz = math_comb(N_sel, N_sel//2 + sz)
                status.write(f"  Sz={sz:+d}: D={D_sz:,}")
                H_mat, D = build_sector_vectorized(L_sel, sz)
                if H_mat is None:
                    continue
                e0 = lowest_eigenvalue(H_mat, D)
                sector_e0[sz] = e0
                # Save Sz=0 low eigenvalues for spectrum plot
                if sz == 0 and D > 0:
                    k = min(n_eigs_show, D - 1)
                    sz0_evals = lowest_k_eigenvalues(H_mat, D, k)

            # ── H-sweep via E(Sz,H) = E0(Sz) + J*H*Sz ──────────────
            # (No matrix rebuilds! O(n_sectors × n_H) arithmetic)
            H_arr_12 = np.linspace(0, H_max_12 * J_12, 300)
            gs_E_arr = np.full(len(H_arr_12), np.inf)
            mz_arr   = np.zeros(len(H_arr_12))

            for h_idx, H_field in enumerate(H_arr_12):
                best_E = np.inf
                best_sz = 0
                for sz, e0 in sector_e0.items():
                    E_tot = J_12 * e0 + H_field * sz   # J scales E0; Zeeman adds H*Sz
                    if E_tot < best_E:
                        best_E  = E_tot
                        best_sz = sz
                gs_E_arr[h_idx] = best_E / N_sel
                mz_arr[h_idx]   = best_sz / N_sel

            status.update(label="✅ Done!", state="complete")

        # ── Plots ─────────────────────────────────────────────────
        with col_p:
            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.8))
            fig2.patch.set_facecolor("white")

            plot_colors = plt.cm.plasma(np.linspace(0.1, 0.85, n_eigs_show))

            # Panel 1: Sz=0 energy levels (H-independent for Sz=0 block)
            ax = axes2[0]
            ax.set_facecolor("white")
            if sz0_evals is not None:
                for i, e in enumerate(sz0_evals):
                    label = f"Level {i+1}:  {J_12*e/N_sel:.3f} J/site"
                    ax.axhline(J_12 * e / N_sel, color=plot_colors[i], lw=2.2,
                               label=label)
                ax.set_xlim(0, 1)
                ax.set_xlabel("(H-independent within Sz=0 block)", fontsize=10,
                              color="#555555")
            else:
                ax.text(0.5, 0.5, "Sz=0 sector\nnot computed\n(D > limit)",
                        ha="center", va="center", fontsize=11, color="#555555",
                        transform=ax.transAxes)
            ax.set_ylabel(r"$E\,/\,(J \cdot N)$", fontsize=11, color="#1a1a2e")
            ax.set_title(f"{L_sel}×{L_sel}  Sz=0 Energy Levels", fontsize=12,
                         color="#1a1a2e")
            for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")
            ax.legend(fontsize=8, facecolor="white", edgecolor="#cccccc",
                      labelcolor="#1a1a2e")
            ax.grid(alpha=0.3, color="#aaaaaa")

            # Panel 2: Ground state energy vs H
            ax2p = axes2[1]
            ax2p.set_facecolor("white")
            ax2p.plot(H_arr_12 / J_12, gs_E_arr, color="#c0392b", lw=2.5)
            ax2p.set_xlabel(r"$H/J$", fontsize=11)
            ax2p.set_ylabel(r"$E_0\,/\,(J \cdot N)$", fontsize=11, color="#1a1a2e")
            ax2p.set_title(f"{L_sel}×{L_sel}  Ground State Energy vs $H$",
                           fontsize=12)
            for sp in ax2p.spines.values(): sp.set_edgecolor("#cccccc")
            ax2p.grid(alpha=0.3, color="#aaaaaa")

            # Panel 3: Magnetisation staircase
            ax3p = axes2[2]
            ax3p.set_facecolor("white")
            ax3p.step(H_arr_12 / J_12, mz_arr, color="#2980b9", lw=2.5, where="post",
                      label=r"$\langle M_z \rangle$")
            ax3p.axhline( 0.5, color="#555555", ls="--", lw=1.2, label="Saturation ±½")
            ax3p.axhline(-0.5, color="#555555", ls="--", lw=1.2)
            ax3p.set_xlabel(r"$H/J$", fontsize=11)
            ax3p.set_ylabel(r"$\langle M_z \rangle = \langle S^z_{tot}\rangle / N$",
                            fontsize=11, color="#1a1a2e")
            ax3p.set_title(f"{L_sel}×{L_sel}  Magnetisation Staircase",
                           fontsize=12)
            for sp in ax3p.spines.values(): sp.set_edgecolor("#cccccc")
            ax3p.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
                        labelcolor="#1a1a2e")
            ax3p.grid(alpha=0.3, color="#aaaaaa")

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # ── Validation box ─────────────────────────────────────
            if L_sel == 2 and sz0_evals is not None:
                e0_num = float(J_12 * sz0_evals[0])
                e0_ana = -2.0 * J_12
                err    = abs(e0_num - e0_ana)
                st.markdown(f"""
<div class="{'ok-box' if err < 1e-6 else 'err-box'}">
<b>2×2 Validation:</b>  Sz=0 ground state energy = {e0_num:.6f} J
(analytical = {e0_ana:.6f} J, error = {err:.2e})
{'✅ Matches!' if err < 1e-6 else '⚠️ Mismatch — check bond count'}
</div>""", unsafe_allow_html=True)

            # Critical fields
            jumps = np.where(np.diff(mz_arr) != 0)[0]
            if len(jumps):
                hc_str = ",  ".join([f"H/J ≈ {H_arr_12[j]/J_12:.3f}" for j in jumps[:8]])
                st.markdown(f'<div class="info-box"><b>Critical fields (magnetisation jumps):</b><br>{hc_str}</div>',
                            unsafe_allow_html=True)

            if len(skipped) > 0:
                st.markdown(f"""
<div class="warn-box">
<b>6×6 note:</b> {len(skipped)} Sz-sectors with D &gt; {MAX_DIM:,} were skipped
(the Sz=0 sector alone has C(36,18) ≈ 9 × 10⁹ states — impossible on any laptop).
The magnetisation staircase shows the near-saturation regime (|Sz| ≥ 14) where
exact diagonalisation is feasible. The AFM ground state at low H lives in the
inaccessible Sz≈0 sectors.
</div>""", unsafe_allow_html=True)

    else:
        with col_p:
            st.markdown(f"""
<div class="info-box">
<b>Key optimisation:</b> Each Sz-sector Hamiltonian is built <em>once</em> using
vectorised NumPy operations, then diagonalised once to get E₀(Sz).
The entire H-sweep is then computed analytically as<br>
&nbsp;&nbsp;<code>E(Sz, H) = J·E₀(Sz) + H·Sz</code><br>
with zero additional matrix work — O(n_sectors) diagonalisations total.<br><br>
• <b>2×2</b> – full Hilbert space, validates against E₀ = −2J<br>
• <b>4×4</b> – all 17 Sz-sectors (max D = 12,870), ~10 s<br>
• <b>6×6</b> – feasible sectors only (|Sz| ≥ 14, max D = 58,905), ~15 s<br>
  (Sz=0 sector has D ≈ 9 × 10⁹ — intractable without translation symmetry)
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
<b>Magnetisation staircase:</b> As H/J increases, S<sup>z</sup><sub>tot</sub> jumps
0 → 1 → … → N/2.  In the thermodynamic limit N→∞ this becomes a continuous
AFM→FM transition; (H/J)<sub>c</sub> is identified as the field at which
⟨M<sub>z</sub>⟩ reaches saturation.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 – Checkpoint 2.2.4 Bose-Fermi mixture
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-hdr">Checkpoint 2.2.4 – Bose-Fermi Mixture Density Profiles</p>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
Self-consistent Thomas-Fermi (bosons) + Local Density Approximation (fermions) solver.
All lengths are in units of the bosonic harmonic-oscillator length
a<sub>ho</sub> = √(ℏ/m<sub>B</sub>ω<sub>B</sub>);
energies in units of ℏω<sub>B</sub>.
</div>""", unsafe_allow_html=True)

    col_params, col_plots = st.columns([1, 2.8])

    with col_params:
        st.subheader("Parameters")
        st.markdown("**Masses (amu)**")
        mB_amu = st.number_input("Boson mass mB", value=87.0, min_value=1.0, step=1.0,
                                  help="e.g. 87Rb")
        mF_amu = st.number_input("Fermion mass mF", value=40.0, min_value=1.0, step=1.0,
                                  help="e.g. 40K")

        st.markdown("**Trap frequencies**")
        omega_ratio = st.slider("ωF / ωB", 0.5, 3.0, 1.0, 0.05)

        st.markdown("**Particle numbers**")
        NB = st.number_input("NB", value=10000, min_value=100, step=500)
        NF = st.number_input("NF", value=5000,  min_value=100, step=500)

        st.markdown("**Scattering lengths (in units of a_ho)**")
        aB_over_aho = st.slider("aB / a_ho", 0.001, 0.05, 0.01, 0.001, format="%.3f",
                                 help="Boson–boson scattering length")
        aBF_over_aho = st.slider("aBF / a_ho", -0.05, 0.05, 0.0, 0.001, format="%.3f",
                                  help="Inter-species: negative=attractive, positive=repulsive")

        regime = ("Non-interacting" if abs(aBF_over_aho) < 0.002
                  else ("Repulsive → phase separation" if aBF_over_aho > 0
                        else "Attractive → collapse tendency"))
        colour = "#1a5276" if abs(aBF_over_aho) < 0.002 else \
                 ("#7d6608" if aBF_over_aho > 0 else "#922b21")
        st.markdown(f'<div class="info-box" style="border-color:{colour};">'
                    f'<b>Regime:</b> {regime}</div>', unsafe_allow_html=True)

        run_bf = st.button("▶  Compute Density Profiles", type="primary")

    with col_plots:
        if run_bf:
            # ── Dimensionless couplings ───────────────────────────
            mass_ratio  = mF_amu / mB_amu          # mF/mB
            mu_red_frac = mass_ratio / (1 + mass_ratio)   # (mF/mB) / (1 + mF/mB)

            gB  = 4 * np.pi * aB_over_aho
            gBF = 2 * np.pi * aBF_over_aho / mu_red_frac

            omF = float(omega_ratio)

            # Radial grid
            R_max = 30.0
            Nr    = 1000
            r     = np.linspace(0.0, R_max, Nr)

            def norm3d(n, r_arr):
                integrand = n * 4 * np.pi * r_arr**2
                # np.trapz was removed in NumPy 2.0; use np.trapezoid with fallback
                trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
                return trapz_fn(integrand, r_arr)

            def nF_from_muF(muF, r_arr, nB_arr):
                VF  = 0.5 * mass_ratio * omF**2 * r_arr**2
                arg = muF - VF - gBF * nB_arr
                return np.where(arg > 0,
                                (2 * mass_ratio * arg)**1.5 / (6 * np.pi**2),
                                0.0)

            def nB_from_muB(muB, r_arr, nF_arr):
                VB = 0.5 * r_arr**2
                return np.maximum((muB - VB - gBF * nF_arr) / gB, 0.0)

            def solve_profiles(gBF_val, NB_target, NF_target, max_iter=100, tol=1e-5):
                # Initial chemical potentials from non-interacting TF/LDA estimates
                muB = (15 * NB_target * gB / (16 * np.pi))**(2/5) if gB > 0 else 5.0
                muF = (6 * np.pi**2 * NF_target
                       / (4/3 * np.pi * (8 * NF_target / (4/3 * np.pi))**(1))
                      )**(2/3) / (2 * mass_ratio) if NF_target > 0 else 1.0

                nB_arr = np.maximum((muB - 0.5*r**2) / max(gB, 1e-9), 0.0)
                nF_arr = np.zeros_like(r)

                for _ in range(max_iter):
                    nB_old = nB_arr.copy()
                    nF_old = nF_arr.copy()

                    # Rescale muF → NF
                    for _ in range(40):
                        nF_t = nF_from_muF(muF, r, nB_arr)
                        nf   = norm3d(nF_t, r)
                        if nf > 1e-15:
                            muF *= (NF_target / nf)**(1/3)
                        else:
                            muF *= 1.5
                        if abs(nf / max(NF_target, 1) - 1) < 1e-5: break
                    nF_arr = nF_from_muF(muF, r, nB_arr)

                    # Rescale muB → NB
                    for _ in range(40):
                        nB_t = nB_from_muB(muB, r, nF_arr)
                        nb   = norm3d(nB_t, r)
                        if nb > 1e-15:
                            muB *= (NB_target / nb)**(1/3)
                        else:
                            muB *= 1.5
                        if abs(nb / max(NB_target, 1) - 1) < 1e-5: break
                    nB_arr = nB_from_muB(muB, r, nF_arr)

                    delta = (np.max(np.abs(nB_arr - nB_old)) +
                             np.max(np.abs(nF_arr - nF_old)))
                    if delta < tol:
                        break

                return nB_arr, nF_arr, muB, muF

            def tf_radius(n_arr, r_arr, thresh=0.01):
                mx = np.max(n_arr)
                if mx < 1e-30: return 0.0
                idx = np.where(n_arr > thresh * mx)[0]
                return float(r_arr[idx[-1]]) if len(idx) else 0.0

            with st.spinner("Solving self-consistent profiles…"):
                # Non-interacting reference
                nB_ni, nF_ni, muB_ni, muF_ni = solve_profiles(0.0, NB, NF)
                # Interacting
                nB_int, nF_int, muB_int, muF_int = solve_profiles(gBF, NB, NF)

            RTF_B_ni  = tf_radius(nB_ni,  r)
            RTF_F_ni  = tf_radius(nF_ni,  r)
            RTF_B_int = tf_radius(nB_int, r)
            RTF_F_int = tf_radius(nF_int, r)

            # ── Plots ──────────────────────────────────────────────
            fig3 = plt.figure(figsize=(15, 5))
            fig3.patch.set_facecolor("white")
            gs_p = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.38)

            # Panel 1: density profiles
            ax1 = fig3.add_subplot(gs_p[0])
            ax1.set_facecolor("white")
            x_max = max(RTF_B_int*1.3, RTF_F_int*1.3, RTF_B_ni*1.3, RTF_F_ni*1.3, 6.0)
            ax1.plot(r, nB_int, color="#c0392b", lw=2.5, label=r"$n_B$ (interacting)")
            ax1.plot(r, nB_ni,  color="#c0392b", lw=1.5, ls="--", alpha=0.55,
                     label=r"$n_B$ (non-inter.)")
            ax1.plot(r, nF_int, color="#2980b9", lw=2.5, label=r"$n_F$ (interacting)")
            ax1.plot(r, nF_ni,  color="#2980b9", lw=1.5, ls="--", alpha=0.55,
                     label=r"$n_F$ (non-inter.)")
            ax1.set_xlim(0, x_max)
            ax1.set_xlabel(r"$r\,/\,a_{ho}$", fontsize=11)
            ax1.set_ylabel(r"$n(r)$ [arb. units]", fontsize=11, color="#1a1a2e")
            ax1.set_title("Density Profiles", fontsize=12)
            for sp in ax1.spines.values(): sp.set_edgecolor("#cccccc")
            ax1.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
                       labelcolor="#1a1a2e")
            ax1.grid(alpha=0.3, color="#aaaaaa")

            # Panel 2: TF radii bar chart
            ax2 = fig3.add_subplot(gs_p[1])
            ax2.set_facecolor("white")
            labels2 = ["BEC\n(non-int.)", "BEC\n(int.)", "Fermi\n(non-int.)", "Fermi\n(int.)"]
            vals2   = [RTF_B_ni, RTF_B_int, RTF_F_ni, RTF_F_int]
            bar_cols = ["#e8a0a0", "#c0392b", "#a0c4e8", "#2980b9"]
            bars = ax2.bar(labels2, vals2, color=bar_cols, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, vals2):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.05 * max(vals2),
                         f"{val:.2f}", ha="center", va="bottom",
                         fontsize=9, color="#1a1a2e")
            ax2.set_ylabel(r"Thomas-Fermi radius $[a_{ho}]$", fontsize=10,
                           color="#1a1a2e")
            ax2.set_title("Cloud Sizes", fontsize=12)
            ax2.tick_params(axis="x", labelsize=9)
            for sp in ax2.spines.values(): sp.set_edgecolor("#cccccc")
            ax2.grid(axis="y", alpha=0.3, color="#aaaaaa")

            # Panel 3: central density bar chart
            ax3 = fig3.add_subplot(gs_p[2])
            ax3.set_facecolor("white")
            vals3 = [nB_ni[0], nB_int[0], nF_ni[0], nF_int[0]]
            bars3 = ax3.bar(labels2, vals3, color=bar_cols, edgecolor="white", linewidth=1.5)
            mx3   = max(vals3) if max(vals3) > 0 else 1
            for bar, val in zip(bars3, vals3):
                ax3.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.02 * mx3,
                         f"{val:.1f}", ha="center", va="bottom",
                         fontsize=9, color="#1a1a2e")
            ax3.set_ylabel(r"$n(r=0)$ [arb. units]", fontsize=10, color="#1a1a2e")
            ax3.set_title("Central Density", fontsize=12)
            ax3.tick_params(axis="x", labelsize=9)
            for sp in ax3.spines.values(): sp.set_edgecolor("#cccccc")
            ax3.grid(axis="y", alpha=0.3, color="#aaaaaa")

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            # ── Metrics ───────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BEC TF radius",
                      f"{RTF_B_int:.2f} a_ho",
                      delta=f"{RTF_B_int - RTF_B_ni:+.2f} vs non-int.")
            c2.metric("Fermi TF radius",
                      f"{RTF_F_int:.2f} a_ho",
                      delta=f"{RTF_F_int - RTF_F_ni:+.2f} vs non-int.")
            c3.metric("BEC central density",
                      f"{nB_int[0]:.1f}",
                      delta=f"{nB_int[0] - nB_ni[0]:+.1f} vs non-int.")
            c4.metric("Fermi central density",
                      f"{nF_int[0]:.1f}",
                      delta=f"{nF_int[0] - nF_ni[0]:+.1f} vs non-int.")

            # Stability check
            dEF_dnF = ((1.0 / mass_ratio)
                       * (6 * np.pi**2 * nF_int[0])**(- 1/3)
                       if nF_int[0] > 1e-20 else 1e30)
            stable = gB * dEF_dnF > gBF**2
            box_cls = "ok-box" if stable else "err-box"
            stab_msg = ("✅ <b>Stable</b> mixture: g<sub>B</sub> · ∂E<sub>F</sub>/∂n<sub>F</sub> &gt; g<sub>BF</sub>²"
                        if stable else
                        "⚠️ <b>Unstable</b>: g<sub>B</sub> · ∂E<sub>F</sub>/∂n<sub>F</sub> &lt; g<sub>BF</sub>² → "
                        + ("phase separation" if gBF > 0 else "mean-field collapse"))
            st.markdown(f'<div class="{box_cls}">{stab_msg}</div>', unsafe_allow_html=True)

            if abs(aBF_over_aho) < 0.002:
                phys = ("No inter-species interaction; clouds are independent. "
                        "BEC forms a dense central peak; Fermi cloud extends further due to Fermi pressure.")
            elif gBF > 0:
                phys = ("Repulsive g<sub>BF</sub> &gt; 0: the BEC acts as a plunger, "
                        "pushing fermions outward. Fermi radius grows; BEC may shrink. "
                        "Large aBF leads to phase separation.")
            else:
                phys = ("Attractive g<sub>BF</sub> &lt; 0: bosons pull fermions toward "
                        "the centre, increasing central densities of both species. "
                        "Large |aBF| leads to mean-field collapse.")
            st.markdown(f'<div class="info-box"><b>Physical interpretation:</b> {phys}</div>',
                        unsafe_allow_html=True)

        else:
            st.markdown("""
<div class="info-box">
Set parameters on the left and click <b>▶ Compute Density Profiles</b>.<br><br>
The solver iterates the coupled TF/LDA equations until convergence,
renormalising chemical potentials at each step to enforce particle number:<br><br>
&nbsp;&nbsp;<b>Bosons (TF):</b>&nbsp;&nbsp;
n<sub>B</sub>(r) = [µ<sub>B</sub> − V<sub>B</sub>(r) − g<sub>BF</sub>·n<sub>F</sub>(r)] / g<sub>B</sub><br>
&nbsp;&nbsp;<b>Fermions (LDA):</b>&nbsp;
n<sub>F</sub>(r) = (1/6π²)[2m<sub>F</sub>(µ<sub>F</sub> − V<sub>F</sub>(r) − g<sub>BF</sub>·n<sub>B</sub>(r))/ℏ²]<sup>3/2</sup>
</div>""", unsafe_allow_html=True)