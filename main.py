import io
import streamlit
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import shgo

COEFF_MU_B = 927.40100783/1.380649*1e-3 # mu_B/k_B
COEFF_INV_CM = 0.124 * 11.6 #cm^-1 -> meV -> K

def calc_energy(beta_i, moment_modulus_i, dict_J_ij, gamma_i, D_i, theta, B):
    """
    Calc sublattice-moment-related free energy 
    
    FIXME: put correct coefficients.
    """
    m_i_x = calc_m_i_x(beta_i, moment_modulus_i)
    m_i_z = calc_m_i_z(beta_i, moment_modulus_i)
    m_i_z_sq = numpy.square(m_i_z)

    energy_ex = 0
    for (index_i, index_j), j_iso in dict_J_ij.items():
        energy_ex += j_iso * moment_modulus_i[index_i]* moment_modulus_i[index_j] * numpy.cos(beta_i[index_i]-beta_i[index_j])

    b_x = calc_b_x(theta, B)
    b_z = calc_b_z(theta, B)

    

    energy_an = - numpy.sum(D_i * numpy.square(numpy.cos(gamma_i) * m_i_z + numpy.sin(gamma_i) * m_i_x), axis=0)
    energy_zee = - (
        b_x * numpy.sum(m_i_x, axis=0) +
        b_z * numpy.sum(m_i_z, axis=0))

    coeff_ex = COEFF_INV_CM * COEFF_MU_B * COEFF_MU_B
    coeff_an = coeff_ex
    coeff_zee = COEFF_MU_B * COEFF_MU_B
    
    energy = coeff_ex*energy_ex + coeff_an*energy_an + coeff_zee*energy_zee
    return energy

def calc_m_i_x(beta_i, moment_modulus_i):
    return moment_modulus_i*numpy.sin(beta_i)

def calc_m_i_z(beta_i, moment_modulus_i):
    return moment_modulus_i*numpy.cos(beta_i)

def calc_b_x(theta, B):
    return B*numpy.sin(theta)

def calc_b_z(theta, B):
    return B*numpy.cos(theta)


def calc_beta12(moment_modulus_i, dict_J_ij, gamma_i, D_i, theta, B):
    n_beta_i = moment_modulus_i.size
    bounds_beta = n_beta_i * [(0, 2*numpy.pi),]
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds_beta,
        "args": (moment_modulus_i, dict_J_ij, gamma_i, D_i, theta, B),}

    beta_i = numpy.zeros((n_beta_i), dtype=float)
    res = shgo.shgo(lambda x: calc_energy(x, moment_modulus_i, dict_J_ij, gamma_i, D_i, theta, B), bounds_beta)
    
    beta_i_opt = res["x"]
    energy = res["fun"]
    return beta_i_opt, energy


def plot_moments(np_b_x, np_b_z, np_m_i_x, np_m_i_z):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    np_m_x = numpy.sum(np_m_i_x, axis=1)
    np_m_z = numpy.sum(np_m_i_z, axis=1)
    hh = numpy.max([
        numpy.abs(np_m_i_x).max(), numpy.abs(np_m_i_z).max(),
        numpy.abs(np_m_x).max(), numpy.abs(np_m_z).max(),
        numpy.abs(np_b_x).max(), numpy.abs(np_b_z).max(),
        ])
    ax.set_xlim(-hh*1.1, hh*1.1)
    ax.set_ylim(-hh*1.1, hh*1.1)
    ax.set_xlabel("$M_x$")
    ax.set_ylabel("$M_z$")
    # ax.arrow(0,0, 0, 1, color="black", head_width=0.1)
    
    ax.plot(np_b_x, np_b_z, ":k", alpha=0.1)
    n_i = np_m_i_x.shape[1]
    for i_ion in range(n_i):
        ax.plot(np_m_i_x[:,i_ion], np_m_i_z[:,i_ion], "o", alpha=0.1)
    ax.plot(np_m_x, np_m_z, "og", alpha=0.1)
    # ax.set_title(f"B={B:.2f}, J={dict_J_ij[(0,1)]:.2f}, D={D:.2f}, ")
    
    ims = []
    for b_x, b_z, m_i_x, m_i_z, m_x, m_z in zip(np_b_x, np_b_z, np_m_i_x, np_m_i_z, np_m_x, np_m_z):
        
        im1 = ax.arrow(0,0, b_x, b_z, color="black", head_width=0.1, animated=True)
        l_im = [im1]
        for i_ion in range(n_i):
            im_i = ax.arrow(0,0, m_i_x[i_ion], m_i_z[i_ion], head_width=0.1, alpha=0.5, animated=True)
            l_im.append(im_i)
        im4 = ax.arrow(0, 0, m_x, m_z, head_width=0.1, color="green", alpha=0.5, animated=True)
        l_im.append(im4)
        ims.append(tuple(l_im))
        
        
    anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=100)
    # anim.save('animation.gif', writer='imagemagick', fps=60, dpi=300)
    return fig, anim


def calc_and_plot_arrows(l_d_ion, d_exchange, theta, field):
    l_moment, l_D, l_gamma = [], [], []
    for d_ion in l_d_ion:
        l_moment.append(d_ion["magnetic_moment"])
        l_D.append(d_ion["D"])
        l_gamma.append(d_ion["gamma"])
    moment_modulus_i = numpy.array(l_moment, dtype=float)
    np_d_i = numpy.array(l_D, dtype=float)
    np_gamma_i = numpy.array(l_gamma, dtype=float)

    dict_J_ij = {}
    for index, d_ex in d_exchange.items():
        dict_J_ij[index] = d_ex["J-scalar"]

    beta_i, energy = calc_beta12(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, theta, field)
    m_i_x = calc_m_i_x(beta_i, moment_modulus_i)
    m_i_z = calc_m_i_z(beta_i, moment_modulus_i)
    b_x = calc_b_x(theta, field)
    b_z = calc_b_z(theta, field)

    m_x = numpy.sum(m_i_x, axis=0)
    m_z = numpy.sum(m_i_z, axis=0)


    hh = numpy.max([numpy.abs(m_x), numpy.abs(m_z), numpy.abs(b_x), numpy.abs(b_z), numpy.max(moment_modulus_i)])
    delta_text = 0.01*hh
    kk_text = 1.1
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-hh*1.3, hh*1.3)
    ax.set_ylim(-hh*1.3, hh*1.3)
    ax.set_xlabel("$M_x$")
    ax.set_ylabel("$M_z$")
    # ax.arrow(0,0, 0, 1, color="black", head_width=0.1)
    n_i = moment_modulus_i.size

    ls_report = []
    ls_report.append(f"The energy $E$ is minimal ({energy:.3f} K) at parameters")
    ls_report.append(f"|       |         $x$ |         $z$ |")
    ls_report.append(f"| ----- | ---------:| ---------:|")
    ls_report.append(f"| $B (T)$ | ${b_x:.3f}$ | ${b_z:.3f}$ |")

    ax.arrow(0,0, b_x, b_z, color="black", head_width=0.1, length_includes_head=True)
    ax.text(kk_text*b_x+delta_text, kk_text*b_z+delta_text, "B")
    for i_ion in range(n_i):
        ax.arrow(0,0, m_i_x[i_ion], m_i_z[i_ion], head_width=0.1, length_includes_head=True, alpha=0.5)
        ls_report.append(f"| $M_{i_ion+1:} (\mu_B)$ | ${m_i_x[i_ion]:.3f}$ | ${m_i_z[i_ion]:.3f}$|")
        ax.text(kk_text*m_i_x[i_ion]+delta_text, kk_text*m_i_z[i_ion]+delta_text, f"{i_ion+1:}")
    ax.arrow(0, 0, m_x, m_z, head_width=0.1, color="green", alpha=0.5, length_includes_head=True)
    ls_report.append(f"| Total $M (\mu_B)$ | ${m_x:.3f}$ | ${m_z:.3f}$| ")
    ax.text(kk_text*m_x+delta_text, kk_text*m_z+delta_text, f"m")

    n_points = 90
    np_b_x, np_b_z, np_m_i_x, np_m_i_z = calc_fields_and_moments(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, theta, field, n_points=n_points)
    np_m_x = numpy.sum(np_m_i_x, axis=1)
    np_m_z = numpy.sum(np_m_i_z, axis=1)
    hh = numpy.max([
        numpy.abs(np_m_i_x).max(), numpy.abs(np_m_i_z).max(),
        numpy.abs(np_m_x).max(), numpy.abs(np_m_z).max(),
        numpy.abs(np_b_x).max(), numpy.abs(np_b_z).max(),
        ])
    ax.set_xlim(-hh*1.1, hh*1.1)
    ax.set_ylim(-hh*1.1, hh*1.1)
    ax.set_xlabel("$M_x$")
    ax.set_ylabel("$M_z$")
    # ax.arrow(0,0, 0, 1, color="black", head_width=0.1)
    
    ax.plot(np_b_x, np_b_z, ":k", alpha=0.1)
    n_i = np_m_i_x.shape[1]
    for i_ion in range(n_i):
        ax.plot(np_m_i_x[:,i_ion], np_m_i_z[:,i_ion], "o", alpha=0.1)
    ax.plot(np_m_x, np_m_z, "og", alpha=0.1)


    # ax.legend(loc=0)

    return fig, "\n".join(ls_report)

def calc_fields_and_moments(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, theta, field, n_points: int = 90):

    l_m_i_x, l_m_i_z = [], []
    l_b_x, l_b_z = [], []
    np_theta = numpy.linspace(0, 2*numpy.pi, n_points+1)
    for i_theta, theta in enumerate(np_theta):
        beta_i, energy = calc_beta12(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, theta, field)
        m_i_x = calc_m_i_x(beta_i, moment_modulus_i)
        m_i_z = calc_m_i_z(beta_i, moment_modulus_i)
        b_x = calc_b_x(theta, field)
        b_z = calc_b_z(theta, field)

        print(f"Theta {theta*180/numpy.pi:.1f}; energy {energy:.3f}", end="\r")
        l_m_i_x.append(m_i_x)
        l_m_i_z.append(m_i_z)
        l_b_x.append(b_x)
        l_b_z.append(b_z)

    np_m_i_x = numpy.stack(l_m_i_x, axis=0)
    np_m_i_z = numpy.stack(l_m_i_z, axis=0)
    np_b_x = numpy.stack(l_b_x, axis=0)
    np_b_z = numpy.stack(l_b_z, axis=0)
    return np_b_x, np_b_z, np_m_i_x, np_m_i_z

def calc_gif_by_dictionary(l_d_ion, d_exchange, field):
    l_moment, l_D, l_gamma = [], [], []
    for d_ion in l_d_ion:
        l_moment.append(d_ion["magnetic_moment"])
        l_D.append(d_ion["D"])
        l_gamma.append(d_ion["gamma"])
    moment_modulus_i = numpy.array(l_moment, dtype=float)
    np_d_i = numpy.array(l_D, dtype=float)
    np_gamma_i = numpy.array(l_gamma, dtype=float)

    dict_J_ij = {}
    for index, d_ex in d_exchange.items():
        dict_J_ij[index] = d_ex["J-scalar"]

    n_points = 90
    np_b_x, np_b_z, np_m_i_x, np_m_i_z = calc_fields_and_moments(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, theta, field, n_points=n_points)
    fig_1, anim_1 = plot_moments(np_b_x, np_b_z, np_m_i_x, np_m_i_z)
    # fig_1.show()
    return fig_1, anim_1


def get_latex_zeeman():
    str_h_zee = r"-\mu_B \sum_i^{N} \left( \mathbf{B} \cdot \mathbf{M}_i\right) "
    return str_h_zee


def get_latex_zfs():
    str_h_zfs = r"- \sum_i^{N} D_i M_i^2 \cdot \cos^2 \left( \gamma_i-\beta_i \right) " #
    return str_h_zfs

def get_latex_exchange():
    str_h_ex = r"+ \sum_{i,j, i \ne j} J_{ij} M_{i} M_{j} \cdot  \cos \left( \beta_i  - \beta_j\right)" #
    return str_h_ex



streamlit.markdown("Calculate magnetic moments $\mathbf{M}_i$ at given parameters of free energy $E$ defined by Zeeman splitting, anisotropy and exchange terms: ")


str_h_zee = get_latex_zeeman()
    
str_h_zfs = get_latex_zfs()

str_h_ex = get_latex_exchange()

streamlit.latex("E = " + str_h_zee  + str_h_zfs + str_h_ex)


container_left, container_right = streamlit.columns(2)

def form_ion(key, d_ion: dict, i_ion: int):
    d_ion_keys = d_ion.keys()

    expander = container_right.expander(f"Ion {i_ion:}")
    
    if "magnetic_moment" in d_ion_keys:
        value = d_ion["magnetic_moment"]
    else:
        value = 1.
    d_ion["magnetic_moment"] = expander.number_input(f"M_{i_ion:} (mu_B):", 0., 10., value=value, step=0.1, key=key+"_MM")
    

    
    if "D" in d_ion_keys:
        value_1 = d_ion["D"]
        value_2 = d_ion["gamma"]
    else:
        value_1 = 0.
        value_2 = 0.
    d_ion["D"] = expander.number_input(f"D_{i_ion:} (cm^-1):", -3000., 3000., value=value_1, step=1.00, key=key+"_D")
    d_ion["gamma"] = expander.number_input(f"gamma_{i_ion:}:", -180., 180., value=value_2, step=1.00, key=key+"_gamma") * numpy.pi/180.

    return 


# sb = streamlit.sidebar
# f_parameters = sb.file_uploader("Load parameters from '.npy'", type="npy")
# if f_parameters is not None:
#     D_PARAMETERS = numpy.load(f_parameters, allow_pickle=True).take(0)
#     l_D_ION = D_PARAMETERS["ions"]
#     n_ions = len(l_D_ION)
#     D_EXCHANGE = D_PARAMETERS["exchange"]
#     n_ions = streamlit.number_input("Number of magnetic ions (N)", 1, 5, value=n_ions, step=1)
# else:
n_ions = container_right.number_input("Number of magnetic ions (N)", 1, 5, value=1, step=1)
l_D_ION = [{} for i_ion in range(n_ions)]
D_EXCHANGE = {}
for i_ion_1 in range(0, n_ions-1):
    for i_ion_2 in range(i_ion_1+1, n_ions):
        t_name = (i_ion_1, i_ion_2)
        D_EXCHANGE[t_name] = {}


for i_ion, D_ION in enumerate(l_D_ION):
    # First ion
    # streamlit.markdown(f'## Ion {i_ion+1:}')
    form_ion(f"ion_{i_ion+1:}", D_ION, i_ion+1)


# Exchange term
if n_ions>1:
    # streamlit.markdown('## Exchange term')
    expander_exchange = container_right.expander("Exchange")
    for i_ion_1 in range(0, n_ions-1):
        for i_ion_2 in range(i_ion_1+1, n_ions):
            t_name = (i_ion_1, i_ion_2)
            if t_name in D_EXCHANGE.keys():
                D_EXCHANGE_ij = D_EXCHANGE[t_name]
            else:
                D_EXCHANGE_ij = {}
                D_EXCHANGE[t_name] = D_EXCHANGE_ij
    
            if "J-scalar" in D_EXCHANGE_ij.keys():
                value = D_EXCHANGE_ij["J-scalar"]
            else:
                value = 0.
            D_EXCHANGE_ij["J-scalar"] = expander_exchange.number_input(f"J-iso between ions {i_ion_1+1:} and {i_ion_2+1:}", -3000., 3000., value=value, step=0.1)



# expander_moments = sb.container()
ni_field = container_left.number_input("Magnetic field (in T)", 0., 20., value=1., step=0.1)
ni_theta = container_left.slider("Theta", min_value=0, max_value=360, value=0, step=1) * numpy.pi/180



b_moments = container_left.button("Calculate moments")
if b_moments:
    with streamlit.spinner("Please wait..."):
        fig, s_report = calc_and_plot_arrows(l_D_ION, D_EXCHANGE, ni_theta, ni_field)
    # fig, anim = calc_by_dictionary(l_D_ION, D_EXCHANGE, ni_field)
    # col1, col2 = container_left.columns(2)
    container_left.pyplot(fig)
    container_left.markdown(s_report)



# D_PARAMETERS = {
#     "ions": l_D_ION,
#     "exchange": D_EXCHANGE
# }
# f_io = io.BytesIO()
# numpy.save(f_io, D_PARAMETERS, allow_pickle=True)
# 
# sb.download_button("Download parameters into '.npy'", f_io, file_name="parameters.npy")


