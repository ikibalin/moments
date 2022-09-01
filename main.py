import io
import streamlit
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import shgo



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
    energy = energy_ex + energy_an + energy_zee
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

    ax.arrow(0,0, b_x, b_z, color="black", head_width=0.1, label=f"B: ({b_x:.2f}, {b_z:.2f})")
    ax.text(kk_text*b_x+delta_text, kk_text*b_z+delta_text, "B")
    for i_ion in range(n_i):
        ax.arrow(0,0, m_i_x[i_ion], m_i_z[i_ion], head_width=0.1, alpha=0.5, label=f"{i_ion+1:}: ({m_i_x[i_ion]:.1f}, {m_i_z[i_ion]:.1f})")
        ax.text(kk_text*m_i_x[i_ion]+delta_text, kk_text*m_i_z[i_ion]+delta_text, f"{i_ion+1:}")
    ax.arrow(0, 0, m_x, m_z, head_width=0.1, color="green", alpha=0.5, label=f"m: ({m_x:.2f}, {m_z:.2f})")
    ax.text(kk_text*m_x+delta_text, kk_text*m_z+delta_text, f"m")
    ax.legend(loc=0)
    return fig

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
        # ax.text(1.5,0.9, f"$\phi={phi*180/numpy.pi:5.1f}$\n$B={B:3.1f}$\n$D={D:3.1f}$\n$J={J:3.1f}$\n$b_1 ={beta_1*180/numpy.pi:5.0f}$\n$b_2={beta_2*180/numpy.pi:5.0f}$\n$E={energy:.1f}$")
        l_m_i_x.append(m_i_x)
        l_m_i_z.append(m_i_z)
        l_b_x.append(b_x)
        l_b_z.append(b_z)

    np_m_i_x = numpy.stack(l_m_i_x, axis=0)
    np_m_i_z = numpy.stack(l_m_i_z, axis=0)
    np_b_x = numpy.stack(l_b_x, axis=0)
    np_b_z = numpy.stack(l_b_z, axis=0)
    fig_1, anim_1 = plot_moments(np_b_x, np_b_z, np_m_i_x, np_m_i_z)
    # fig_1.show()
    return fig_1, anim_1


def get_latex_zeeman():
    str_h_zee = r"-\mu_B \left( B_x \cdot M_x + B_z \cdot M_z \right) "
    return str_h_zee


def get_latex_zfs():
    str_h_zfs = r"- D \cdot \left( \cos \gamma M_z + \sin \gamma M_x \right)^2" #
    return str_h_zfs





streamlit.markdown('# Energy')

streamlit.markdown("The program calculate induced magnetic moment.")



def form_ion(key, d_ion: dict):
    d_ion_keys = d_ion.keys()

    expander = streamlit.expander("Parameters of magnetic ion")
    
    if "magnetic_moment" in d_ion_keys:
        value = d_ion["magnetic_moment"]
    else:
        value = 1.
    d_ion["magnetic_moment"] = expander.number_input(r"Magnetic moment (mu_B)", 0., 10., value=value, step=0.1, key=key)
    
    str_h_zee = get_latex_zeeman()
    
    str_h_zfs_1 = get_latex_zfs()
    
    expander.latex("E_{ion} = " + str_h_zee  + str_h_zfs_1)
    
    if "D" in d_ion_keys:
        value_1 = d_ion["D"]
        value_2 = d_ion["gamma"]
    else:
        value_1 = 0.
        value_2 = 0.
    d_ion["D"] = expander.number_input("D", -3000., 3000., value=value_1, step=1.00, key=key)
    d_ion["gamma"] = expander.number_input("gamma", -180., 180., value=value_2, step=1.00, key=key) * numpy.pi/180.

    return 



sb = streamlit.sidebar
f_parameters = sb.file_uploader("Load parameters from '.npy'", type="npy")
if f_parameters is not None:
    D_PARAMETERS = numpy.load(f_parameters, allow_pickle=True).take(0)
    l_D_ION = D_PARAMETERS["ions"]
    n_ions = len(l_D_ION)
    D_EXCHANGE = D_PARAMETERS["exchange"]
    n_ions = streamlit.number_input("Number of magnetic ions", 1, 5, value=n_ions, step=1)
else:
    n_ions = streamlit.number_input("Number of magnetic ions", 1, 5, value=1, step=1)
    l_D_ION = [{} for i_ion in range(n_ions)]
    D_EXCHANGE = {}
    for i_ion_1 in range(0, n_ions-1):
        for i_ion_2 in range(i_ion_1+1, n_ions):
            t_name = (i_ion_1, i_ion_2)
            D_EXCHANGE[t_name] = {}


for i_ion, D_ION in enumerate(l_D_ION):
    # First ion
    streamlit.markdown(f'## Ion {i_ion+1:}')
    form_ion(f"ion_{i_ion+1:}", D_ION)


# Exchange term
streamlit.markdown('## Exchange term')
expander_exchange = streamlit.expander("Parameters of exchange term")
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



expander_moments = sb.container()
ni_field = expander_moments.number_input("Magnetic field (in T)", 0., 20., value=1., step=0.1)
ni_theta = expander_moments.number_input("Theta", 0., 360., value=0., step=1.) * numpy.pi/180


b_moments = expander_moments.button("Interacting ions")
if b_moments:
    fig = calc_and_plot_arrows(l_D_ION, D_EXCHANGE, ni_theta, ni_field)
    # fig, anim = calc_by_dictionary(l_D_ION, D_EXCHANGE, ni_field)
    streamlit.pyplot(fig)

    # s_report = get_report_energy(hamiltonian, ni_temp)

    # plt_fig = get_fig_ellipsoids(l_D_ION, D_EXCHANGE, ni_temp, ni_f_x, ni_f_y, ni_f_z)
    # streamlit.pyplot(plt_fig# )

    # streamlit.code("\n".join(ls_report))

# b_gif = expander_moments.button("Calc gif into file")
# if b_gif:
#     fig, anim = calc_gif_by_dictionary(l_D_ION, D_EXCHANGE, ni_field)
#     f_io_2 = io.BytesIO()
#     anim.save(f_io_2,  writer='imagemagick', fps=30, dpi=300)
#     sb.download_button("Download calculations", f_io_2, file_name="anim.gif")
#     # streamlit.pyplot(fig)
    

# for i_ion in range(n_ions):
#     b_report_1 = sb.button(f"Isolated ion {i_ion+1:}")
#     if b_report_1:
#         D_ION = l_D_ION[i_ion]
#     
#         # plt_fig = get_fig_ellipsoids([D_ION, ], {}, ni_temp, ni_f_x, ni_f_y, ni_f_z)
#         # streamlit.pyplot(plt_fig)
#         # streamlit.code(s_moment+s_report)



D_PARAMETERS = {
    "ions": l_D_ION,
    "exchange": D_EXCHANGE
}
f_io = io.BytesIO()
numpy.save(f_io, D_PARAMETERS, allow_pickle=True)

sb.download_button("Download parameters into '.npy'", f_io, file_name="parameters.npy")


# df = pandas.DataFrame({
#   'first column': [1, 2, 3, 4],
#   'second column': [10, 20, 30, 40]
# })



# streamlit.dataframe(df)
# b_clicked = streamlit.button("Click it")
# cb = streamlit.checkbox("I read the conditions")
# rb = streamlit.radio("What you prefer?", ["Tea", "Coffee", "Nothing"])
# sb = streamlit.selectbox("Do you want cookies?", ["Yes", "No"])
# ms = streamlit.multiselect("Buy?", ["Apples", "Banana", "Coffee", "Tea"])
# ti = streamlit.text_input("First name")
# ni = streamlit.number_input("Choose your age", 5, 127)
# te = streamlit.text_area("Text to translate.")
# di = streamlit.date_input("Date of the meeting")
# fu = streamlit.file_uploader("Choose input file")
# ci = streamlit.camera_input("You face")
# streamlit.metric("Fe2", 76, +3)

# streamlit.line_chart(df)