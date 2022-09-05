import io
import streamlit
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import base64

import shgo

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    # Add some CSS on top
    center = True
    css_justify = "center" if center else "left"
    css = '<p style="text-align:center; display: flex; justify-content: {};">'.format(css_justify)
    html = r'{}<img src="data:image/svg+xml;base64,{}"/>'.format(
        css, b64)
        
    # html = r'%s' % b64
    streamlit.write(html, unsafe_allow_html=True)
    return



def svg_circle(x,y,rx):
    ls_out = []
    return "\n".join(ls_out)


def from_coord_to_frame_1d(coord_limit, frame_limit, coord):
    coord_min = coord_limit[0]
    delta_coord = coord_limit[1] - coord_min

    frame_min = frame_limit[0]
    delta_frame = frame_limit[1] - frame_min

    frame = (coord - coord_min) * delta_frame / delta_coord
    return frame

def from_coord_to_frame_2d(coord_limit, frame_limit, coord):
    coord_limit_x = (coord_limit[0], coord_limit[2])
    coord_limit_y = (coord_limit[1], coord_limit[3])
    frame_limit_x = (frame_limit[0], frame_limit[2])
    frame_limit_y = (frame_limit[1], frame_limit[3])
    coord_x = coord[0]
    coord_y = -1*coord[1]
    frame_x = from_coord_to_frame_1d(coord_limit_x, frame_limit_x, coord_x)
    frame_y = from_coord_to_frame_1d(coord_limit_y, frame_limit_y, coord_y)
    frame = numpy.stack([frame_x, frame_y], axis=0)
    return frame


def svg_head(frame_limit):
    ls_out = []
    ls_out.append(f"<svg width=\"15cm\" height=\"15cm\" viewBox=\"{frame_limit[0]:} {frame_limit[1]:} {frame_limit[2]:} {frame_limit[3]:}\"")
    s_tail = """xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve">
<title>Example of path</title>
<defs>
<marker id='head' orient="auto"
        markerWidth='3' markerHeight='4'
        refX='0.1' refY='2'>
<path d='M0,0 V4 L2,2 Z' fill="black"/>
</marker>

</defs>
"""
    ls_out.append(s_tail)
    return "\n".join(ls_out)

def svg_tail():
    s_out = "</svg>"
    return s_out

def svg_circle(frame_xy, **d_arg):
    d_arg_keys = d_arg.keys()
    color = "#000000"
    if "color" in d_arg_keys:
        color = d_arg["color"]


    ls_circle = []
    for x, y in zip(frame_xy[0], frame_xy[1]):
        s_circle = f"<circle id=\"point_b\" r=\"2\" cx=\"{x:.2f}\" cy=\"{y:.2f}\" opacity=\"0.2\" fill=\"none\" stroke=\"{color:}\">\n</circle>\n"
        ls_circle.append(s_circle)
    return "\n".join(ls_circle)


def svg_animated_line_from_center(frame_xy, frame_center, **d_arg):
    d_arg_keys = d_arg.keys()
    color = "#000000"
    if "color" in d_arg_keys:
        color = d_arg["color"]

    ls_x, ls_y = [], []
    for x, y in zip(frame_xy[0], frame_xy[1]):
        ls_x.append(f"{x:.2f}")
        ls_y.append(f"{y:.2f}")
    s_x = ";".join(ls_x)
    s_y = ";".join(ls_y)

    ls_out = []
    ls_out.append(f"<line x1=\"{frame_center[0]:.2f}\" y1=\"{frame_center[1]:.2f}\" x2=\"{frame_xy[0,0]:.2f}\" y2=\"{frame_xy[1,0]:.2f}\" stroke=\"{color:}\" marker-end='url(#head)'>")
    ls_out.append(f"<animate attributeName=\"x2\" values=\"{s_x:}\" dur=\"9s\" repeatCount=\"indefinite\" />")
    ls_out.append(f"<animate attributeName=\"y2\" values=\"{s_y:}\" dur=\"9s\" repeatCount=\"indefinite\" />")
    ls_out.append("</line>")
    return "\n".join(ls_out)

def svg_animated_text(frame_xy, s_text: str, **d_arg):
    d_arg_keys = d_arg.keys()
    color = "#000000"
    if "color" in d_arg_keys:
        color = d_arg["color"]

    ls_x, ls_y = [], []
    for x, y in zip(frame_xy[0], frame_xy[1]):
        ls_x.append(f"{x:.2f}")
        ls_y.append(f"{y:.2f}")
    s_x = ";".join(ls_x)
    s_y = ";".join(ls_y)

    ls_out = []
    ls_out.append(f"<text x=\"{frame_xy[0,0]:.2f}\" y=\"{frame_xy[1,0]:.2f}\" font-size=\"12\" fill=\"{color:}\">{s_text}")
    ls_out.append(f"<animate attributeName=\"x\" values=\"{s_x:}\" dur=\"9s\" repeatCount=\"indefinite\" />")
    ls_out.append(f"<animate attributeName=\"y\" values=\"{s_y:}\" dur=\"9s\" repeatCount=\"indefinite\" />")
    ls_out.append("</text>")

    return "\n".join(ls_out)

def animated_block(frame_limit, coord_limit, coord_b, label_b, **d_arg):
    # simplified expression for the center
    frame_center = (0.5 * frame_limit[0]+ 0.5 * frame_limit[2], 0.5 * frame_limit[1]+ 0.5 * frame_limit[3])
    frame_b = from_coord_to_frame_2d(coord_limit, frame_limit, coord_b)
    svg_circle_b =  svg_circle(frame_b, **d_arg)
    svg_line = svg_animated_line_from_center(frame_b, frame_center, **d_arg)
    svg_text = svg_animated_text(frame_b, label_b, **d_arg)
    ls_out = [svg_circle_b, svg_line, svg_text]
    return "\n".join(ls_out)

def coord_limit(*argv):
    border = 1.1
    l_xy_max = []
    for coord_xy in argv:
        abs_coord_xy = numpy.abs(coord_xy)
        l_xy_max.append(numpy.max(abs_coord_xy, axis=1))
    np_xy_max = numpy.stack(l_xy_max, axis=1)
    xy_max = numpy.max(np_xy_max)*border
    coord_limit = (-xy_max, -xy_max, xy_max, xy_max)
    return coord_limit





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
    np_b_x, np_b_z, np_m_i_x, np_m_i_z = calc_fields_and_moments(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, field, n_points=n_points)
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

def calc_fields_and_moments(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, field, n_points: int = 90):

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
    np_b_x, np_b_z, np_m_i_x, np_m_i_z = calc_fields_and_moments(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, field, n_points=n_points)
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
# ni_theta = container_left.slider("Theta", min_value=0, max_value=360, value=0, step=1) * numpy.pi/180



b_moments = container_left.button("Calculate moments")
if b_moments:
    with streamlit.spinner("Please wait..."):
        l_moment, l_D, l_gamma = [], [], []
        for d_ion in l_D_ION:
            l_moment.append(d_ion["magnetic_moment"])
            l_D.append(d_ion["D"])
            l_gamma.append(d_ion["gamma"])
        moment_modulus_i = numpy.array(l_moment, dtype=float)
        np_d_i = numpy.array(l_D, dtype=float)
        np_gamma_i = numpy.array(l_gamma, dtype=float)

        dict_J_ij = {}
        for index, d_ex in D_EXCHANGE.items():
            dict_J_ij[index] = d_ex["J-scalar"]
        n_points = 90
        # fig, s_report = calc_and_plot_arrows(l_D_ION, D_EXCHANGE, ni_theta, ni_field)
        np_b_x, np_b_z, np_m_i_x, np_m_i_z = calc_fields_and_moments(moment_modulus_i, dict_J_ij, np_gamma_i, np_d_i, ni_field, n_points=n_points)
    
    
    coord_b = numpy.stack([np_b_x, np_b_z], axis=0)
    coord_m = numpy.stack([np_m_i_x, np_m_i_z], axis=0)
    n_m = coord_m.shape[2]
    coord_m_total = numpy.sum(coord_m, axis=2)
    l_coord_m = [coord_m[:, :, i] for i in range(n_m)]
    
    coord_limit = coord_limit(coord_b, coord_m_total, *l_coord_m)
    frame_limit = (0, 0, 150, 150)
    label_b = "B"
    label_m_total = "m"
    ls_svg = []
    ls_svg.append(svg_head(frame_limit))

    svg_block_b = animated_block(frame_limit, coord_limit, coord_b, label_b, color="#ff0000")
    ls_svg.append(svg_block_b)

    svg_block_m_total = animated_block(frame_limit, coord_limit, coord_m_total, label_m_total, color="#0000ff")
    ls_svg.append(svg_block_m_total)

    for i_c_m, c_m in enumerate(l_coord_m):
        label_m = f"m{i_c_m+1:}"
        svg_block_m_i = animated_block(frame_limit, coord_limit, c_m, label_m, color="#444444")
        ls_svg.append(svg_block_m_i)

    ls_svg.append(svg_tail())

    render_svg("\n".join(ls_svg))

    

# D_PARAMETERS = {
#     "ions": l_D_ION,
#     "exchange": D_EXCHANGE
# }
# f_io = io.BytesIO()
# numpy.save(f_io, D_PARAMETERS, allow_pickle=True)
# 
# sb.download_button("Download parameters into '.npy'", f_io, file_name="parameters.npy")


