import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from scipy.optimize import root_scalar
import argparse
import matplotlib as mpl
import plotly.graph_objects as go

BOHR_TO_ANGSTROM = 0.5291772109

# User input
parser = argparse.ArgumentParser(
    description='Select the state, charge, and spin of the orbital.')


def parse_bool(value):
    """Parse the string boolean values used by the command-line interface."""
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise argparse.ArgumentTypeError("expected 'True' or 'False'")


parser.add_argument('--st', type=int, required=True,
                    help='Target state number.')
parser.add_argument('--q', type=int, default=0, help='Target charge state.')
parser.add_argument('--k', type=int, default=1,
                    choices=[1, 2], help='spin: up=1, dn=2.')
parser.add_argument('--dir', default='static/', help='Path to static.')
parser.add_argument('--lmax', type=int, default=10,
                    help='Maximum l of orbitals.')
parser.add_argument('--tol', type=float, default=1.e-7,
                    help='Asymptotic value for matching wave function.')
parser.add_argument('--plot2D', type=parse_bool, default=False,
                    choices=[True, False], help='Plot 2D wave function')
parser.add_argument('--plane', default='xz',choices=['xy', 'xz', 'yz'],
                    help='Plane to plot 2D wave function.')
parser.add_argument('--plot3D', type=parse_bool, default=False,
                    choices=[True, False], help='Visualize 3D wave function')
parser.add_argument('--iso', type=float, default=0.1,
                    help='Value for isosurface of the wave function.')
parser.add_argument('--outwf', type=parse_bool, default=False,
                    choices=[True, False],
                    help='Output the wave function on the grid.')
parser.add_argument('--plotMolecule', type=parse_bool, default=False,
                    choices=[True, False],
                    help='Plot the molecule from an XYZ file as a 3D ball-and-stick model.')
parser.add_argument('--xyz', default='fenchone_cat_maxTIR_rotation.xyz',
                    help='XYZ file used for the molecular ball-and-stick plot.')
parser.add_argument('--molout', default='',
                    help='Output HTML file for the molecular plot. Defaults beside the XYZ file.')
# Parse the command-line arguments
args = parser.parse_args()
st = args.st
k = args.k
q = args.q
dir = args.dir
l_max = args.lmax
match_tol = args.tol
Plot2D = args.plot2D
Plot3D = args.plot3D
iso_val = abs(args.iso)
plane = args.plane
print_output = args.outwf
PlotMolecule = args.plotMolecule
xyz_file_path = args.xyz
if k == 1:
    choose_spin = 'up'
elif k == 2:
    choose_spin = 'dn'
Z = q+1.
info_file_path = dir + 'info'

# Read static/info to obtain eigenvalues


def read_static_info(file_path):
    in_data_section = False
    stop_reading = False
    spin_pol = False
    Ip = []
    count = 0
    with open(file_path, 'r') as info_file:
        for line in info_file:
            line = line.strip()
            if in_data_section:
                if len(line) == 0:
                    stop_reading = True
                if not stop_reading:
                    values = line.split()
                    if values[1] != '--' and count == 0:
                        Ip_dn = []
                        spin_pol = True
                    if spin_pol:
                        if values[1] == 'up':
                            Ip.append(float(values[2]))
                        elif values[1] == 'dn':
                            Ip_dn.append(float(values[2]))
                    else:
                        Ip.append(float(values[2]))
                    count += 1
            elif line.startswith("#st "):
                in_data_section = True
    if spin_pol:
        return spin_pol, [Ip, Ip_dn]
    else:
        return spin_pol, Ip


print('Reading input file', info_file_path)
spin_pol, Ip_list = read_static_info(info_file_path)
print('Spin polarized is', spin_pol)
if spin_pol:
    print('The spin orbital is', choose_spin)
    Nst = np.size(Ip_list[0])
    print('Total number of states is', Nst)
else:
    Nst = np.size(Ip_list)
    print('Total number of states is', Nst)
print('The target orbital is', st)
print('Maximum l is', l_max)
print('Effective charge for Coulomb tail is', Z)
print('Matching Coulomb tail with value', match_tol)

if spin_pol:
    if choose_spin == 'up':
        dx_file_path = dir + 'wf-k001-st' + str(st).zfill(5) + '.dx'
    elif choose_spin == 'dn':
        dx_file_path = dir + 'wf-k002-st' + str(st).zfill(5) + '.dx'
else:
    dx_file_path = dir + 'wf-st' + str(st).zfill(5) + '.dx'


def read_dx_wf(dx_file_path):
    grid_values = []
    grid_counts = []
    grid_origin = []
    grid_spacing = []
    in_data_section = False
    stop_reading = False
    with open(dx_file_path, 'r') as dx_file:
        for line in dx_file:
            line = line.strip()
            if in_data_section:
                if line.startswith("object "):
                    stop_reading = True
                if not stop_reading:
                    values = line.split()
                    grid_values.extend(map(float, values))
            elif line.startswith("object 1"):
                values = line.split()[-3:]
                grid_counts.extend(map(int, values))
            elif line.startswith("origin"):
                values = line.split()[-3:]
                grid_origin.extend(map(float, values))
                count = 1
            elif line.startswith("delta"):
                values = line.split()[count]
                grid_spacing.append(float(values))
                count += 1
            elif line.startswith("object 3 class array"):
                in_data_section = True

    return grid_counts, grid_origin, grid_spacing, grid_values


# Read grid and electron wave functions values
print('Reading input file', dx_file_path)
grid_counts, grid_origin, grid_spacing, wf_values = read_dx_wf(dx_file_path)
print('grid_counts=', grid_counts)
print('grid_origin=', grid_origin)
print('grid_spacing=', grid_spacing)

r_max = np.abs(grid_origin).max()

# Reshape the values into a 3D grid
wf_grid = np.array(wf_values).reshape(grid_counts)

# Create coordinate arrays for each axis
x_coords = np.linspace(grid_origin[0], grid_origin[0] +
                       (grid_counts[0] - 1) * grid_spacing[0], grid_counts[0])
y_coords = np.linspace(grid_origin[1], grid_origin[1] +
                       (grid_counts[1] - 1) * grid_spacing[1], grid_counts[1])
z_coords = np.linspace(grid_origin[2], grid_origin[2] +
                       (grid_counts[2] - 1) * grid_spacing[2], grid_counts[2])

# Create a regular grid interpolator
print('Interpolating wave function on the grid...')
wf_interpol = RegularGridInterpolator((x_coords, y_coords, z_coords), wf_grid)
print('Done.')

# Calculate the permanent dipole moment of the orbital.
X_g, Y_g, Z_g = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
pdm = np.zeros(3)
tmp = np.trapz(np.abs(wf_grid)**2*X_g,x_coords,axis=0)
tmp1 = np.trapz(tmp, y_coords, axis=0)
pdm[0] = np.trapz(tmp1, z_coords)
tmp = np.trapz(np.abs(wf_grid)**2*Y_g,y_coords,axis=1)
tmp1 = np.trapz(tmp, x_coords, axis=0)
pdm[1] = np.trapz(tmp1, z_coords)
tmp = np.trapz(np.abs(wf_grid)**2*Z_g,z_coords,axis=2)
tmp1 = np.trapz(tmp, x_coords, axis=0)
pdm[2] = np.trapz(tmp1, y_coords)
print('Permanet dipole moment (a.u.) is', pdm)
# ----------------------------------------------------------------------------
#             Set up the frame of the figure
# ----------------------------------------------------------------------------
#plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 18
#plt.rcParams['text.usetex'] = True
#plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['xtick.major.size'] = 8.
plt.rcParams['xtick.minor.size'] = 5.
plt.rcParams['ytick.major.size'] = 5.
plt.rcParams['ytick.minor.size'] = 3.
plt.rcParams['lines.linewidth'] = 2.5


if print_output:
    # Create a meshgrid of coordinates
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    if spin_pol:
        if choose_spin == 'up':
            file = open(dir + 'wf_k001_' + str(st) + '.dat', 'w')
        elif choose_spin == 'dn':
            file = open(dir + 'wf_k002_' + str(st) + '.dat', 'w')
    else:
        file = open(dir + 'wf_' + str(st) + '.dat', 'w')
    print('Writing wave function on the whole grid. (for debugging)')
    # Print out the grid coordinates and their corresponding electron density values
    for i in range(grid_counts[0]):
        for j in range(grid_counts[1]):
            for k in range(grid_counts[2]):
                wf = wf_grid[i, j, k]
                print(
                    f"{X[i, j, k]:.6f}    {Y[i, j, k]:.6f}    {Z[i, j, k]:.6f}    {wf:.6e}", file=file)
    print('Done.')

if Plot2D:
    if spin_pol:
        if choose_spin == 'up':
            fname = dir + 'wf_k001_' + str(st) + plane + '.pdf'
        elif choose_spin == 'dn':
            fname = dir + 'wf_k002_' + str(st) + plane + '.pdf'
    else:
        fname = dir + 'wf_' + str(st) + plane + '.pdf'
    print('Plotting wave function on the plane', plane, 'as', fname)
    fig, ax = plt.subplots(layout='constrained')
    if plane == 'xy':
        wf_2D = wf_grid[:, :, grid_counts[2]//2]
        cf = ax.contourf(x_coords, y_coords, wf_2D.T, levels=150, cmap='bwr')
        ax.set_xlabel(r'$x (a_0)$')
        ax.set_ylabel(r'$y (a_0)$')
    elif plane == 'yz':
        wf_2D = wf_grid[grid_counts[2]//2, :, :]
        cf = ax.contourf(y_coords, z_coords, wf_2D.T, levels=150, cmap='bwr')
        ax.set_xlabel(r'$y (a_0)$')
        ax.set_ylabel(r'$z (a_0)$')
    elif plane == 'xz':
        wf_2D = wf_grid[:, grid_counts[2]//2, :]
        cf = ax.contourf(x_coords, z_coords, wf_2D.T, levels=150, cmap='bwr')
        ax.set_xlabel(r'$x (a_0)$')
        ax.set_ylabel(r'$z (a_0)$')
    cbar = fig.colorbar(cf, ax=ax)
    fig.savefig(fname)
    print('Done.')

if Plot3D:
    plot_min_ang = -5.0
    plot_max_ang = 5.0
    x_plots = np.linspace(plot_min_ang / BOHR_TO_ANGSTROM,
                          plot_max_ang / BOHR_TO_ANGSTROM, 100)
    y_plots = np.linspace(plot_min_ang / BOHR_TO_ANGSTROM,
                          plot_max_ang / BOHR_TO_ANGSTROM, 100)
    z_plots = np.linspace(plot_min_ang / BOHR_TO_ANGSTROM,
                          plot_max_ang / BOHR_TO_ANGSTROM, 100)
    Xp, Yp, Zp = np.meshgrid(x_plots, y_plots, z_plots, indexing='ij')
    f = np.asarray(wf_interpol((Xp, Yp, Zp)))
    Xp_ang = Xp * BOHR_TO_ANGSTROM
    Yp_ang = Yp * BOHR_TO_ANGSTROM
    Zp_ang = Zp * BOHR_TO_ANGSTROM

    fig = go.Figure(data=go.Isosurface(
        x=Xp_ang.flatten(),
        y=Yp_ang.flatten(),
        z=Zp_ang.flatten(),
        value=f.flatten(),
        surface_fill=0.6,
        isomin=-iso_val,
        isomax=iso_val,
        surface_count=2, # number of isosurfaces, 2 by default: only min and max
        showscale=False,
        opacity=0.45,
        colorscale='Portland',
        caps=dict(x_show=False, y_show=False, z_show=False)
        ))
    fig.update_layout(
        autosize=False,
        minreducedwidth=100,
        minreducedheight=100,
        width=600,
        height=500,
        margin=dict(t=0, l=0, r=0, b=0),
        scene = dict(
            xaxis=dict(nticks=5, range=[plot_min_ang, plot_max_ang],
                       title=dict(text='x (Angstrom)', font=dict(size=24, family='Old Standard TT, serif'))),
            yaxis=dict(nticks=5, range=[plot_min_ang, plot_max_ang],
                       title=dict(text='y (Angstrom)', font=dict(size=24, family='Old Standard TT, serif'))),
            zaxis=dict(nticks=5, range=[plot_min_ang, plot_max_ang],
                       title=dict(text='z (Angstrom)', font=dict(size=24, family='Old Standard TT, serif')))),
        scene_camera_eye=dict(x=1.6, y=1.6, z=1.2),        
    )
    fname = dir + "orb" + str(st) + ".pdf"
    print('Saving orbital isosurfaces as', fname)
    fig.write_image(fname)
    print('Done.')
    # fig.show()


def read_xyz(filename):
    """Read atom symbols and Cartesian coordinates from an XYZ file."""
    with open(filename, 'r') as xyz_file:
        lines = [line.strip() for line in xyz_file if line.strip()]
    atom_count = int(lines[0])
    atom_lines = lines[2:2 + atom_count]
    atoms = []
    coordinates = []
    for line in atom_lines:
        values = line.split()
        if len(values) < 4:
            raise ValueError(f'Invalid XYZ atom line: {line}')
        atoms.append(values[0].capitalize())
        coordinates.append([float(values[1]), float(values[2]), float(values[3])])
    if len(atoms) != atom_count:
        raise ValueError(f'Expected {atom_count} atoms in {filename}, found {len(atoms)}')
    return atoms, np.asarray(coordinates, dtype=float)


if PlotMolecule:
    print('Reading molecular structure from', xyz_file_path)
    atoms, coordinates = read_xyz(xyz_file_path)
    radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    }
    colors = {
        'H': '#f5f5f5', 'C': '#3b3b3b', 'N': '#2f6fed', 'O': '#e53935',
        'F': '#55a630', 'P': '#f28c28', 'S': '#e1c542', 'Cl': '#55a630',
        'Br': '#8f2d56', 'I': '#6a4c93',
    }
    plot_coordinates = coordinates
    atom_sizes = {element: 16 * radius for element, radius in radii.items()}
    molecule_traces = []

    # Infer bonds from the sum of covalent radii with a modest tolerance.
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            bond_limit = 1.25 * (radii.get(atoms[i], 0.77) + radii.get(atoms[j], 0.77))
            if distance <= bond_limit:
                molecule_traces.append(go.Scatter3d(
                    x=[plot_coordinates[i, 0], plot_coordinates[j, 0]],
                    y=[plot_coordinates[i, 1], plot_coordinates[j, 1]],
                    z=[plot_coordinates[i, 2], plot_coordinates[j, 2]],
                    mode='lines',
                    line=dict(color='#777777', width=7),
                    hoverinfo='skip',
                    showlegend=False,
                ))

    molecule_traces.append(go.Scatter3d(
        x=plot_coordinates[:, 0],
        y=plot_coordinates[:, 1],
        z=plot_coordinates[:, 2],
        mode='markers+text',
        text=atoms,
        textposition='top center',
        hovertemplate='%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
        marker=dict(
            size=[atom_sizes.get(element, 12) for element in atoms],
            color=[colors.get(element, '#bdbdbd') for element in atoms],
            line=dict(color='#222222', width=1),
            opacity=0.98,
        ),
        showlegend=False,
    ))

    if Plot3D:
        molecule_traces.insert(0, fig.data[0])
    molecule_fig = go.Figure(data=molecule_traces)
    axis_labels = ('x (Angstrom)', 'y (Angstrom)', 'z (Angstrom)')
    molecule_fig.update_layout(
        title='Camphor at maximum TI rate' + (' with orbital isosurface' if Plot3D else ''),
        width=850,
        height=700,
        margin=dict(t=45, l=0, r=0, b=0),
        scene=dict(
            xaxis=dict(title=axis_labels[0],
                       range=[plot_min_ang, plot_max_ang] if Plot3D else None),
            yaxis=dict(title=axis_labels[1],
                       range=[plot_min_ang, plot_max_ang] if Plot3D else None),
            zaxis=dict(title=axis_labels[2],
                       range=[plot_min_ang, plot_max_ang] if Plot3D else None),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
    )
    if args.molout:
        molecule_file_path = args.molout
    else:
        molecule_file_path = xyz_file_path.rsplit('.', 1)[0] + '_ball_stick.html'
    print('Saving molecular ball-and-stick model as', molecule_file_path)
    molecule_fig.write_html(molecule_file_path, include_plotlyjs=True)
    print('Done.')

def wf_sph(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return wf_interpol((x, y, z))


def Y_lm(l, m, theta, phi):
    return sph_harm_y(l, m, theta, phi)


# Define spherical grid
dtheta = 0.01*np.pi
dphi = 0.01*np.pi

theta = np.arange(0, np.pi+dtheta, dtheta)
phi = np.arange(0., 2.*np.pi+dphi, dphi)

Ntheta = np.size(theta)
Nphi = np.size(phi)

theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')

if spin_pol:
    if choose_spin == 'up':
        kappa = np.sqrt(2.*np.abs(Ip_list[0][st-1]))
    elif choose_spin == 'dn':
        kappa = np.sqrt(2.*np.abs(Ip_list[1][st-1]))
else:
    kappa = np.sqrt(2.*np.abs(Ip_list[st-1]))


def asym(Z, kappa, r):
    return r**(Z/kappa - 1.)*np.exp(-kappa*r)


def f_solve(r):
    return asym(Z, kappa, r) - match_tol
#-------------------------------------------------------------------------------
#graphs f_solve
plt.plot(np.linspace(8., r_max, 50) , f_solve(np.linspace(8., r_max, 50)))
plt.ylim(-0.001, .003)
plt.axhline(y=0, color='red' , linewidth=0.7)
plt.title("F_solve vs Bracket")
plt.savefig("f_solve_plot.pdf")
#--------------------------------------------------------------------------------

sol = root_scalar(f_solve, bracket=[5., r_max], method='brentq')
r_m = sol.root
print(sol)
print('Ionization potential (eV) of the orbital is',
      f"{kappa**2/2.*27.211:.3f}")
print('Computing wave function on the angular gird at R =', f"{r_m:.3f}")

psi = wf_sph(r_m, theta_m, phi_m)/asym(Z, kappa, r_m)
print('Done.')

fig, ax = plt.subplots(layout='constrained')
norm = mpl.colors.CenteredNorm(vcenter=0)
cf = ax.contourf(theta/np.pi*180, phi/np.pi*180, psi.T, norm=norm, levels=100, cmap='bwr')
cbar = fig.colorbar(cf, ax=ax)
if spin_pol:
    if choose_spin == 'up':
        fname = dir + 'wf_k001_' + str(st) + '_ang_dist.pdf'
    elif choose_spin == 'dn':
        fname = dir + 'wf_k002_' + str(st) + '_ang_dist.pdf'
else:
    fname = dir + 'wf_' + str(st) + '_ang_dist.pdf'
xtick = np.arange(0, 181, 30, dtype=int)
ax.set_xticks(xtick)
ax.set_xticklabels(xtick)
ax.set_xlabel(r'$\theta$ (degree)')
ytick = np.arange(0, 361, 60, dtype=int)
ax.set_yticks(ytick)
ax.set_yticklabels(ytick)
ax.set_ylabel(r'$\phi$ (degree)')
print('Saving the angular distribution of wave function as', fname)
fig.savefig(fname)


test_fn = np.zeros((np.shape(theta_m)), dtype=complex)

if spin_pol:
    if choose_spin == 'up':
        fname = dir + 'Clm_k001_' + str(st) + '.dat'
    elif choose_spin == 'dn':
        fname = dir + 'Clm_k002_' + str(st) + '.dat'
else:
    fname = dir + 'Clm_' + str(st) + '.dat'

file = open(fname, 'w')
C_lm = []
l_whole = []
m_whole = []
for l in range(0, l_max+1):
    C_l = []
    m_set = []
    for m in range(-l, l+1):
        int_00 = psi * \
            np.conjugate(Y_lm(l=l, m=m, theta=theta_m,
                              phi=phi_m))*np.sin(theta_m)
        tmp = np.trapz(int_00, phi, axis=1)
        tmp1 = np.trapz(tmp, theta)
        if np.abs(tmp1) > 1.e-2:
            m_set.append(m)
            tmp1 = np.round(tmp1, decimals=3)
            C_l.append(tmp1)
            test_fn += tmp1*Y_lm(l=l, m=m, theta=theta_m, phi=phi_m)
    if len(m_set) > 0:
        l_whole.append(l)
        C_lm.append(C_l)
        m_whole.append(m_set)


# Find the maximum abs of C_lm
Clm_max = 0
for i in range(0, len(C_lm)):
    tmp = np.abs(C_lm[i])
    if tmp.max() - Clm_max > 1.e-2:
        Clm_max = tmp.max()
        ind_i = i   # First max indice of l in C_lm
        ind_j = np.argmax(tmp)  # Second max indice of m in C_lm

# Fix the phase of the wave function
phi1 = np.angle(C_lm[ind_i][ind_j])
for i in range(0, len(C_lm)):
    for j in range(0, len(C_lm[i])):
        C_lm[i][j] *= np.exp(-1.j*phi1)
        C_lm[i][j] = np.round(C_lm[i][j], decimals=2)

print(l_whole, file=file)
print(m_whole, file=file)
print(C_lm, file=file)
print('Structure parameters are saved as', fname)

fig1, ax1 = plt.subplots(layout='constrained')
norm = mpl.colors.CenteredNorm(vcenter=0)
cf1 = ax1.contourf(theta/np.pi*180, phi/np.pi*180,
                   test_fn.T.real, levels=100, norm=norm, cmap='bwr')
cbar = fig1.colorbar(cf1, ax=ax1)
xtick = np.arange(0, 181, 30, dtype=int)
ax1.set_xticks(xtick)
ax1.set_xticklabels(xtick)
ax1.set_xlabel(r'$\theta$ (degree)')
ytick = np.arange(0, 361, 60, dtype=int)
ax1.set_yticks(ytick)
ax1.set_yticklabels(ytick)
ax1.set_ylabel(r'$\phi$ (degree)')
fname = dir + 'test_' + str(st) + '_fn.pdf'
print('Saving the angular distribution of test function as', fname)
fig1.savefig(fname)
