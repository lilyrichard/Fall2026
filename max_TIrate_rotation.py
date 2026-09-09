import numpy as np

Target = 'fenchone_cat'
inp =  Target + '_eq.xyz'

# ----------------------------------------------------------------------------
#             Read XYZ file
# ----------------------------------------------------------------------------
def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    comment = lines[1].strip()
    atom = []
    target_cart = []
    for i in range(2, 2 + natom):
        parts = lines[i].strip().split()
        atom.append(parts[0])
        target_cart.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return natom, atom, target_cart

natom, atom, target_cart = read_xyz(inp)

# ----------------------------------------------------------------------------
#             Rotation matrices (z-y-z)
# ----------------------------------------------------------------------------
def R_z(alpha):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(alpha)
    R[0, 1] = -np.sin(alpha)
    R[1, 0] = np.sin(alpha)
    R[1, 1] = np.cos(alpha)
    R[2, 2] = 1.
    return R


def R_y(beta):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(beta)
    R[2, 2] = np.cos(beta)
    R[0, 2] = np.sin(beta)
    R[2, 0] = -np.sin(beta)
    R[1, 1] = 1.
    return R


# ----------------------------------------------------------------------------
#             Find max TI rate and rotate structure
# ----------------------------------------------------------------------------
target_w_r = np.load(Target + '_DCS/TI_rate_R_' + Target + '.npy')

da = 0.04*np.pi
beta = np.arange(0., np.pi + da, da)
gamma = np.arange(0., 2.*np.pi + da, da)

max_idx = np.unravel_index(np.argmax(target_w_r), target_w_r.shape)
max_beta = beta[max_idx[0]]
max_gamma = gamma[max_idx[1]]

print('Max TI rate index:', max_idx)
print('Max TI rate beta (rad):', max_beta, ', beta (deg):', np.rad2deg(max_beta))
print('Max TI rate gamma (rad):', max_gamma, ', gamma (deg):', np.rad2deg(max_gamma))

target_cart_rot = [row.copy() for row in target_cart]
for i in range(len(atom)):
    target_cart_rot[i][:] = R_z(0.0) @ R_y(max_beta) @ R_z(max_gamma) @ target_cart_rot[i][:]

out_file = Target + '_maxTIR_rotation.xyz'
with open(out_file, 'w') as f:
    f.write(str(natom) + '\n')
    f.write('Rotated at max TI rate angles: alpha=0.0 deg, beta=' + str(np.rad2deg(max_beta)) + ' deg, gamma=' + str(np.rad2deg(max_gamma)) + ' deg\n')
    for i in range(natom):
        f.write(atom[i] + '  ' + str(target_cart_rot[i][0]) + '  ' + str(target_cart_rot[i][1]) + '  ' + str(target_cart_rot[i][2]) + '\n')

print('Saved rotated structure to', out_file)
