# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2018-03-23: Created by Mathew J. Bryant, The Cambridge Crystallographic Data Centre
#
"""
    mechanical_properties.py   -   Calculate topological descriptors for mechanical properties prediction
    Run from the command-line
"""

from ccdc.io import CrystalReader
from ccdc._lib import MathsLib
from ccdc.descriptors import GeometricDescriptors, MolecularDescriptors
from collections import OrderedDict
import numpy as np
import math as m
import itertools
import argparse
import re


def euclidean(p1, p2):
    """Euclidean distance for coordinates"""
    return m.sqrt((abs(p1[0] - p2[0]) ** 2) + (abs(p1[1] - p2[1]) ** 2) + (abs(p1[2] - p2[2]) ** 2))


def vector_euclidean(p1, p2):
    """Euclidean distance for vector object"""
    return np.sqrt((abs(p1.x() - p2.x()) ** 2) + (abs(p1.y() - p2.y()) ** 2) + (abs(p1.z() - p2.z()) ** 2))


def plane_order(red, crystal):
    """
    Assigns the correct periodicity to the plane
    - ie. 101, 202, 303 or 404 based on its periodicity across the unit cell, not just relative to the origin position
    This is based on generating atom slabs for multiples of the root index (eg. for 100:  200, 300, 400)
    and noting where identical slabs are generated. If for example 100 gives the same slab as 200, but not 400,
    then the correct periodicity of the slip plane is 200
    """

    best_order = 1
    pop = {}
    typ = {}
    h, k, l = red
    for n in range(1, 5):
        try:
            typ[n] = len([atom for atom in calc_slab(n * h, n * k, n * l,
                                                     0, 30, 1.8, crystal, 'only')[0].atoms
                          if atom.atomic_symbol == 'C'])
            pop[n] = len([atom for atom in calc_slab(n * h, n * k, n * l,
                                                     0, 30, 1.8, crystal, 'only')[0].atoms])
        except ValueError:
            typ[n] = 0
            pop[n] = 0

    if abs(pop[1] - pop[2]) < 4 and abs(typ[1] - typ[2]) < 4:
        if abs(pop[2] - pop[4]) < 4 and abs(typ[2] - typ[4]) < 4:
            best_order = 4
        else:
            best_order = 2
    elif abs(pop[2] - pop[4]) < 4 and abs(typ[2] - typ[4]) < 4:
        best_order = 1
    elif abs(pop[1] - pop[4]) < 3 and abs(typ[1] - typ[4]) < 3:
        best_order = 3
    elif abs(pop[1] - pop[4]) < 4 and abs(typ[1] - typ[4]) < 4:
        best_order = 4
    return [best_order * h, best_order * k, best_order * l]


def reduced(index):
    """Returns most reduced miller plane. eg. [222] = [111], [102] = [102], [224] = [112]"""

    signs = [np.sign(i) for i in index]
    index = [abs(ix) for ix in index]
    hkl = np.array(index)
    mask = sorted(hkl[np.nonzero(hkl)])
    for x in mask:
        red = [float(z) / x for z in index]
        if any(not num.is_integer() for num in red):
            continue
        else:
            return [y * int(z) for y, z in zip(signs, red)]
    return [y * int(z) for y, z in zip(signs, index)]


def slip_order(planes_list):
    """Remove duplicates from slip plane list"""
    planes = OrderedDict()
    for index in planes_list:
        if str(reduced(index[0])) not in planes:
            planes[str(reduced(index[0]))] = [reduced(index[0]), index[1]]
        else:
            if planes[str(reduced(index[0]))][1] < index[1]:
                planes[str(reduced(index[0]))] = [reduced(index[0]), index[1]]
    return sorted([planes[p] for p in planes], key=lambda x: x[1])


def slab_divider(plane, atoms1, g1, atoms2, g2):
    """ Calculates the exact plane that divides the two slabs"""
    a, b, c, = plane.normal
    #  Determine the position of the slabs from an arbitrary point at 100A along the vector (stack direction)
    distance1 = a * g1[0] + b * g1[1] + c * g1[2] + 100
    distance2 = a * g2[0] + b * g2[1] + c * g2[2] + 100

    if distance1 < distance2:  # If slab1 is the nearest slab to the arbitrary point
        frontier1 = \
            sorted([[point, point_to_plane(plane, point, 100)] for point in atoms1], key=lambda x: x[1])[0]
        frontier2 = \
            sorted([[point, point_to_plane(plane, point, -100)] for point in atoms2], key=lambda x: x[1])[0]
    else:
        frontier1 = \
            sorted([[point, point_to_plane(plane, point, -100)] for point in atoms1], key=lambda x: x[1])[-1]
        frontier2 = \
            sorted([[point, point_to_plane(plane, point, 100)] for point in atoms2], key=lambda x: x[1])[-1]

    division_plane = [(a + b) / 2 for a, b in zip(frontier1[0], frontier2[0])]
    return division_plane


def point_to_plane(plane, point, divide):
    """Calculate the perpendicular distance of a point to a plane."""

    intercept_x = point[0] - ((((point[0] * (plane.normal[0])) + (point[1] * (plane.normal[1]))
                                + (point[2] * (plane.normal[2])) - divide) /
                               ((plane.normal[0]) ** 2 + (plane.normal[1]) ** 2 +
                                (plane.normal[2]) ** 2)) * (plane.normal[0]))
    intercept_y = point[1] - ((((point[0] * (plane.normal[0])) + (point[1] * (plane.normal[1])) +
                                (point[2] * (plane.normal[2]) - divide)) /
                               ((plane.normal[0]) ** 2 + (plane.normal[1]) ** 2 +
                                (plane.normal[2]) ** 2)) * (plane.normal[1]))
    intercept_z = point[2] - ((((point[0] * (plane.normal[0])) + (point[1] * (plane.normal[1])) +
                                (point[2] * (plane.normal[2]) - divide)) /
                               ((plane.normal[0]) ** 2 + (plane.normal[1]) ** 2 +
                                (plane.normal[2]) ** 2)) * (plane.normal[2]))

    overlap = euclidean(point, [intercept_x, intercept_y, intercept_z])
    return overlap


def reciprocal_lattice(vector_a, vector_b, vector_c):
    """Calculate the reciprocal lattice"""
    reciprocal_a = np.cross(vector_b, vector_c) / np.dot(vector_a, np.cross(vector_b, vector_c))
    reciprocal_b = np.cross(vector_c, vector_a) / np.dot(vector_a, np.cross(vector_b, vector_c))
    reciprocal_c = np.cross(vector_a, vector_b) / np.dot(vector_a, np.cross(vector_b, vector_c))
    return reciprocal_a, reciprocal_b, reciprocal_c


def generate_cart_matrix(a, b, c, alpha, beta, gamma):
    """Create matrix to convert fractional to cartesian"""
    v = (m.sqrt(1 - (m.pow(m.cos(m.radians(alpha)), 2)) -
                (m.pow(m.cos(m.radians(beta)), 2)) - (m.pow(m.cos(m.radians(gamma)), 2))
                + 2 * (m.cos(m.radians(alpha))) * (m.cos(m.radians(beta))) * (m.cos(m.radians(gamma)))))
    xx = a
    xy = (b * (m.cos(m.radians(gamma))))
    xz = (c * (m.cos(m.radians(beta))))
    yx = 0
    yy = (b * (m.sin(m.radians(gamma))))
    yz = (c * (((m.cos(m.radians(alpha))) - ((m.cos(m.radians(beta))) *
                                             (m.cos(m.radians(gamma))))) /
               (m.sin(m.radians(gamma)))))
    zx = 0
    zy = 0
    zz = (c * (v / (m.sin(m.radians(gamma)))))
    matrix = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])
    matrix = np.reshape(matrix, (3, 3))
    return matrix


def converter(c, matrix):
    """Convert fractional to cartesian (generally use generate cart matrix)"""
    x, y, z = (a for a in c)
    frac_coord = np.array([x, y, z])
    frac_coord = np.reshape(frac_coord, (3, 1))
    frac_coords = np.array(np.dot(matrix, frac_coord))
    xx = frac_coords[0][0]
    yy = frac_coords[1][0]
    zz = frac_coords[2][0]
    return xx, yy, zz


def fract_to_cartesian(cell_parameters, hkl):
    """Convert fractional coordinates to cartesian"""
    a, b, c, alpha, beta, gamma = cell_parameters
    matrix = generate_cart_matrix(a, b, c, alpha, beta, gamma)
    h, k, l = hkl
    x, y, z = converter(hkl, matrix)
    xx = np.array([x, y, z]) * h
    yy = np.array([x, y, z]) * k
    zz = np.array([x, y, z]) * l
    return xx + yy + zz


def get_vector(cell_params, hkl):
    """Converts face data to cartesian growth directions"""
    (vector_a, vector_b, vector_c) = (fract_to_cartesian(cell_params, np.array([1, 0, 0])),
                                      fract_to_cartesian(cell_params, np.array([0, 1, 0])),
                                      fract_to_cartesian(cell_params, np.array([0, 0, 1])))

    reciprocal_a, reciprocal_b, reciprocal_c = reciprocal_lattice(vector_a, vector_b, vector_c)

    xyz = np.array(hkl)[0] * reciprocal_a + np.array(hkl)[1] * reciprocal_b + np.array(hkl)[2] * reciprocal_c
    return xyz * -100


def calc_slab(h, k, l, disp, wth, thk, crystal, *args):
    """Calculate slab of atoms representing the miller plane given"""
    slab = crystal.slicing(crystal.miller_indices(int(h), int(k), int(l)).plane,
                           width=wth, displacement=disp, thickness=thk, inclusion=args[0])
    return slab, slab.centre_of_geometry()


def get_direction(plane, point, divide):
    """
    Determines the half-space (for the given plane) a point atom occupies relative to our assigned plane order
     +ve values indicate that the atom is in the same half-space as its parent slab
    """
    x, y, z = point
    a, b, c, = plane.normal
    d = point_to_plane(plane, divide, 0)
    direction = a * x + b * y + c * z - d
    return direction, d


def interdigitation(plane, points, divide):
    """Calculates the position of atoms relative to the dividing plane"""
    overlap = []
    for point in points:
        direction, d = get_direction(plane, point, divide)
        if direction < 0:  # if negative values, the atom is overlapping the next slab
            magnitude = point_to_plane(plane, point, d)
            overlap.append(-1 * magnitude)
        else:
            if not any(x < 0 for x in overlap):
                magnitude = point_to_plane(plane, point, d)
                overlap.append(magnitude)
    return sorted(overlap, key=lambda xx: xx)


def plane_angles(hkl1, hkl2, crystal):
    """Calculate dihedral angle between two planes"""
    plane1 = crystal.miller_indices(int(hkl1[0]), int(hkl1[1]), int(hkl1[2])).plane
    plane2 = crystal.miller_indices(int(hkl2[0]), int(hkl2[1]), int(hkl2[2])).plane
    return plane1.plane_angle(plane2)


def measure_angle(a, b, c):
    """Calculates angle between three atoms (in order given)"""
    v1 = np.array([p1 - p2 for p1, p2 in zip(b, a)])  # calculate vectors
    v2 = np.array([p1 - p2 for p1, p2 in zip(b, c)])
    u1 = v1 / np.linalg.norm(v1)  # Unit vectors
    u2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))


def calculate_slabs(h, k, l, crystal):
    """Calculate the bottom and top slabs for the given plane"""
    slab1, g1 = calc_slab(h, k, l, -4.8, 30, 10, crystal, 'centroid')
    slab2, g2 = calc_slab(h, k, l, 5.01, 30, 10, crystal, 'centroid')
    slab1atoms = [[c for c in a.coordinates] for a in slab1.atoms]
    slab2atoms = [[c for c in a.coordinates] for a in slab2.atoms]
    for component in slab2.components:
        if any([a for a in atom.coordinates] in slab1atoms for atom in component.atoms):
            for atom in component.atoms:
                if [a for a in atom.coordinates] in slab2atoms:
                    slab2atoms.remove([a for a in atom.coordinates])
    return slab1atoms, g1, slab2atoms, g2, slab1, slab2


def get_overlap(h, k, l, slab1atoms, g1, slab2atoms, g2, crystal):
    """Calculates interdigitation between slabs"""
    plane = crystal.miller_indices(int(h), int(k), int(l)).plane  # Generate plane object for the Miller plane
    divide = slab_divider(plane, slab1atoms, g1, slab2atoms, g2)  # Correctly position plane at the interface
    overlap = interdigitation(plane, slab2atoms, divide)[0]  # Measure maximum atomic displacement past interface
    return [(h, k, l), round(overlap, 2)]


def filter_and_run(slices, crystal):
    """Filters out duplicate slice interfaces and runs the analysis using the "get_overlap" function"""
    corrugation = []
    for hkl in slices:
        h, k, l = hkl
        if not (-1 * h, -1 * k, -1 * l) in [x[0] for x in corrugation] and [h, k, l] != [0, 0, 0]:
            s1, g1, s2, g2, sl1, sl2, = calculate_slabs(h, k, l, crystal)
            corrugation.append(get_overlap(h, k, l, s1, g1, s2, g2, crystal))
    return sorted(corrugation, key=lambda xx: xx[1])


def obb(atoms):
    """Minimum volume oriented bounding box"""
    all_pts = np.array([[c for c in atom.coordinates] for atom in atoms])
    hull = {
        (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        for p in all_pts
    }
    bbox = MathsLib.MinimumVolumeBox(
        [MathsLib.Point(c[0], c[1], c[2]) for c in hull]
    )

    corners = bbox.corners()
    lengths = [
        vector_euclidean(corners[0], corners[1]),
        vector_euclidean(corners[0], corners[2]),
        vector_euclidean(corners[0], corners[4])
    ]
    lengths.sort()
    return corners, lengths


def detect_interslab_hbonds(slab1, slab2):
    """Detects whether hydrogen bonds exist between the two stacked layers"""
    atom_search = MolecularDescriptors.AtomDistanceSearch(slab1)
    bonds = []
    for atom in [a for a in slab2.atoms]:
        if atom.is_acceptor:
            prox = atom_search.atoms_within_range(atom.coordinates, 0.2 + (atom.vdw_radius * 2))
            dons = [d for d in prox if d.is_donor]
            for don in dons:
                for n in don.neighbours:
                    if n.atomic_symbol == 'H':
                        if measure_angle(don.coordinates, n.coordinates, atom.coordinates) > 120:
                            bonds.append([don, atom])
        if atom.is_donor:
            prox = atom_search.atoms_within_range(atom.coordinates, 0.2 + (atom.vdw_radius * 2))
            accs = [a for a in prox if a.is_acceptor]
            for acc in accs:
                for n in atom.neighbours:
                    if n.atomic_symbol == 'H':
                        if measure_angle(atom.coordinates, n.coordinates, acc.coordinates) > 120:
                            bonds.append([atom, acc])
    if len(bonds) > 0:
        return True
    else:
        return False


def hbonds_between_layers(crystal, h, k, l):
    """This function analyses the Hydrogen bonding of the crystal"""

    top, c1, bottom, c2, sl1, sl2 = calculate_slabs(h, k, l, crystal)  # Generate the slab to analyse for bridging
    if not crystal.hbonds():  # If there are no hydrogen bonds at all
        return False, 'No Hydrogen Bonds'

    # Generate two shells of molecules expanded to different degrees by Hydrogen bonding contacts only
    shells = [
        crystal.hbond_network(repetitions=3), crystal.hbond_network(repetitions=5)
    ]

    # Remove hydrogens without atomic coordinates
    for s in shells:
        if not s.all_atoms_have_sites:
            s.remove_hydrogens()

    # Calculate a minimum-volume oriented bounding box around each of the two shells
    corners1, lengths1 = obb(shells[0].atoms)
    s1, m1, l1 = lengths1

    corners2, lengths2 = obb(shells[1].atoms)
    s2, m2, l2 = lengths2

    # Now we have everything we need to calculate the two descriptors (dimensionalisty and bridging)

    # Work out the dimensionalisty from the change in aspect ratio of the bounding boxes
    ratios = [s2 / s1, m2 / m1, l2 / l1]
    thresh = 1.1
    ndims = sum(r > thresh for r in ratios)

    bound = detect_interslab_hbonds(sl1, sl2)  # Binary bridging hydrogen bonds descriptor
    dimens = dimensionality(ndims)  # Overall dimensionality descriptor
    return bound, dimens


def dimensionality(fig):
    """Takes the dimensionality figure and generates the accompanying string"""
    if fig == 0:
        return 'Ring/enclosed'
    elif fig == 1:
        return 'Chain (1D)'
    elif fig == 2:
        return 'Sheet (2D)'
    elif fig == 3:
        return 'Lattice (3D)'


def generate_descriptors(corrugation, crystal):
    """Take the predicted flattest plane from the list "corrugation" and generate the subsequent descriptors"""
    slip_planes = [plane for plane in corrugation if plane[1] > 0]  # List of all separated planes
    # Are any additional planes found at >45 degrees to the most separated plane?
    perpendicular = len([sp for sp in slip_planes[:-1] if 45 < plane_angles(sp[0], slip_planes[-1][0], crystal) < 135])
    # Take predicted slip plane, and assign the correct periodicity (origin independant index)
    final_plane = plane_order(corrugation[-1][0], crystal)
    final_separation = corrugation[-1][1]  # The separation/interdigitation decriptor
    spacing = crystal.miller_indices(int(final_plane[0]),
                                     int(final_plane[1]),
                                     int(final_plane[2])).plane.distance  # the d-spacing descriptor

    hbond_dims = None
    hbound = None

    # Run Hbonding analysis (see hbonds_between_layers function), and print out the resulting descriptors
    if final_separation < 0:
        hbound, hbond_dims = hbonds_between_layers(crystal, *final_plane)  # Calculate Hbond descriptors
        print('No unobstructed slip plane, Most probable slip-plane: {} - interlocked by  = {} '.format(
            final_plane, final_separation * 2))

        if hbound:
            print('Layers are bound together with hydrogen bonds')
        else:
            print('No hydrogen bonding between layers')

    else:
        for n, slc in enumerate(slip_planes):
            hbound, hbond_dims = hbonds_between_layers(crystal, *slc[0])  # Calculate Hbond descriptors
            slip_planes[n].append(hbound)

        if perpendicular > 0:
            print('Slip plane arrangement likely to result in highly plastic or bendy crystal')
            print('Most probable primary slip-plane: {} -  flat layer with ~{} spacing'.format(
                final_plane, final_separation * 2))
            if hbound:
                print('Layers are bound together with hydrogen bonds')
                if [s for s in slip_planes if not s[-1]]:
                    print('Alternative flat planes without hydrogen bonding are present:')
                    for s in [s for s in slip_planes if not s[-1]]:
                        print('     Plane: {}    Separation: {}'.format(s[0], s[1]))
            else:
                print('No hydrogen bonding between layers')

        elif len(slip_planes) > 0:
            if len(slip_planes) > 1:
                print('Multiple slip planes')
            print('Most probable slip-plane: {} -  flat layer with ~{} spacing'.format(
                final_plane, final_separation * 2))
            if hbound:
                print('Layers are bound together with hydrogen bonds')
                if [s for s in slip_planes if not s[-1]]:
                    print('Alternative flat planes without hydrogen bonding are present:')
                    for s in [s for s in slip_planes if not s[-1]]:
                        print('     Plane: {}    Separation: {}'.format(s[0], s[1]))
            else:
                print('No hydrogen bonding between layers')

    print('d-Spacing = {}'.format(spacing))
    print('Overall hydrogen bonding network: {}'.format(hbond_dims))


def calc_slip_planes(ipt, precision, extend, suppress, filetype):
    """Import crystal object from either CSD or from file provided"""
    if filetype == 'refcode':
        reader = CrystalReader('CSD')
        crystal = reader.crystal(ipt.upper())
    else:
        crystal = CrystalReader(ipt)[0]
    
    if not suppress:
        mol = crystal.molecule
        mol.assign_bond_types()
        mol.add_hydrogens(mode='missing')
        crystal.molecule = mol
        print("Added missing hydrogen atoms and assigned bond types")

    # Generate a list if miller planes to iterate over.
    if precision == 'h':  # If the precision has been manually set to high, scan to 444 from the beginning
        slices = itertools.product([4, 3, 2, 1, 0, -1, -2, -3, -4], repeat=3)
    else:  # If not running high-precision, planes between -2-2-2 and 222 are generated
        slices = itertools.product([2, 1, 0, -1, -2], repeat=3)

    # Run the analysis on the list of slip planes
    corrugation = filter_and_run(slices, crystal)  # generate a list of slip planes, sorted by their separation

    # Check for the presence of separated planes in the list "corrugation"
    # If no slip planes are separated, the additional scan isn't suppressed in the input arguments, and the
    # scan to 444 hasn't yet been done, run to 444
    if corrugation[-1][1] <= 0 and extend and not precision == 'h':
        print('No clear slip planes found between miller sets -2-2-2 and 222, performing more extensive search')
        slices = [x for x in itertools.product([4, 3, 2, 1, 0, -1, -2, -3, -4], repeat=3)
                  if x not in itertools.product([2, 1, 0, -1, -2], repeat=3)]
        corrugation2 = filter_and_run(slices, crystal)
        for dat in corrugation2:
            corrugation.append(dat)  # Add additional data to the planes list

    corrugation = slip_order(corrugation)  # Strip out duplicate planes with the same spacing
    generate_descriptors(corrugation, crystal)  # Final plane analysis and generate the descriptors


if __name__ == '__main__':
    """
    By default the tool will scan for "flat" planes between miller planes [-2 -2 -2] and [2 2 2].
    If none are found, it will extend the search from [-4 -4 -4] to [4 4 4]. This range can be explicitly set,
    or the additional scan on failure suppressed.
    """
    parser = argparse.ArgumentParser(description='''Enter refcode or cif file''')
    parser.add_argument("entry", help="Provide CSD refcode or path to file")
    parser.add_argument("--e", help="suppress structure edit", default=False, action="store_true")  # Suppress normalisation of bond types and addition of missing hydrogens
    parser.add_argument("--p", help="precision (high/low)")  # Explicitly sets the miller plane search to [222] or [444]
    parser.add_argument("--s", help="suppress further search- 't' or 'true'")  # Suppress run to [444]
    inputs = parser.parse_args()

    input_components = re.split(r'\b[./]\b|\b[.\\]\b', inputs.entry)  # Is the input a file or CSD refcode
    if len(input_components) > 1:
        ext = input_components[-1]
        filename = input_components[-2]
        if ext == 'cif':
            print('Reading {} file {}.{}'.format(ext, filename, ext))
        else:
            print('Provide a .cif file (include extension)')
            ext = None
            exit()

    else:
        filename = input_components[0]
        filename = filename.upper()
        print('Reading CSD entry {}'.format(filename))
        rdr = EntryReader('CSD')
        if rdr.entry(filename):
            ext = 'refcode'
        else:
            ext = None
            exit()

    if inputs.s:
        if inputs.s.lower() == 't' or inputs.s.lower() == 'true':
            inputs.s = False
        else:
            print('command --s: {} not recognised, expected "t" or "true"'.format(inputs.p))
            print('Extended search will be performed if necessary')
            inputs.s = True
    else:
        inputs.s = True

    if inputs.p:
        if inputs.p.lower() == 'l' or inputs.p.lower() == 'low':
            inputs.p = 'l'
        elif inputs.p.lower() == 'h' or inputs.p.lower() == 'high':
            inputs.p = 'h'
        else:
            print('command --p: {} not recognised, precision set to low'.format(inputs.p))
            inputs.p = 'l'
    else:
        inputs.p = 'l'

    calc_slip_planes(inputs.entry, inputs.p, inputs.s, inputs.e, ext)  # Run main program
