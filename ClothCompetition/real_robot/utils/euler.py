""" Generic Euler rotations

See:

* http://en.wikipedia.org/wiki/Rotation_matrix
* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html

See also: *Representing Attitude with Euler Angles and Quaternions: A
Reference* (2006) by James Diebel. A cached PDF link last found here:

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134

******************
Defining rotations
******************

Euler's rotation theorem tells us that any rotation in 3D can be described by 3
angles.  Let's call the 3 angles the *Euler angle vector* and call the angles
in the vector :math:`alpha`, :math:`beta` and :math:`gamma`.  The vector is [
:math:`alpha`, :math:`beta`. :math:`gamma` ] and, in this description, the
order of the parameters specifies the order in which the rotations occur (so
the rotation corresponding to :math:`alpha` is applied first).

In order to specify the meaning of an *Euler angle vector* we need to specify
the axes around which each of the rotations corresponding to :math:`alpha`,
:math:`beta` and :math:`gamma` will occur.

There are therefore three axes for the rotations :math:`alpha`, :math:`beta`
and :math:`gamma`; let's call them :math:`i` :math:`j`, :math:`k`.

Let us express the rotation :math:`alpha` around axis `i` as a 3 by 3 rotation
matrix `A`.  Similarly :math:`beta` around `j` becomes 3 x 3 matrix `B` and
:math:`gamma` around `k` becomes matrix `G`.  Then the whole rotation expressed
by the Euler angle vector [ :math:`alpha`, :math:`beta`. :math:`gamma` ], `R`
is given by::

   R = np.dot(G, np.dot(B, A))

See http://mathworld.wolfram.com/EulerAngles.html

The order :math:`G B A` expresses the fact that the rotations are
performed in the order of the vector (:math:`alpha` around axis `i` =
`A` first).

To convert a given Euler angle vector to a meaningful rotation, and a
rotation matrix, we need to define:

* the axes `i`, `j`, `k`;
* whether the rotations move the axes as they are applied (intrinsic
  rotations) - compared the situation where the axes stay fixed and the
  vectors move within the axis frame (extrinsic);
* whether a rotation matrix should be applied on the left of a vector to
  be transformed (vectors are column vectors) or on the right (vectors
  are row vectors);
* the handedness of the coordinate system.

See: http://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

This module implements intrinsic and extrinsic axes, with standard conventions
for axes `i`, `j`, `k`.  We assume that the matrix should be applied on the
left of the vector, and right-handed coordinate systems.  To get the matrix to
apply on the right of the vector, you need the transpose of the matrix we
supply here, by the matrix transpose rule: $(M . V)^T = V^T M^T$.

*************
Rotation axes
*************

Rotations given as a set of three angles can refer to any of 24 different ways
of applying these rotations, or equivalently, 24 conventions for rotation
angles.  See http://en.wikipedia.org/wiki/Euler_angles.

The different conventions break down into two groups of 12.  In the first
group, the rotation axes are fixed (also, global, static), and do not move with
rotations.  These are called *extrinsic* axes.  The axes can also move with the
rotations.  These are called *intrinsic*, local or rotating axes.

Each of the two groups (*intrinsic* and *extrinsic*) can further be divided
into so-called Euler rotations (rotation about one axis, then a second and then
the first again), and Tait-Bryan angles (rotations about all three axes).  The
two groups (Euler rotations and Tait-Bryan rotations) each have 6 possible
choices.  There are therefore 2 * 2 * 6 = 24 possible conventions that could
apply to rotations about a sequence of three given angles.

This module gives an implementation of conversion between angles and rotation
matrices for which you can specify any of the 24 different conventions.

****************************
Specifying angle conventions
****************************

You specify conventions for interpreting the sequence of angles with a four
character string.

The first character is 'r' (rotating == intrinsic), or 's' (static ==
extrinsic).

The next three characters give the axis ('x', 'y' or 'z') about which to
perform the rotation, in the order in which the rotations will be performed.

For example the string 'szyx' specifies that the angles should be interpreted
relative to extrinsic (static) coordinate axes, and be performed in the order:
rotation about z axis; rotation about y axis; rotation about x axis. This
is a relatively common convention, with customized implementations in
:mod:`taitbryan` in this package.

The string 'rzxz' specifies that the angles should be interpreted
relative to intrinsic (rotating) coordinate axes, and be performed in the
order: rotation about z axis; rotation about the rotated x axis; rotation
about the rotated z axis. Wolfram Mathworld claim this is the most common
convention : http://mathworld.wolfram.com/EulerAngles.html.

*********************
Direction of rotation
*********************

The direction of rotation is given by the right-hand rule (orient the thumb of
the right hand along the axis around which the rotation occurs, with the end of
the thumb at the positive end of the axis; curl your fingers; the direction
your fingers curl is the direction of rotation).  Therefore, the rotations are
counterclockwise if looking along the axis of rotation from positive to
negative.

****************************
Terms used in function names
****************************

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *euler* : (sequence of) rotation angles about the z, y, x axes (in that
  order)
* *axangle* : rotations encoded by axis vector and angle scalar
* *quat* : quaternion shape (4,)
"""

"""
The quat format is changed to (x,y,z,w) from original version (w,x,y,z) by Yang Linhan 

On ur robot teaching pendant, the orientation is represented by rpy with intrinsic z-y-x order which is equivalent to extrinsic x-y-z order.

So we use sxyz as the default order in this file.
"""

import math

import numpy as np

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0

_MAX_FLOAT = np.maximum_sctype(np.float64)
_FLOAT_EPS = np.finfo(np.float64).eps

def euler2mat(ai, aj, ak, axes='sxyz'):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array (3, 3)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def mat2euler(mat, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS4:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS4:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler2quat(ai, aj, ak, axes='sxyz'):
    """Return `quaternion` from Euler angles and axis sequence `axes`

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return np.array([q[1], q[2], q[3], q[0]])


def quat2euler(quaternion, axes='sxyz'):
    """Euler angles from `quaternion` for specified axis sequence `axes`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> angles = quat2euler([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True
    """
    return mat2euler(quat2mat(quaternion), axes)

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    x, y, z, w = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def mat2quat(M):
    ''' Calculate quaternion corresponding to given rotation matrix

    Method claimed to be robust to numerical errors in `M`.

    Constructs quaternion by calculating maximum eigenvector for matrix
    ``K`` (constructed from input `M`).  Although this is not tested, a maximum
    eigenvalue of 1 corresponds to a valid rotation.

    A quaternion ``q*-1`` corresponds to the same rotation as ``q``; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).

    See notes.

    Parameters
    ----------
    M : array-like
      3x3 rotation matrix

    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]

    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090

    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
    quaternion from a rotation matrix", AIAA Journal of Guidance,
    Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
    0731-5090

    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q

if __name__ == '__main__':
    print(euler2quat(np.pi/2,np.pi/2,np.pi/2))
    # print(quat2euler([0.915917,-0.0261175,-0.000283125,-0.400518]))