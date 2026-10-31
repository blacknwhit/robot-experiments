"""Rigid physical surface sampling and transforms (no renderer, no future selection).

MuJoCo's compiled mesh vertices already include compiler scaling/centering. They
must be transformed by the compiled geom pose, not the uncompiled XML body pose.
Primitive geoms are triangulated locally so primitive-only fixtures are retained.
"""
import numpy as np


def transform_points(local, rotations, translations):
    return np.einsum("nij,nj->ni", rotations, local) + translations


def box_triangles(size):
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * size
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
             [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
             [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return v[np.asarray(faces)]


def round_triangles(kind, size, segments=32, rings=16):
    """Triangulate curved rendering primitives; tessellation is cache-keyed."""
    if segments < 8 or rings < 4:
        raise ValueError("insufficient primitive tessellation")
    theta = np.arange(segments) * (2 * np.pi / segments)
    c, s = np.cos(theta), np.sin(theta)
    if kind in ("sphere", "ellipsoid", "capsule"):
        phi = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
        radius = float(size[0])
        xyz = np.stack([np.cos(phi)[:, None] * c, np.cos(phi)[:, None] * s,
                        np.broadcast_to(np.sin(phi)[:, None], (len(phi), segments))], -1)
        if kind == "ellipsoid":
            xyz *= np.asarray(size)
        else:
            xyz *= radius
            if kind == "capsule":
                # Duplicate equator rings at +/-half-length to form the cylinder.
                lower = xyz[: rings // 2 + 1].copy()
                upper = xyz[rings // 2 :].copy()
                lower[..., 2] -= float(size[1])
                upper[..., 2] += float(size[1])
                xyz = np.concatenate([lower, upper], axis=0)
    elif kind == "cylinder":
        radius, half = map(float, size[:2])
        xyz = np.array([np.stack([radius * c, radius * s, np.full(segments, z)], -1) for z in (-half, half)])
    else:
        raise ValueError(f"unsupported primitive: {kind}")
    tris = []
    for i in range(len(xyz) - 1):
        for j in range(segments):
            k = (j + 1) % segments
            tris.extend([[xyz[i, j], xyz[i, k], xyz[i + 1, k]],
                         [xyz[i, j], xyz[i + 1, k], xyz[i + 1, j]]])
    if kind == "cylinder":
        for i in range(2):
            center = np.array([0, 0, xyz[i, 0, 2]])
            for j in range(segments):
                tris.append([center, xyz[i, j], xyz[i, (j + 1) % segments]])
    return np.asarray(tris, dtype=np.float64)


def geom_triangles(model, geom_id, *, primitive_segments=32, primitive_rings=16,
                   plane_extent=10.0):
    import mujoco
    kind = int(model.geom_type[geom_id])
    mj = mujoco.mjtGeom
    size = model.geom_size[geom_id]
    if kind == int(mj.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom_id])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        vertices = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float64)
        faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int64)
        return vertices[faces]
    if kind == int(mj.mjGEOM_BOX):
        return box_triangles(size)
    if kind == int(mj.mjGEOM_PLANE):
        a = float(plane_extent)
        return np.array([[[-a, -a, 0], [a, -a, 0], [a, a, 0]],
                         [[-a, -a, 0], [a, a, 0], [-a, a, 0]]])
    names = {int(mj.mjGEOM_SPHERE): "sphere", int(mj.mjGEOM_ELLIPSOID): "ellipsoid",
             int(mj.mjGEOM_CAPSULE): "capsule", int(mj.mjGEOM_CYLINDER): "cylinder"}
    if kind not in names:
        raise ValueError(f"unsupported selected geom type {kind} at geom {geom_id}")
    return round_triangles(names[kind], size, primitive_segments, primitive_rings)


def clip_triangle_cube(triangle, half_extent):
    """Clip a triangle in cube coordinates, retaining original barycentric IDs."""
    polygon = [(p.copy(), b.copy()) for p, b in zip(triangle, np.eye(3))]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            output = []
            if not polygon:
                return []
            p, b = polygon[-1]
            dp = half_extent - sign * p[axis]
            for q, c in polygon:
                dq = half_extent - sign * q[axis]
                if (dp >= 0) != (dq >= 0):
                    alpha = dp / (dp - dq)
                    output.append((p + alpha * (q - p), b + alpha * (c - b)))
                if dq >= 0:
                    output.append((q, c))
                p, b, dp = q, c, dq
            polygon = output
    return [(np.array([polygon[0][0], polygon[i][0], polygon[i + 1][0]]),
             np.array([polygon[0][1], polygon[i][1], polygon[i + 1][1]]))
            for i in range(1, len(polygon) - 1)]


def sample_surfaces(geometries, geom_rotations, geom_positions, base_pose, *,
                    count, half_extent, seed):
    """Area-uniform sample of first-frame surfaces *inside* a base-aligned cube.

    ``geometries`` maps geom ID to compiled geom-local triangles. Clipping is
    performed once. Future replay costs O(N), not O(all scene triangles).
    A fixed face and barycentric coordinates identify each physical point.
    """
    if count < 1 or half_extent <= 0:
        raise ValueError("invalid point count/cube")
    identity_chunks, area_chunks, mapping_chunks, boundary_maps = [], [], [], []
    R, t = np.asarray(base_pose[:3, :3]), np.asarray(base_pose[:3, 3])
    for gid, triangles in geometries.items():
        world = triangles @ np.asarray(geom_rotations[gid]).T + geom_positions[gid]
        cube = (world - t) @ R
        # Most scene faces lie wholly outside or wholly inside: skip Python clipping.
        outside = np.any((cube.min(axis=1) > half_extent) | (cube.max(axis=1) < -half_extent), axis=1)
        inside = np.all(np.abs(cube) <= half_extent, axis=(1, 2))
        # Vectorize all fully contained triangles; do not allocate per-face
        # Python objects or a dense F,3,3 array of identity barycentric maps.
        full_ids = np.flatnonzero(inside)
        if len(full_ids):
            full = cube[full_ids]
            area = np.linalg.norm(np.cross(full[:, 1] - full[:, 0], full[:, 2] - full[:, 0]), axis=-1) * .5
            good = area > 1e-14
            full_ids, area = full_ids[good], area[good]
            identity_chunks.append(np.stack([np.full(len(full_ids), gid), full_ids], -1))
            area_chunks.append(area)
            mapping_chunks.append(np.full(len(full_ids), -1, dtype=np.int64))
        for face_id in np.flatnonzero(~outside & ~inside):
            pieces = clip_triangle_cube(cube[face_id], half_extent)
            for tri, weights in pieces:
                area = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) * .5
                if area > 1e-14:
                    identity_chunks.append(np.array([[gid, face_id]]))
                    area_chunks.append(np.array([area]))
                    mapping_chunks.append(np.array([len(boundary_maps)], dtype=np.int64))
                    boundary_maps.append(weights)
    if not area_chunks:
        raise ValueError("no nondegenerate physical surfaces within sampling cube")
    rng = np.random.default_rng(seed)
    areas = np.concatenate(area_chunks)
    if not len(areas) or areas.sum() <= 0:
        raise ValueError("no nondegenerate physical surfaces within sampling cube")
    indices = rng.choice(len(areas), size=count, p=areas / areas.sum())
    u = rng.random((count, 2))
    sqrt_u = np.sqrt(u[:, 0])
    weights = np.stack([1 - sqrt_u, sqrt_u * (1 - u[:, 1]), sqrt_u * u[:, 1]], -1)
    map_ids = np.concatenate(mapping_chunks)[indices]
    selected_bary = weights.copy()
    clipped_indices = map_ids >= 0
    if clipped_indices.any():
        selected_bary[clipped_indices] = np.einsum("ni,nij->nj", weights[clipped_indices],
                                                  np.asarray(boundary_maps)[map_ids[clipped_indices]])
    ids = np.concatenate(identity_chunks).astype(np.int32)[indices]
    local = np.empty((count, 3), dtype=np.float64)
    for gid in np.unique(ids[:, 0]):
        select = ids[:, 0] == gid
        faces = geometries[int(gid)][ids[select, 1]]
        local[select] = np.einsum("ni,nij->nj", selected_bary[select], faces)
    return {"geom_ids": ids[:, 0], "face_ids": ids[:, 1], "local": local,
            "barycentric": selected_bary,
            "surface_area_m2": float(areas.sum()), "clipped_triangles": len(areas),
            "selected_geom_count": len(np.unique(ids[:, 0]))}
