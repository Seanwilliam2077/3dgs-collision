"""Meshing of binary voxel grids (torch, GPU or CPU).

merged_faces(S): the boundary faces of the solid voxels S, merged into rectangles (runs along one
in-plane axis, then identical runs in consecutive rows). Exact to the voxels and closed (T-junctions
lie exactly on edges, so there are no gaps). Flat, axis-aligned floors and walls become a few large
rectangles; noisy or sloped surfaces become terraces of small ones.

smooth_binary(S, n): n rounds of 3x3x3 opening + closing: drops solid specks and fills free pockets of
1-2 voxels, moving surfaces by at most one voxel.
"""
import numpy as np
import torch
import torch.nn.functional as F


def smooth_binary(S, n=1):
    x = S[None, None].half()
    dil = lambda t: F.max_pool3d(t, 3, stride=1, padding=1)
    ero = lambda t: -F.max_pool3d(-t, 3, stride=1, padding=1)
    for _ in range(n):
        x = dil(ero(x))
        x = ero(dil(x))
    return x[0, 0] > 0.5


@torch.no_grad()
def merged_faces(S):
    """S: bool [nx,ny,nz]. Returns (V [M,3] float64 in index coordinates, voxel centres at integers;
    F [T,3] int64 with normals pointing out of S; per-orientation rectangle counts)."""
    dev = S.device
    Vl, Fl, per, off = [], [], {}, 0
    for ax in range(3):
        oth = [d for d in range(3) if d != ax]                              # (u, w): in-plane axes
        P = S.permute(ax, oth[0], oth[1])                                    # [n, u, w]
        Ul, Wl = P.shape[1], P.shape[2]
        s_ax = 1 if ax != 1 else -1                                          # sign of (e_u x e_w) along ax
        for sgn in (1, -1):
            M = torch.zeros_like(P)
            if sgn == 1:
                M[:-1] = P[:-1] & ~P[1:]                                     # face between n and n+1
            else:
                M[1:] = P[1:] & ~P[:-1]                                      # face between n and n-1
            d = F.pad(M.to(torch.int8), (1, 1))
            d = d[..., 1:] - d[..., :-1]
            st, en = (d == 1).nonzero(), (d == -1).nonzero()                 # run starts / ends (exclusive)
            del d, M
            if len(st) == 0:
                continue
            n_, u_, w0, w1 = st[:, 0], st[:, 1], st[:, 2], en[:, 2]
            key = (n_ * (Wl + 1) + w0) * (Wl + 1) + w1
            o = torch.argsort(key * (Ul + 1) + u_)
            n_, u_, w0, w1, key = n_[o], u_[o], w0[o], w1[o], key[o]
            brk = torch.ones(len(key), dtype=torch.bool, device=dev)
            brk[1:] = (key[1:] != key[:-1]) | (u_[1:] != u_[:-1] + 1)
            first = brk.nonzero()[:, 0]
            last = torch.cat([first[1:] - 1, torch.tensor([len(key) - 1], device=dev)])
            n_, u0, u1, w0, w1 = n_[first], u_[first], u_[last], w0[first], w1[first]
            m = len(n_)
            pa = n_.double() + 0.5 * sgn
            corners = [(u0 - 0.5, w0 - 0.5), (u1 + 0.5, w0 - 0.5), (u1 + 0.5, w1 - 0.5), (u0 - 0.5, w1 - 0.5)]
            V4 = torch.zeros((m, 4, 3), dtype=torch.float64, device=dev)
            for q, (cu, cw) in enumerate(corners):
                V4[:, q, ax], V4[:, q, oth[0]], V4[:, q, oth[1]] = pa, cu.double(), cw.double()
            base = (torch.arange(m, device=dev) * 4)[:, None] + off
            tri = torch.cat([base + torch.tensor([0, 1, 2], device=dev), base + torch.tensor([0, 2, 3], device=dev)], 0)
            if sgn * s_ax < 0:
                tri = tri[:, [0, 2, 1]]
            Vl.append(V4.reshape(-1, 3).cpu().numpy())
            Fl.append(tri.cpu().numpy())
            off += 4 * m
            per[f"{'xyz'[ax]}{'+' if sgn > 0 else '-'}"] = m
    if not Vl:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), per
    V = np.concatenate(Vl)
    Fc = np.concatenate(Fl)
    key = np.round(V * 2).astype(np.int64)                                   # weld: corners are half-integers
    _, first_, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[first_], inv.reshape(-1)[Fc], per


if __name__ == "__main__":                                                   # self-test on a synthetic room
    import trimesh
    S = torch.ones((80, 60, 30), dtype=torch.bool)
    S[5:75, 5:55, 3:22] = False                                              # room interior (free)
    S[30:42, 20:28, 3:8] = True                                              # a table block
    S[0:80, 30, 3:22] = True                                                 # a thin wall ...
    S[50:56, 30, 3:17] = False                                               # ... with a door
    S = F.pad(S[None, None].float(), (1, 1, 1, 1, 1, 1))[0, 0] > 0           # free border: the mesh is closed
    V, Fc, per = merged_faces(S)
    m = trimesh.Trimesh(V, Fc, process=False)
    nvox = int(S.sum())
    print("rectangles", per, "triangles", len(Fc), "vertices", len(V))
    print("volume", round(float(m.volume), 2), "solid voxels in the closed part", nvox,
          "-> normals point outward" if m.volume > 0 else "-> normals INWARD")
