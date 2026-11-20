"""Minimal helpers: load an explicit 3DGS PLY and render RGB / depth / alpha with gsplat.

World frame is left untouched so that every output (mesh, SDF) stays aligned with the splat.
Cameras use the OpenCV convention (x right, y down, z forward); viewmats are world->camera.
The "level frame" is world rotated so that +Z is up:  p_level = R @ (p_world - c).
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from plyfile import PlyData


def _msvc_jit_shim():
    """gsplat JIT-compiles its CUDA backend on first use and passes GCC-only flags to the host
    compiler, which cl.exe rejects. Rewrite them before torch sees them (Windows only). The flags
    are part of torch's build hash, so the shim must be active on every run, not just the build.
    """
    if sys.platform != "win32":
        return
    import torch.utils.cpp_extension as ce
    if getattr(ce, "_gs_msvc_shim", False):
        return
    orig = ce._jit_compile

    def fix(flags):
        return [{"-O3": "/O2", "-O0": "/Od"}.get(f, f) for f in (flags or []) if f != "-Wno-attributes"]

    def _jit_compile(name, sources, extra_cflags, *args, **kw):
        return orig(name, sources, fix(extra_cflags), *args, **kw)

    ce._jit_compile = _jit_compile
    ce._gs_msvc_shim = True
    # CUDA 13's CCCL headers refuse MSVC's traditional preprocessor (only matters while building).
    os.environ.setdefault("NVCC_APPEND_FLAGS", "-Xcompiler /Zc:preprocessor")


_msvc_jit_shim()
from gsplat import rasterization  # noqa: E402  (must come after the shim)


@dataclass
class Gaussians:
    means: torch.Tensor      # [N,3]
    quats: torch.Tensor      # [N,4] wxyz
    scales: torch.Tensor     # [N,3] (activated)
    opac: torch.Tensor       # [N]   (activated)
    sh: torch.Tensor         # [N,K,3]
    sh_degree: int

    def __len__(self):
        return self.means.shape[0]

    def subset(self, mask):
        return Gaussians(self.means[mask], self.quats[mask], self.scales[mask],
                         self.opac[mask], self.sh[mask], self.sh_degree)


def load_ply(path, device="cuda"):
    v = PlyData.read(path)["vertex"]
    names = [p.name for p in v.properties]
    n = len(v["x"])
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
    n_rest = sum(1 for s in names if s.startswith("f_rest_"))
    if n_rest:
        rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], 1)
        rest = rest.reshape(n, 3, n_rest // 3).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    sh_degree = int(round(np.sqrt(sh.shape[1]))) - 1
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1))
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], 1)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().to(device)

    quats = torch.nn.functional.normalize(t(quats), dim=-1)
    return Gaussians(t(xyz), quats, t(scales), t(opac), t(sh), sh_degree)


def contribution(g, viewmats, Ks, width, height, near=1e-3, far=1e10, batch=8):
    """Per-Gaussian visible contribution summed over views: sum over pixels of alpha_i G_i T_i.

    Rendered as the gradient of a constant-1 feature image w.r.t. that feature, so a Gaussian that is
    hidden behind others in every view gets ~0. viewmats [C,4,4] world->camera, Ks [C,3,3]."""
    w = torch.zeros(len(g), device=g.means.device)
    ones = torch.ones((len(g), 1), device=g.means.device, requires_grad=True)
    for s in range(0, len(viewmats), batch):
        img, _, _ = rasterization(g.means, g.quats, g.scales, g.opac, ones, viewmats[s:s + batch], Ks[s:s + batch],
                                  width, height, near_plane=near, far_plane=far, sh_degree=None, render_mode="RGB")
        img.sum().backward()
        w += ones.grad[:, 0]
        ones.grad = None
    return w


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def estimate_frame(xyz, pct=3.0, up=None):
    """Level frame from the bulk of the Gaussians. Returns (c, R) with rows of R = e1, e2, up.

    up defaults to the smallest-variance axis of the core points, signed so that the long
    tail of the height distribution (buildings, trees) points up.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    lo, hi = np.percentile(xyz, [pct, 100 - pct], axis=0)
    core = ((xyz >= lo) & (xyz <= hi)).all(1)
    c = xyz[core].mean(0)
    if up is None:
        _, V = np.linalg.eigh(np.cov((xyz[core] - c).T))
        up = V[:, 0]
        h = (xyz[core] - c) @ up
        if ((h - h.mean()) ** 3).mean() < 0:
            up = -up
    up = normalize(up)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = normalize(ref - (ref @ up) * up)
    e2 = np.cross(up, e1)
    return c, np.stack([e1, e2, up], 0)


def look_at(eye, target, up):
    """World->camera 4x4 (OpenCV convention). `up` is only a hint for the image-up direction."""
    eye = np.asarray(eye, dtype=np.float64)
    f = normalize(np.asarray(target, dtype=np.float64) - eye)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        raise ValueError("look_at: view direction parallel to up hint")
    r = normalize(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f], 0)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ eye
    return M


def intrinsics(width, height, fov_x_deg):
    fx = 0.5 * width / np.tan(np.radians(fov_x_deg) / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)


@torch.no_grad()
def render(g, viewmat, K, width, height, near=1e-3, far=1e6, with_rgb=True, ortho=False):
    """Returns rgb [H,W,3] (or None), expected z-depth [H,W], alpha [H,W]."""
    dev = g.means.device
    vm = torch.as_tensor(np.asarray(viewmat), dtype=torch.float32, device=dev)[None]
    Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=dev)[None]
    out, alpha, _ = rasterization(
        g.means, g.quats, g.scales, g.opac, g.sh, vm, Kt, width, height,
        sh_degree=g.sh_degree, render_mode="RGB+ED" if with_rgb else "ED",
        near_plane=near, far_plane=far, packed=True,
        camera_model="ortho" if ortho else "pinhole",
    )
    alpha = alpha[0, ..., 0]
    if with_rgb:
        return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], alpha
    return None, out[0, ..., 0], alpha


def _rot_and_inv_var(g):
    """Per-Gaussian rotation matrices [N,3,3] and inverse variances [N,3] (cached on the object)."""
    if getattr(g, "_rot", None) is None:
        w, x, y, z = g.quats.unbind(-1)
        g._rot = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
        g._inv_var = 1.0 / g.scales.clamp(min=1e-7) ** 2
    return g._rot, g._inv_var


@torch.no_grad()
def depth_plane_coeffs(g, viewmat, K, ortho=False, max_tan=5.7):
    """Per-Gaussian affine depth  t(u,v) = a + b*u + c*v  in pixel coordinates.

    gsplat composites the depth of each Gaussian's CENTRE, so a large tilted Gaussian shows up as
    a camera-facing plate. The depth at which a ray meets the Gaussian's maximum density is
        t*(r) = (r^T P mu) / (r^T P r),   P = Sigma^-1, mu = centre, r = (x, y, 1)
    which for a flat Gaussian is the ray / tangent-plane intersection. Its first-order expansion
    around the centre ray is affine in the pixel, with gradient  -mu_z^2 (P mu)_xy / (mu^T P mu)
    (exactly affine for an orthographic camera: -(P_zx, P_zy) / P_zz). Rendering (a, b, c) as three
    colour channels therefore gives the composited plane-aware depth as  C0 + u*C1 + v*C2.
    The slope is clamped to max_tan (80 deg) so edge-on Gaussians cannot explode.
    """
    dev = g.means.device
    vm = torch.as_tensor(np.asarray(viewmat), dtype=torch.float32, device=dev)
    Kd = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = float(Kd[0, 0]), float(Kd[1, 1]), float(Kd[0, 2]), float(Kd[1, 2])
    Rg, iv = _rot_and_inv_var(g)
    Rwc = vm[:3, :3]
    dc = g.means @ Rwc.T + vm[:3, 3]                              # centre in camera space
    mu_z = dc[:, 2]
    if ortho:
        zw = Rwc[2]                                               # camera z axis in world
        pz = torch.einsum("nij,nj->ni", Rg, torch.einsum("nij,i->nj", Rg, zw) * iv) @ Rwc.T   # row z of P (camera frame)
        grad = -pz[:, :2] / pz[:, 2:3].clamp(min=1e-20)           # d depth / d (X, Y) in camera-plane units
        lim = torch.full_like(mu_z, max_tan)
        off_x, off_y = dc[:, 0], dc[:, 1]
    else:
        eye = -(Rwc.T @ vm[:3, 3])
        d = g.means - eye
        q = torch.einsum("nij,nj->ni", Rg, torch.einsum("nij,ni->nj", Rg, d) * iv)   # P d (world)
        m = (d * q).sum(-1).clamp(min=1e-20)
        zs = torch.where(mu_z > 1e-6, mu_z, torch.ones_like(mu_z))
        grad = -(zs ** 2)[:, None] * (q @ Rwc.T)[:, :2] / m[:, None]   # d depth / d (x, y) normalised coords
        lim = max_tan * zs
        off_x, off_y = dc[:, 0] / zs, dc[:, 1] / zs
    grad = grad * (lim / grad.norm(dim=1).clamp(min=1e-20)).clamp(max=1.0)[:, None]
    b, c = grad[:, 0] / fx, grad[:, 1] / fy
    a = mu_z - grad[:, 0] * (cx / fx + off_x) - grad[:, 1] * (cy / fy + off_y)
    return torch.stack([a, b, c], 1)


@torch.no_grad()
def render_median_depth(g, viewmat, K, width, height, far_planes, near=1e-3, ortho=False, min_alpha=0.3, plane=True):
    """Robust surface depth: the depth at which accumulated opacity reaches half of its total.

    gsplat only offers expected depth, which semi-transparent floaters and far background drag
    away from the surface. Here the scene is rendered once per far plane (Gaussians behind the
    plane are culled, those in front are unaffected), giving exact prefix sums of opacity A_k and
    of opacity-weighted depth D_k. For each pixel the bracket in which A crosses half of the total
    opacity is located and the opacity-weighted mean depth of just that bracket,
    (D_k - D_{k-1}) / (A_k - A_{k-1}), is returned. With plane=True the per-Gaussian depth is the
    tangent-plane depth at that pixel (see depth_plane_coeffs) instead of the centre depth.

    Returns (depth [H,W], 0 where the total opacity inside the last far plane is below min_alpha;
    total alpha [H,W]).
    """
    dev = g.means.device
    vm = torch.as_tensor(np.asarray(viewmat), dtype=torch.float32, device=dev)[None]
    Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=dev)[None]
    if plane:
        feat = depth_plane_coeffs(g, viewmat, K, ortho)
        vv, uu = torch.meshgrid(torch.arange(height, device=dev) + 0.5, torch.arange(width, device=dev) + 0.5, indexing="ij")
    else:
        feat = g.opac[:, None]

    def raster(far):
        out, alpha, _ = rasterization(
            g.means, g.quats, g.scales, g.opac, feat, vm, Kt, width, height,
            sh_degree=None, render_mode="RGB" if plane else "D", near_plane=near, far_plane=float(far), packed=True,
            camera_model="ortho" if ortho else "pinhole",
        )
        o = out[0]
        return (o[..., 0] + uu * o[..., 1] + vv * o[..., 2]) if plane else o[..., 0], alpha[0, ..., 0]

    d_tot, a_tot = raster(far_planes[-1])
    thr = torch.where(a_tot >= min_alpha, 0.5 * a_tot, torch.full_like(a_tot, float("inf")))
    depth = torch.zeros((height, width), device=dev)
    found = torch.zeros((height, width), dtype=torch.bool, device=dev)
    prev_d = torch.zeros((height, width), device=dev)
    prev_a = torch.zeros((height, width), device=dev)
    for k, far in enumerate(far_planes):
        d, a = (d_tot, a_tot) if k == len(far_planes) - 1 else raster(far)
        cross = (~found) & (a >= thr)
        depth = torch.where(cross, (d - prev_d) / (a - prev_a).clamp(min=1e-6), depth)
        found |= cross
        prev_d, prev_a = d, a
    return depth.clamp(min=0), a_tot
