"""One-off: JIT-compile gsplat's CUDA backend on Windows (run through py.bat so MSVC is on PATH).

Known limitation of this build: gsplat 1.5.3 declares two kernels with a mix of `const at::Tensor`
and `at::Tensor` by-value parameters. cl.exe and nvcc's front end mangle those differently, so
  launch_rasterize_to_pixels_2dgs_bwd_kernel            (2DGS backward)
  launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel (3DGUT forward)
stay unresolved at link time. /FORCE:UNRESOLVED links anyway; calling either of them would crash.
Plain 3DGS forward rendering, which is all this prototype uses, is unaffected.
"""
import os, time
os.environ.setdefault("VERBOSE", "1")
os.environ["LINK"] = "/FORCE:UNRESOLVED"
import gs_core  # installs the MSVC flag shim before gsplat's backend is imported
t = time.time()
from gsplat.cuda._backend import _C
print("gsplat backend:", _C, f"({time.time()-t:.0f}s)")
