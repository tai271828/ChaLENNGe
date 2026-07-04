"""Structural verification of BoundedBlendReconstruction (doc 03 Stage A / doc 04 §1)."""

import numpy as np
import keras
from keras import backend as K

K.set_floatx("float64")

from lbm_ml.lattice.stencil import LB_stencil
from lbm_ml.lattice.symmetry import BoundedBlendReconstruction, LBrot90, LBmirror
from lbm_ml.model.network import MODEL_REGISTRY

rng = np.random.default_rng(0)
c, w, cs2, compute_feq = LB_stencil()

# Random positive mass-normalised populations near equilibrium
f = np.abs(rng.normal(1.0, 0.2, size=(64, 9)))
f /= f.sum(axis=1, keepdims=True)

# --- 1. equilibrium matches the repo's numpy stencil ---------------------
layer = BoundedBlendReconstruction()
feq_layer = np.asarray(layer._equilibrium(keras.ops.convert_to_tensor(f)))
rho = f.sum(axis=1)
ux = (f * c[:, 0]).sum(axis=1) / rho
uy = (f * c[:, 1]).sum(axis=1) / rho
feq_np = compute_feq(
    np.zeros((64, 1, 9)), rho[:, None], ux[:, None], uy[:, None], c, w
)[:, 0, :]
err_eq = np.abs(feq_layer - feq_np).max()
print(
    f"1. equilibrium vs stencil     max err = {err_eq:.3e}  {'PASS' if err_eq < 1e-14 else 'FAIL'}"
)

results = {}
for name in ("lenn_18_18_18_softmax_bounded", "resnet_softmax_bounded"):
    model = MODEL_REGISTRY[name]()
    out = np.asarray(model(keras.ops.convert_to_tensor(f)))

    # --- 2. exact conservation -------------------------------------------
    d_mass = np.abs(out.sum(1) - f.sum(1)).max()
    d_px = np.abs((out * c[:, 0]).sum(1) - (f * c[:, 0]).sum(1)).max()
    d_py = np.abs((out * c[:, 1]).sum(1) - (f * c[:, 1]).sum(1)).max()
    cons = max(d_mass, d_px, d_py)
    print(
        f"2. conservation [{name}]  max defect = {cons:.3e}  {'PASS' if cons < 1e-12 else 'FAIL'}"
    )

    # --- 3. D4 equivariance ------------------------------------------------
    worst = 0.0
    ft = keras.ops.convert_to_tensor(f)
    for tf_in, tf_out in ((lambda x: LBrot90(x, 1),) * 2, (LBmirror,) * 2):
        lhs = np.asarray(model(tf_in(ft)))
        rhs = np.asarray(tf_out(keras.ops.convert_to_tensor(np.asarray(model(ft)))))
        worst = max(worst, np.abs(lhs - rhs).max())
    print(
        f"3. D4 equivariance [{name}]  max dev = {worst:.3e}  {'PASS' if worst < 1e-10 else 'FAIL'}"
    )
    results[name] = model

# --- 4. bound behaviour: g -> 0.5 damps toward equilibrium, g -> 1 raw ----
fpred = np.abs(rng.normal(1.0, 0.3, size=(64, 9)))
fpred /= fpred.sum(axis=1, keepdims=True)
fpre_t, fpred_t = map(keras.ops.convert_to_tensor, (f, fpred))
dists = []
for theta in (-10.0, 0.0, 10.0):
    lay = BoundedBlendReconstruction(theta_init=theta)
    out = np.asarray(lay(fpre_t, fpred_t))
    dists.append(np.linalg.norm(out - feq_np))
mono = dists[0] < dists[1] < dists[2]
print(
    f"4. bound: |out - feq| for g=(0.5, 0.75, 1) = {[f'{d:.4f}' for d in dists]}  {'PASS' if mono else 'FAIL'}"
)
# g -> 1 must recover the plain symmetric projection of the raw prediction
from lbm_ml.lattice.symmetry import SymmetricAlgReconstruction

raw_proj = np.asarray(SymmetricAlgReconstruction()(fpre_t, fpred_t))
lim = np.abs(
    np.asarray(BoundedBlendReconstruction(theta_init=30.0)(fpre_t, fpred_t)) - raw_proj
).max()
print(
    f"   g->1 recovers symmetric projection: max err = {lim:.3e}  {'PASS' if lim < 1e-10 else 'FAIL'}"
)

# --- 5. serialization round-trip ------------------------------------------
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "m.keras")
m = results["lenn_18_18_18_softmax_bounded"]
m.save(path)
m2 = keras.models.load_model(path)
rt = np.abs(
    np.asarray(m2(keras.ops.convert_to_tensor(f)))
    - np.asarray(m(keras.ops.convert_to_tensor(f)))
).max()
print(
    f"5. save/load round-trip  max err = {rt:.3e}  {'PASS' if rt < 1e-15 else 'FAIL'}"
)

# --- param count delta (budget-matching info for doc 02) -------------------
for a, b in (("lenn_18_18_18_softmax_cons", "lenn_18_18_18_softmax_bounded"),):
    pa = MODEL_REGISTRY[a]().count_params()
    pb = MODEL_REGISTRY[b]().count_params()
    print(f"6. params: {a}={pa}  {b}={pb}  (+{pb - pa})")
