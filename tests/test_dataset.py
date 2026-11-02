import numpy as np
import pytest

torch = pytest.importorskip("torch")

from exoplanet.data import CLASSES
from exoplanet.views import SAMPLE_VIEWS, load_sample, load_views, normalize

pytestmark = pytest.mark.skipif(not SAMPLE_VIEWS.exists(), reason="sin vistas de la muestra A en data/")


def test_dataset_shapes_types_and_labels():
    from torch.utils.data import DataLoader
    from exoplanet.dataset import ViewDataset
    stars = load_sample(["val"])
    ds = ViewDataset(stars)
    assert len(ds) == len(stars)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    assert batch["local"].shape == (8, 1, 81) and batch["glob"].shape == (8, 1, 201) and batch["log_depth"].shape == (8, 1)
    assert batch["local"].dtype == torch.float32 and batch["y"].dtype == torch.long
    assert all(torch.isfinite(t).all() for t in (ds.local, ds.glob, ds.log_depth)), "NaN o inf en las entradas"
    assert [CLASSES[i] for i in ds.y[:20]] == stars.disposition[:20].tolist()
    assert torch.allclose(ds.local.amin(dim=2), torch.tensor(-1.0)), "la caída de la local debe valer -1"


def test_dataset_matches_sklearn_input():
    from exoplanet.dataset import ViewDataset
    stars = load_sample(["val"]).head(50)
    ds = ViewDataset(stars)
    v = load_views(stars.star_id)
    expected = np.hstack(normalize(v["local"], v["glob"]))
    got = torch.cat([ds.local.squeeze(1), ds.glob.squeeze(1), ds.log_depth], dim=1).numpy()
    assert np.allclose(got, expected, atol=1e-5)


def test_class_weights_like_sklearn_balanced():
    from sklearn.utils.class_weight import compute_class_weight
    from exoplanet.dataset import ViewDataset
    ds = ViewDataset(load_sample(["train"]))
    y = ds.y.numpy()
    assert np.allclose(ds.class_weights().numpy(), compute_class_weight("balanced", classes=np.arange(3), y=y))
