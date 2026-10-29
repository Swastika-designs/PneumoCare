"""T4-T6 image processing on a synthetic phantom and the bundled samples."""
import numpy as np
import pytest

from app.core.errors import ValidationFailed
from app.services import imaging as im
from tests.conftest import png_bytes, requires_samples


def test_psnr_and_ssim_identity(phantom):
    assert im.psnr(phantom, phantom) == float("inf")
    assert im.ssim(phantom, phantom) == pytest.approx(1.0, abs=1e-6)


def test_denoising_improves_psnr(phantom):
    r = im.restore(phantom, "salt_pepper", 10)
    metrics = {m["method"]: m["psnr"] for m in r["metrics"]}
    assert metrics["Median filter (5x5)"] > metrics["Noisy input"] + 5


def test_enhancement_panels(phantom):
    r = im.enhance(phantom)
    assert len(r["panels"]) == 6
    assert all(p["image"].startswith("data:image/png;base64,") for p in r["panels"])
    with pytest.raises(ValidationFailed):
        im.enhance(phantom, gamma=10)


def test_otsu_threshold_between_modes(phantom):
    r = im.threshold(phantom)
    assert 60 < r["otsu_threshold"] < 170


def test_segmentation_finds_two_lungs(phantom):
    r = im.segment(phantom)
    assert r["components"] == 2
    # The two phantom lungs cover ~2 * pi * 26 * 50 / 160^2 = 32% of the image.
    assert 20 < r["lung_area_pct"] < 45
    assert r["quality"]["plausible"]
    assert set(r["field_area_pct"]) == {"left", "right"}
    assert "k-means" not in " ".join(p["title"] for p in r["panels"])


def test_segmentation_matches_phantom_lungs(phantom):
    yy, xx = np.mgrid[:160, :160]
    truth = np.zeros((160, 160), bool)
    for cx in (50, 110):
        truth |= ((xx - cx) / 26) ** 2 + ((yy - 85) / 50) ** 2 < 1
    mask = im.lung_mask(phantom)[0] > 0
    dice = 2 * (mask & truth).sum() / (mask.sum() + truth.sum())
    assert dice > 0.85


def test_segmentation_ignores_dark_corner_and_edge_strip(phantom):
    img = phantom.copy()
    img[:30, :30] = 5  # dark corner: the old flood fill turned this into a 100% mask
    img[:, 150:] = 40  # dark strip along the right edge (air outside the body)
    r = im.segment(img)
    assert r["lung_area_pct"] < 50
    mask = im.lung_mask(img)[0]
    assert not mask[:30, :30].any() and not mask[:, 152:].any()


def test_segmentation_flags_blank_image():
    r = im.segment(np.full((128, 128), 140, np.uint8))
    assert r["components"] == 0 and r["lung_area_pct"] == 0
    assert not r["quality"]["plausible"]


def test_compression_ratio_increases_as_quality_drops(phantom):
    rows = [r for r in im.compress(phantom)["table"] if r["format"].startswith("JPEG")]
    sizes = [r["bytes"] for r in rows]
    assert sizes == sorted(sizes, reverse=True)
    assert rows[0]["psnr"] > rows[-1]["psnr"]


def test_ransac_recovers_synthetic_homography(phantom):
    rng = np.random.default_rng(1)
    textured = np.clip(phantom.astype(int) + rng.integers(-40, 40, phantom.shape), 0, 255).astype(np.uint8)
    r = im.match_ransac(textured, angle=10, scale=0.95, perspective=0.0)
    assert r["ransac_inliers"] >= 4
    assert r["corner_error_ransac_px"] < 3.0


def test_sift_keypoints_reported(phantom):
    r = im.sift_keypoints(phantom)
    assert r["keypoints"] > 0 and r["descriptor_shape"][1] == 128


@requires_samples
def test_imaging_api_with_sample(client):
    samples = client.get("/api/imaging/samples").json()
    assert samples
    name = samples[0]["name"]
    for endpoint in ("enhance", "filters", "threshold", "segment", "compress", "sift", "visualize"):
        r = client.post(f"/api/imaging/{endpoint}", data={"sample": name})
        assert r.status_code == 200, (endpoint, r.text[:200])
        assert r.json()["panels"]
    r = client.post("/api/imaging/match", data={"sample": name})
    assert r.json()["corner_error_ransac_px"] < r.json()["corner_error_least_squares_px"]
    assert client.get(f"/api/imaging/samples/{name}").headers["content-type"] == "image/png"
    assert client.get("/api/imaging/samples/..%2F..%2Fsecret.png").status_code == 404


def test_imaging_api_upload_and_errors(client, phantom):
    r = client.post("/api/imaging/threshold", files={"file": ("p.png", png_bytes(phantom), "image/png")})
    assert r.status_code == 200
    assert client.post("/api/imaging/enhance").status_code == 422
    r = client.post("/api/imaging/threshold?block_size=4", files={"file": ("p.png", png_bytes(phantom), "image/png")})
    assert r.status_code == 422
