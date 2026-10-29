"""End-to-end checks against the running API: error handling, upload validation and report exports.

Usage (backend running):  python scripts/check_live_api.py [--base http://127.0.0.1:8000]
Exported files are parsed (JSON / CSV) or structurally checked (PDF), not just status-checked.
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def png(arr, mode=None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    http = httpx.Client(base_url=args.base, timeout=300)
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    def err(r):
        try:
            return r.json()["error"]
        except Exception:
            return {}

    # --- error handling: every failure uses the {"error": {code, message}} envelope
    cases = [
        ("unknown tutorial -> 404", http.get("/api/tutorials/T99"), 404, "not_found"),
        ("unknown EHR source -> 404", http.get("/api/ehr/overview", params={"source": "nope:x"}), 404, "not_found"),
        (
            "unknown column -> 422",
            http.get("/api/ehr/column", params={"source": "synthea:patients", "column": "XYZ"}),
            422,
            "validation_failed",
        ),
        (
            "bad PCA scaling -> 422",
            http.get("/api/analytics/pca", params={"scaling": "weird"}),
            422,
            "validation_failed",
        ),
        (
            "bad KPI grouping -> 422",
            http.get("/api/operations/kpis", params={"group_by": "shoe"}),
            422,
            "validation_failed",
        ),
        (
            "invalid scenario body -> 422",
            http.post("/api/operations/scenarios", json={"scenarios": [{"staff_fte": -1}]}),
            422,
            "invalid_request",
        ),
        ("empty text -> 422", http.post("/api/text/entities", json={"text": ""}), 422, "invalid_request"),
        ("unknown report -> 404", http.get("/api/reports/nope"), 404, "not_found"),
        (
            "bad report format -> 422",
            http.get("/api/reports/registry", params={"format": "docx"}),
            422,
            "invalid_request",
        ),
        (
            "bad screening context -> 422",
            http.post(
                "/api/screening/predict",
                params={"context": "T1"},
                files={"file": ("a.png", png(np.zeros((64, 64), np.uint8)))},
            ),
            422,
            "invalid_request",
        ),
        ("image op without image -> 422", http.post("/api/imaging/enhance"), 422, "validation_failed"),
        ("sample path traversal -> 404", http.get("/api/imaging/samples/..%2F..%2Fpneumocare.db"), 404, "not_found"),
        ("unknown route -> 404 envelope", http.get("/api/does-not-exist"), 404, "not_found"),
    ]
    for name, r, status, code in cases:
        ok = r.status_code == status and (code is None or err(r).get("code") == code)
        check(name, ok, f"got {r.status_code} {err(r).get('code', '')}")

    # --- upload validation (X-ray endpoint and CSV endpoint)
    uploads = [
        ("text file as X-ray", ("notes.txt", b"hello", "text/plain"), "not a readable image"),
        ("empty file", ("empty.png", b"", "image/png"), "empty"),
        ("tiny image", ("tiny.png", png(np.zeros((8, 8), np.uint8)), "image/png"), "too small"),
        ("oversized file (11 MB)", ("big.png", b"\x89PNG" + b"0" * (11 * 1024 * 1024), "image/png"), "larger than"),
    ]
    for name, f, expect in uploads:
        r = http.post("/api/screening/predict", params={"gradcam": "false"}, files={"file": f})
        check(f"reject {name}", r.status_code == 422 and expect in err(r).get("message", ""), err(r).get("message", ""))
    rgb = np.zeros((128, 128, 3), np.uint8)
    rgb[..., 0] = 220
    r = http.post("/api/screening/predict", params={"gradcam": "false"}, files={"file": ("photo.png", png(rgb, "RGB"))})
    check(
        "colour photo accepted with warning", r.status_code == 200 and any("colour" in w for w in r.json()["warnings"])
    )
    r = http.post("/api/ehr/upload", files={"file": ("x.txt", b"a,b\n1,2", "text/plain")})
    check("reject non-CSV upload", r.status_code == 422, err(r).get("message", ""))
    r = http.post("/api/ehr/upload", files={"file": ("bad.csv", b"", "text/csv")})
    check("reject empty CSV", r.status_code == 422, err(r).get("message", ""))
    r = http.post(
        "/api/ehr/upload",
        files={"file": ("audit_small.csv", b"age,sex,bmi\n40,F,22.5\n55,M,\n61,F,31.2\n", "text/csv")},
    )
    ok = r.status_code == 200
    check("accept valid CSV", ok, r.text[:80])
    if ok:
        ov = http.get("/api/ehr/overview", params={"source": r.json()["id"]}).json()
        check("uploaded CSV analysable", ov["rows"] == 3 and ov["total_missing_cells"] == 1)

    # --- exports: parse every format of every report
    reports = http.get("/api/reports").json()
    for rep in reports:
        for fmt in ("json", "csv", "pdf"):
            r = http.get(f"/api/reports/{rep['id']}", params={"format": fmt})
            if r.status_code != 200:
                check(f"export {rep['id']}.{fmt}", False, f"{r.status_code} {err(r).get('message', '')}")
                continue
            if fmt == "json":
                body = json.loads(r.content)
                ok, detail = body["report"] == rep["id"] and bool(body["data"]), f"{len(r.content)} bytes"
            elif fmt == "csv":
                rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
                ok, detail = len(rows) > 3, f"{len(rows)} rows"
            else:
                ok = (
                    r.content.startswith(b"%PDF")
                    and b"%%EOF" in r.content[-1024:]
                    and r.content.count(b"/Type /Page") >= 1
                )
                pages = r.content.count(b"/Type /Page") - r.content.count(b"/Type /Pages")
                detail = f"{len(r.content)} bytes, {pages} page(s)"
            check(f"export {rep['id']}.{fmt}", ok and "attachment" in r.headers.get("content-disposition", ""), detail)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} live API checks passed")
    (ROOT / "reports" / "live_api_checks.json").write_text(
        json.dumps([{"check": n, "passed": p, "detail": d} for n, p, d in results], indent=2)
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
