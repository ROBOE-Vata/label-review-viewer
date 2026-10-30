#!/usr/bin/env python3
"""COCO 세그멘테이션 라벨 검수 뷰어 (표준 라이브러리만으로 동작).

사용법:
    python3 viewer.py                     # http://127.0.0.1:8765, 화면에서 데이터셋 경로 입력
    python3 viewer.py <데이터셋 폴더>      # 바로 열기

데이터셋 폴더에는 _annotations.coco.json 과 이미지(raw/)가 있어야 한다 (zip을 푼 바깥 폴더도 됨).
검수 결과는 그 폴더의 review_results.json 에 즉시 저장된다.
"""
import argparse
import csv
import gzip
import io
import json
import os
import shutil
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

try:
    import cv2
    import numpy as np
except ImportError:  # 없으면 겹침 검사를 bbox IoU로 대체
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
STATUSES = ("ok", "ng", "hold")


def _bbox_inter(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def _mask_inter(p1, p2, b1, b2):
    """두 폴리곤의 교집합 픽셀 수."""
    x0, y0 = int(min(b1[0], b2[0])), int(min(b1[1], b2[1]))
    x1 = int(max(b1[0] + b1[2], b2[0] + b2[2])) + 2
    y1 = int(max(b1[1] + b1[3], b2[1] + b2[3])) + 2
    masks = []
    for p in (p1, p2):
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = (np.array(p, np.float32).reshape(-1, 2) - (x0, y0)).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        masks.append(m)
    return int((masks[0] & masks[1]).sum())


def auto_checks(anns, w, h, tiny, overlap_thr):
    """자동 의심 항목 검사. 각 annotation의 'checks' 리스트를 채운다."""
    for a in anns:
        seg, (bx, by, bw, bh) = a["seg"], a["bbox"]
        if a["area"] < tiny:
            a["checks"].append("tiny")
        if len(seg) < 8:
            a["checks"].append("few_pts")
        xs, ys = seg[0::2], seg[1::2]
        if xs and (abs(min(xs) - bx) > 2 or abs(min(ys) - by) > 2
                   or abs(max(xs) - bx - bw) > 2 or abs(max(ys) - by - bh) > 2):
            a["checks"].append("bbox")
        if xs and (min(xs) < 0 or min(ys) < 0 or max(xs) > w or max(ys) > h):
            a["checks"].append("oob")
    for i, a in enumerate(anns):
        for b in anns[i + 1:]:
            small = max(1, min(a["area"], b["area"]))
            bi = _bbox_inter(a["bbox"], b["bbox"])
            if bi < overlap_thr * small:
                continue
            inter = _mask_inter(a["seg"], b["seg"], a["bbox"], b["bbox"]) if cv2 else bi
            if inter >= overlap_thr * small:
                for x in (a, b):
                    if "overlap" not in x["checks"]:
                        x["checks"].append("overlap")


def load_dataset(data_dir, tiny, overlap_thr):
    with open(os.path.join(data_dir, "_annotations.coco.json"), encoding="utf-8") as f:
        coco = json.load(f)
    img_dir = os.path.join(data_dir, "raw")
    if not os.path.isdir(img_dir):
        img_dir = data_dir
    by_img = {}
    for a in coco["annotations"]:
        segs = a.get("segmentation") or []
        seg = segs[0] if segs and isinstance(segs, list) else []
        rec = {"id": a["id"], "cat": a["category_id"], "bbox": a["bbox"],
               "area": a.get("area", 0), "seg": [round(v, 1) for v in seg], "checks": []}
        if isinstance(segs, list) and len(segs) > 1:
            rec["checks"].append("multi_part")
        by_img.setdefault(a["image_id"], []).append(rec)
    images = []
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        auto_checks(anns, im["width"], im["height"], tiny, overlap_thr)
        parts = im["file_name"].split("__")
        checks = sorted({c for a in anns for c in a["checks"]})
        if not anns:
            checks.append("no_ann")
        if not os.path.isfile(os.path.join(img_dir, im["file_name"])):
            checks.append("no_file")
        images.append({"id": im["id"], "file": im["file_name"], "w": im["width"],
                       "h": im["height"], "cluster": parts[0] if len(parts) > 1 else "",
                       "session": parts[1] if len(parts) > 2 else "",
                       "anns": anns, "checks": checks})
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    return {"name": os.path.basename(os.path.normpath(data_dir)),
            "categories": cats, "images": images}, img_dir


class Store:
    """검수 결과 저장소 (file_name -> 레코드). 변경 즉시 원자적으로 디스크에 기록."""

    def __init__(self, path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)

    def set(self, file, rec):
        with self.lock:
            empty = not (rec.get("status") or rec.get("tags") or rec.get("memo")
                         or rec.get("flagged") or rec.get("missing"))
            if empty:
                self.data.pop(file, None)
            else:
                rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.data[file] = rec
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)


def find_data_dir(path):
    """_annotations.coco.json 이 있는 폴더를 찾는다. zip을 푼 바깥 폴더를 줘도 두 단계 아래까지 찾아본다."""
    path = os.path.abspath(os.path.expanduser(path.strip().strip('"\'')))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise ValueError("폴더가 없습니다: %s" % path)
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != "__MACOSX" and not d.startswith("."))
        if "_annotations.coco.json" in files:
            return root
        if root.count(os.sep) - path.count(os.sep) >= 2:
            dirs[:] = []
    raise ValueError("_annotations.coco.json 을 찾을 수 없습니다: %s" % path)


class Session:
    """현재 열려 있는 데이터셋. 뷰어 화면에서 경로를 바꾸면 통째로 교체된다."""

    def __init__(self, tiny, overlap):
        self.tiny, self.overlap, self.lock = tiny, overlap, threading.Lock()
        self.data_dir = None
        self.payload, self.files, self.img_dir, self.store, self.dataset = None, set(), None, None, None

    def open(self, path):
        data_dir = find_data_dir(path)
        dataset, img_dir = load_dataset(data_dir, self.tiny, self.overlap)
        store = Store(os.path.join(data_dir, "review_results.json"))
        dataset["path"] = data_dir
        payload = gzip.compress(json.dumps(dataset, separators=(",", ":")).encode(), 5)
        with self.lock:
            self.data_dir, self.dataset, self.img_dir, self.store = data_dir, dataset, img_dir, store
            self.files, self.payload = {im["file"] for im in dataset["images"]}, payload
        n_ann = sum(len(i["anns"]) for i in dataset["images"])
        print("열림: %s\n  이미지 %d장 / 라벨 %d개 / 자동검사 의심 이미지 %d장 / 기존 검수 기록 %d건"
              % (data_dir, len(dataset["images"]), n_ann,
                 sum(1 for i in dataset["images"] if i["checks"]), len(store.data)), flush=True)
        return data_dir

    def snapshot(self):
        with self.lock:
            return self.dataset, self.img_dir, self.store, self.files, self.payload


def make_handler(session):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200, extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json", code,
                       [("Cache-Control", "no-store")])

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            dataset, img_dir, store, files, payload = session.snapshot()
            try:
                if path == "/":
                    with open(os.path.join(HERE, "index.html"), "rb") as f:
                        self._send(f.read(), "text/html; charset=utf-8",
                                   extra=[("Cache-Control", "no-store")])
                elif path == "/api/info":
                    self._json({"path": session.data_dir})
                elif dataset is None and path.startswith("/api/"):
                    self._json({"error": "열린 데이터셋이 없습니다"}, 409)
                elif path == "/api/data":
                    self._send(payload, "application/json",
                               extra=[("Content-Encoding", "gzip"), ("Cache-Control", "no-store")])
                elif path == "/api/review":
                    with store.lock:
                        body = json.dumps(store.data, ensure_ascii=False).encode()
                    self._send(body, "application/json", extra=[("Cache-Control", "no-store")])
                elif path == "/api/export.csv":
                    self._send(self._csv(dataset, store), "text/csv; charset=utf-8", extra=[
                        ("Content-Disposition", 'attachment; filename="review_%s_%s.csv"' % (
                            os.path.basename(session.data_dir), time.strftime("%Y%m%d_%H%M%S"))),
                        ("Cache-Control", "no-store")])
                elif path.startswith("/img/") and path[5:] in files:
                    fp = os.path.join(img_dir, path[5:])
                    with open(fp, "rb") as f:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Content-Length", str(os.fstat(f.fileno()).st_size))
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        shutil.copyfileobj(f, self.wfile)
                else:
                    self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except FileNotFoundError:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            route = urlparse(self.path).path
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except (ValueError, TypeError):
                return self._json({"error": "bad request"}, 400)
            if route == "/api/open":
                try:
                    return self._json({"path": session.open(str(body.get("path", "")))})
                except (ValueError, KeyError, OSError) as e:
                    return self._json({"error": str(e)}, 400)
            if route != "/api/review":
                return self._send(b"not found", "text/plain", 404)
            dataset, _, store, files, _ = session.snapshot()
            try:
                file, rec = body["file"], body["rec"]
                if (dataset is None or body.get("dataset") not in (None, session.data_dir)
                        or file not in files or rec.get("status", "") not in STATUSES + ("",)):
                    raise ValueError("bad request")
                store.set(file, {
                    "status": rec.get("status", ""),
                    "tags": [str(t) for t in rec.get("tags", [])],
                    "memo": str(rec.get("memo", "")),
                    "flagged": [int(i) for i in rec.get("flagged", [])],
                    "missing": [[round(float(x)), round(float(y))] for x, y in rec.get("missing", [])],
                })
                self._json({"ok": True})
            except (ValueError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, 400)

        @staticmethod
        def _csv(dataset, store):
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["file_name", "status", "tags", "memo", "flagged_annotation_ids",
                        "flagged_bboxes_xywh", "missing_points_xy", "annotation_count",
                        "auto_checks", "updated"])
            with store.lock:
                data = json.loads(json.dumps(store.data))
            for im in dataset["images"]:
                r = data.get(im["file"], {})
                bbox = {a["id"]: a["bbox"] for a in im["anns"]}
                flagged = r.get("flagged", [])
                w.writerow([
                    im["file"], r.get("status", ""), "|".join(r.get("tags", [])), r.get("memo", ""),
                    "|".join(map(str, flagged)),
                    "|".join(",".join(map(str, bbox[i])) for i in flagged if i in bbox),
                    "|".join("%d,%d" % tuple(p) for p in r.get("missing", [])),
                    len(im["anns"]), "|".join(im["checks"]), r.get("updated", "")])
            return buf.getvalue().encode("utf-8-sig")  # Excel 한글 호환

    return H


def main():
    ap = argparse.ArgumentParser(description="COCO 라벨 검수 뷰어")
    ap.add_argument("data", nargs="?", help="데이터셋 폴더 (생략하면 뷰어 화면에서 입력)")
    ap.add_argument("--data", dest="data_opt", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="127.0.0.1", help="다른 PC에서 접속하려면 0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tiny", type=float, default=300, help="이 면적(px²) 미만이면 '작은 영역' 의심")
    ap.add_argument("--overlap", type=float, default=0.8,
                    help="교집합/작은쪽 면적이 이 비율 이상이면 '중복/겹침' 의심")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    session = Session(args.tiny, args.overlap)
    path = args.data or args.data_opt
    if path:
        try:
            session.open(path)
        except ValueError as e:
            print("⚠ %s — 뷰어 화면에서 경로를 다시 입력하세요." % e)

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    url = "http://%s:%d/" % ("127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port)
    print("뷰어: %s   (종료: Ctrl+C)" % url, flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
