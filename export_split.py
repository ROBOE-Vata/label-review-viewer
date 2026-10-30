#!/usr/bin/env python3
"""검수 결과(review_results.json)에 따라 데이터셋을 OK / NG 폴더로 나눠 저장한다.

사용법:
    python3 export_split.py <데이터셋 폴더>               # -> <데이터셋 폴더>_reviewed/{ok,ng}
    python3 export_split.py <데이터셋 폴더> --overwrite   # 판정을 바꾼 뒤 다시 만들 때
    python3 export_split.py <데이터셋 폴더> --out <dir> [--copy]

ok/, ng/ 는 각각 원본과 같은 구조(_annotations.coco.json + 이미지)이고 id도 원본 그대로다.
이미지는 하드링크로 넣어 디스크를 추가로 쓰지 않는다 (다른 디스크이거나 --copy 면 실제 복사).
"""
import argparse
import collections
import json
import os
import shutil
import time

from viewer import find_data_dir

SETS = ("ok", "ng")


def place(src, dst, copy):
    if not copy:
        try:
            os.link(src, dst)
            return "하드링크"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "복사"


def main():
    ap = argparse.ArgumentParser(description="검수 결과로 OK / NG 데이터셋 분리")
    ap.add_argument("data", help="검수한 데이터셋 폴더 (review_results.json 이 있는 곳)")
    ap.add_argument("--out", help="출력 폴더 (기본: <data>_reviewed)")
    ap.add_argument("--copy", action="store_true", help="하드링크 대신 이미지를 실제로 복사")
    ap.add_argument("--overwrite", action="store_true", help="이전에 만든 ok/, ng/ 를 지우고 다시 만든다")
    args = ap.parse_args()

    data = find_data_dir(args.data)
    out = os.path.abspath(args.out) if args.out else data + "_reviewed"
    sub_dir = "raw" if os.path.isdir(os.path.join(data, "raw")) else ""
    with open(os.path.join(data, "_annotations.coco.json"), encoding="utf-8") as f:
        coco = json.load(f)
    with open(os.path.join(data, "review_results.json"), encoding="utf-8") as f:
        review = json.load(f)

    for s in SETS:  # 이전 결과와 섞이지 않게
        p = os.path.join(out, s)
        if os.path.exists(p):
            if not args.overwrite:
                raise SystemExit("이미 있습니다: %s\n다시 만들려면 --overwrite 를 붙이세요." % p)
            shutil.rmtree(p)

    status = {im["file_name"]: review.get(im["file_name"], {}).get("status", "") for im in coco["images"]}
    lines = ["원본: %s" % data,
             "검수 기록 기준: %s (마지막 판정 저장 %s)" % (
                 time.strftime("%Y-%m-%d %H:%M:%S"), max((r.get("updated", "") for r in review.values()), default="-"))]
    for s in SETS:
        images = [im for im in coco["images"] if status[im["file_name"]] == s]
        ids = {im["id"] for im in images}
        sub = dict(coco, images=images, annotations=[a for a in coco["annotations"] if a["image_id"] in ids])
        dst = os.path.join(out, s)
        os.makedirs(os.path.join(dst, sub_dir))
        how = collections.Counter(
            place(os.path.join(data, sub_dir, im["file_name"]), os.path.join(dst, sub_dir, im["file_name"]), args.copy)
            for im in images)
        with open(os.path.join(dst, "_annotations.coco.json"), "w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False)
        with open(os.path.join(out, s + "_files.txt"), "w", encoding="utf-8") as f:
            f.write("".join(im["file_name"] + "\n" for im in images))
        lines.append("%s: 이미지 %d장, 라벨 %d개  (%s)  -> %s" % (
            s.upper(), len(images), len(sub["annotations"]),
            ", ".join("%s %d" % kv for kv in how.items()) or "-", dst))

    rest = collections.Counter(s or "미검수" for s in status.values() if s not in SETS)
    if rest:
        lines.append("제외(OK/NG 아님): " + ", ".join("%s %d장" % kv for kv in rest.items()))
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
