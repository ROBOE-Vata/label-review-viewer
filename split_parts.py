#!/usr/bin/env python3
"""데이터셋을 N개(기본 3개)로 나눠 검수자별 폴더를 만든다.

사용법:
    python3 split_parts.py <데이터셋 폴더>                 # 3등분, 원본 순서대로 1000장씩
    python3 split_parts.py <데이터셋 폴더> --size 500       # 500장씩
    python3 split_parts.py <데이터셋 폴더> --parts 4 [--out <dir>] [--copy]

각 폴더는 원본과 같은 구조(_annotations.coco.json + raw/)라서 그대로 뷰어로 열 수 있다.
같은 장면(연속 프레임)이 한 사람에게 가도록 원본 순서를 유지한 채 앞에서부터 자른다.
이미지는 하드링크로 넣어 디스크를 추가로 쓰지 않는다 (다른 디스크이거나 --copy 면 실제 복사).
"""
import argparse
import json
import math
import os

from export_split import place
from viewer import find_data_dir


def main():
    ap = argparse.ArgumentParser(description="데이터셋을 검수자별로 나누기")
    ap.add_argument("data", help="데이터셋 폴더")
    ap.add_argument("--parts", type=int, default=3, help="나눌 개수 (기본 3)")
    ap.add_argument("--size", type=int, help="폴더당 장수 (지정하면 --parts 대신 사용)")
    ap.add_argument("--out", help="출력 폴더 (기본: <데이터셋 폴더>_parts)")
    ap.add_argument("--copy", action="store_true", help="하드링크 대신 이미지를 실제로 복사")
    args = ap.parse_args()

    data = find_data_dir(args.data)
    name = os.path.basename(data)
    out = os.path.abspath(args.out) if args.out else data + "_parts"
    sub_dir = "raw" if os.path.isdir(os.path.join(data, "raw")) else ""
    with open(os.path.join(data, "_annotations.coco.json"), encoding="utf-8") as f:
        coco = json.load(f)
    images = coco["images"]
    size = args.size or math.ceil(len(images) / args.parts)
    chunks = [images[i:i + size] for i in range(0, len(images), size)]
    if os.path.exists(out):
        raise SystemExit("이미 있습니다: %s\n지우거나 --out 으로 다른 폴더를 지정하세요." % out)

    total = 0
    for k, chunk in enumerate(chunks, 1):
        ids = {im["id"] for im in chunk}
        sub = dict(coco, images=chunk, annotations=[a for a in coco["annotations"] if a["image_id"] in ids])
        dst = os.path.join(out, "%s_part%d_%dea" % (name, k, len(chunk)))
        os.makedirs(os.path.join(dst, sub_dir))
        for im in chunk:
            place(os.path.join(data, sub_dir, im["file_name"]), os.path.join(dst, sub_dir, im["file_name"]), args.copy)
        with open(os.path.join(dst, "_annotations.coco.json"), "w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False)
        total += len(sub["annotations"])
        print("part%d: 이미지 %d장, 라벨 %d개  (%s ~ %s)\n  -> %s" % (
            k, len(chunk), len(sub["annotations"]), chunk[0]["file_name"], chunk[-1]["file_name"], dst))
    print("합계: 이미지 %d장 / 라벨 %d개 (원본 %d / %d)" % (
        sum(map(len, chunks)), total, len(images), len(coco["annotations"])))


if __name__ == "__main__":
    main()
