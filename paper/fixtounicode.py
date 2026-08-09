#!/usr/bin/env python3
"""把 PDF ToUnicode CMap 中被誤映射到 U+2010/U+2011 的字符改回 U+002D。

成因：Noto CJK 字型的 cmap 讓 U+002D / U+2010 / U+2011 共用同一字形，
xdvipdfmx 反解 ToUnicode 時挑到非 ASCII 的碼位，導致 PDF 內的網址與
技術名詞複製貼上後失效。此腳本在編譯後修正映射，不改動任何字形。
"""
import re, sys
import pikepdf

TARGETS = {"2010": "002D", "2011": "002D", "2012": "002D", "FE63": "002D"}

def fix_cmap(data: bytes) -> tuple[bytes, int]:
    text = data.decode("latin-1")
    n = 0
    def repl_pair(m):
        nonlocal n
        src, dst = m.group(1), m.group(2).upper()
        if dst in TARGETS:
            n += 1
            return f"<{src}> <{TARGETS[dst]}>"
        return m.group(0)
    text = re.sub(r"<([0-9A-Fa-f]{2,8})>\s*<([0-9A-Fa-f]{4})>", repl_pair, text)
    return text.encode("latin-1"), n

def main(path):
    total = 0
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        for obj in pdf.objects:
            try:
                if not isinstance(obj, pikepdf.Stream):
                    continue
                raw = bytes(obj.read_bytes())
            except Exception:
                continue
            if b"beginbfchar" not in raw and b"beginbfrange" not in raw:
                continue
            new, n = fix_cmap(raw)
            if n:
                obj.write(new)
                total += n
        pdf.save()
    print(f"{path}: 修正 {total} 個 ToUnicode 映射")
    return total

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
