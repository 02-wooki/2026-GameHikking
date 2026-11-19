# src/gpt/build_sft_data.py
"""(근거 + 질문 → 답변) SFT 데이터 자동 생성.
- 근거는 RAG가 넘겨줄 청크와 같은 형태로 만든다 (태그 제거).
- 답변은 근거에서 그대로 복사 가능한 문장만 사용. 근거에 없으면 거절 문장.
- 검증 세트는 챔피언 단위로 분리 (처음 보는 챔피언 근거에서도 복사하는지 확인).
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from lol_tokenizer import load as load_tokenizer, format_chat

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sft"

SEED = 7
VAL_CHAMP_RATIO = 0.12
NEG_RATIO = 0.15
REFUSE = "제공된 근거 문서에서는 확인할 수 없습니다."
rng = random.Random(SEED)

BASE_H = re.compile(r"^=== (.+?) ===$")
PATCH_H = re.compile(r"^=== (.+?) \| patch_(\S+) ===$")
SKILL_RE = re.compile(r"^(기본 지속 효과|Q|W|E|R) - (.+?): (.+)$")
STAT_KEYS = ["기본 체력", "공격력", "방어력", "이동속도"]
SKILL_Q = ["{c} {s} 스킬 알려줘", "{c} {s} 뭐야?", "{c}의 {s} 설명해줘"]
STAT_Q = ["{c} {k} 얼마야?", "{c}의 {k} 알려줘"]
SLOT_NAMES = {"패시브": ["패시브", "기본 지속 효과"], "R": ["R", "궁극기"]}


def clean(text):
    """Data Dragon 마크업 제거 (추론 시 RAG 근거에도 같은 함수를 적용할 것)."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_blocks(text, header_re):
    blocks, cur = [], None
    for line in text.splitlines():
        m = header_re.match(line.strip())
        if m:
            cur = {"m": m, "lines": []}
            blocks.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    return blocks


def base_slot(s):
    return "패시브" if s == "기본 지속 효과" else s


def slot_of(line):
    label = line.split(": ", 1)[0]
    if label.startswith("기본 지속 효과"):
        return "패시브"
    m = re.match(r"^([QWER]) - ", label)
    return m.group(1) if m else label


def slot_name(slot):
    return rng.choice(SLOT_NAMES.get(slot, [slot]))


# ---------------- 원본 로드 ----------------
def load_base():
    text = (RAW / "champion_base_data.txt").read_text(encoding="utf-8")
    out = []
    for b in split_blocks(text, BASE_H):
        champ = b["m"].group(1).strip()
        lines = [clean(l) for l in b["lines"]]
        lines = [l for l in lines if l]
        info = {"champ": champ, "roles": None, "stats": {}, "skills": []}
        for l in lines:
            if l.startswith("역할군:"):
                info["roles"] = l.split(":", 1)[1].strip()
            elif l.startswith("기본 체력:"):
                for k, v in re.findall(r"([가-힣 ]+?): ([\d.]+)", l):
                    info["stats"][k.strip()] = v
            else:
                m = SKILL_RE.match(l)
                if m:
                    info["skills"].append({"slot": m.group(1), "line": l})
                elif info["skills"]:
                    info["skills"][-1]["line"] += " " + l
        info["evidence"] = f"=== {champ} ===\n" + "\n".join(lines)
        out.append(info)
    return out


def load_patch():
    text = (RAW / "chunks.txt").read_text(encoding="utf-8")
    out = []
    for b in split_blocks(text, PATCH_H):
        champ, patch = b["m"].group(1).strip(), b["m"].group(2).strip()
        lines = [clean(l) for l in b["lines"]]
        changes = [l for l in lines if l and not l.endswith("패치 변경사항:")]
        if not changes:
            continue
        title = f"{champ} {patch} 패치 변경사항:"
        evidence = f"=== {champ} | patch_{patch} ===\n{title}\n" + "\n".join(changes)
        out.append({"champ": champ, "patch": patch, "title": title,
                    "changes": changes, "evidence": evidence})
    return out


# ---------------- 샘플 생성 ----------------
def make_sample(kind, champ, evidence, question, answer):
    return {"kind": kind, "champion": champ, "evidence": evidence,
            "question": question, "answer": answer}


def base_samples(b):
    out, c, ev = [], b["champ"], b["evidence"]
    for sk in b["skills"]:
        slot = base_slot(sk["slot"])
        for tpl in rng.sample(SKILL_Q, 2):
            q = tpl.format(c=c, s=slot_name(slot))
            out.append(make_sample("base_skill", c, ev, q, f"{c} {sk['line']}"))
    for k in rng.sample(STAT_KEYS, 2):
        if k in b["stats"]:
            q = rng.choice(STAT_Q).format(c=c, k=k)
            out.append(make_sample("base_stat", c, ev, q, f"{c}의 {k}은(는) {b['stats'][k]}입니다."))
    if b["roles"]:
        out.append(make_sample("base_role", c, ev, f"{c} 역할군 뭐야?", f"{c}의 역할군은 {b['roles']}입니다."))
    return out


def patch_samples(p):
    out, c, pt, ev = [], p["champ"], p["patch"], p["evidence"]
    full = p["title"] + "\n" + "\n".join(p["changes"])
    out.append(make_sample("patch_all", c, ev, f"{c} {pt} 패치에서 뭐 바뀌었어?", full))
    out.append(make_sample("patch_all_nopatch", c, ev, f"{c} 패치 변경사항 알려줘", full))

    by_slot = {}
    for l in p["changes"]:
        by_slot.setdefault(slot_of(l), []).append(l)
    for slot, ls in by_slot.items():
        q = f"{c} {pt} 패치 {slot_name(slot)} 변경점 알려줘"
        out.append(make_sample("patch_slot", c, ev, q, p["title"] + "\n" + "\n".join(ls)))

    seen = set()
    for l in p["changes"]:
        parts = l.split(": ", 2)
        if len(parts) < 3 or "⇒" not in parts[2]:
            continue
        label, field, vals = parts
        q = f"{c} {pt} 패치 {slot_name(slot_of(l))} {field} 어떻게 바뀌었어?"
        if q in seen:
            continue
        seen.add(q)
        out.append(make_sample("patch_field", c, ev, q, f"{c} {pt} 패치 - {label}: {field}: {vals}"))
    return out


def negatives(base_pool, patch_pool, n):
    """근거에 답이 없는 경우 → 거절 문장 (RAG가 엉뚱한 청크를 가져온 상황 재현)."""
    out, tries = [], 0
    base_by = {b["champ"]: b for b in base_pool}
    while len(out) < n and tries < n * 20:
        tries += 1
        t = rng.choice(["base", "patch", "slot", "type"])
        if t == "base" and len(base_pool) >= 2:
            a, b = rng.sample(base_pool, 2)
            if not b["skills"]:
                continue
            sk = rng.choice(b["skills"])
            q = rng.choice(SKILL_Q).format(c=b["champ"], s=slot_name(base_slot(sk["slot"])))
            out.append(make_sample("neg_base", a["champ"], a["evidence"], q, REFUSE))
        elif t == "patch" and len(patch_pool) >= 2:
            a, b = rng.sample(patch_pool, 2)
            if (a["champ"], a["patch"]) == (b["champ"], b["patch"]):
                continue
            q = f"{b['champ']} {b['patch']} 패치에서 뭐 바뀌었어?"
            out.append(make_sample("neg_patch", a["champ"], a["evidence"], q, REFUSE))
        elif t == "slot" and patch_pool:
            p = rng.choice(patch_pool)
            have = {slot_of(l) for l in p["changes"]}
            missing = [s for s in ["Q", "W", "E", "R", "패시브"] if s not in have]
            if not missing:
                continue
            q = f"{p['champ']} {p['patch']} 패치 {slot_name(rng.choice(missing))} 변경점 알려줘"
            out.append(make_sample("neg_slot", p["champ"], p["evidence"], q, REFUSE))
        elif t == "type" and patch_pool:
            p = rng.choice(patch_pool)
            b = base_by.get(p["champ"])
            if not b:
                continue  # 근거=기본정보, 질문=패치노트
            q = f"{p['champ']} {p['patch']} 패치에서 뭐 바뀌었어?"
            out.append(make_sample("neg_type", p["champ"], b["evidence"], q, REFUSE))
    return out


# ---------------- 메인 ----------------
def report(name, data):
    lens = sorted(s["n_tokens"] for s in data)
    print(f"\n[{name}] 샘플 {len(data)}개")
    print("  종류:", dict(Counter(s["kind"] for s in data)))
    if lens:
        print(f"  토큰 길이 min/중앙/90%/최대: {lens[0]} / {lens[len(lens)//2]} / "
              f"{lens[int(len(lens)*0.9)]} / {lens[-1]}")
        for lim in (256, 384, 512, 768, 1024):
            print(f"  {lim}토큰 이하: {sum(l <= lim for l in lens) / len(lens) * 100:.1f}%")


def main():
    tok = load_tokenizer(str(ROOT / "tokenizer_out"))
    base, patch = load_base(), load_patch()
    print(f"기본정보 {len(base)}개 | 패치 청크 {len(patch)}개")

    champs = sorted({b["champ"] for b in base} | {p["champ"] for p in patch})
    val_champs = set(rng.sample(champs, max(1, int(len(champs) * VAL_CHAMP_RATIO))))
    print(f"챔피언 {len(champs)}명 중 검증용 홀드아웃 {len(val_champs)}명")

    splits = {}
    for name in ("train", "val"):
        in_val = name == "val"
        bp = [b for b in base if (b["champ"] in val_champs) == in_val]
        pp = [p for p in patch if (p["champ"] in val_champs) == in_val]
        pos = [s for b in bp for s in base_samples(b)] + [s for p in pp for s in patch_samples(p)]
        data = pos + negatives(bp, pp, int(len(pos) * NEG_RATIO))
        rng.shuffle(data)
        splits[name] = data

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with (OUT / f"sft_{name}.jsonl").open("w", encoding="utf-8") as f:
            for s in data:
                ev = s.pop("evidence")
                prompt = format_chat([{"role": "system", "content": ev},
                                      {"role": "user", "content": s["question"]}],
                                     add_generation_prompt=True)
                target = s["answer"] + "<|end|>"
                s["prompt"], s["target"] = prompt, target
                s["n_tokens"] = (len(tok.encode(prompt, add_special_tokens=False))
                                 + len(tok.encode(target, add_special_tokens=False)))
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        report(name, data)

    shown = set()
    for s in splits["train"]:
        if s["kind"] in ("base_skill", "patch_field", "neg_patch") and s["kind"] not in shown:
            shown.add(s["kind"])
            print(f"\n--- 예시: {s['kind']} ({s['n_tokens']}토큰) ---")
            print(s["prompt"][:300], "...")
            print("Q:", s["question"])
            print("A:", s["answer"])
    print(f"\n저장 완료: {OUT}")


if __name__ == "__main__":
    main()