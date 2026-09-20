# -*- coding: utf-8 -*-
"""合成手单元回归测试 —— 对 gesture-command.py 的 finger_stats/classify 无人值守自查。
通过 head-exec 只加载函数段（不启动摄像头主循环）。
生成规范手势 + 多角度/缩放/噪声变异，断言分类准确率过高线。
用法: python gesture-tests.py   (输出每类结果 + PASS/FAIL)
"""
import math
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

FING = {8: (5, 6, 7), 12: (9, 10, 11), 16: (13, 14, 15), 20: (17, 18, 19)}
ANGLES = {8: -34, 12: -10, 16: 12, 20: 34}       # 指根基准角度(度)
FINGER_NAMES = {8: "idx", 12: "mid", 16: "ring", 20: "pinky"}


def make_hand(curl, thumb_mode="natural", d48=None, rot=0.0, scale=1.0, noise=0.0):
    """curl: {tip_idx: 0..1 卷曲度}（缺省用 'd' 全指值）; 返回 21 点 FakeHand(landmark: x,y)"""
    w = np.array([0.0, 0.0])
    rx = np.array([[math.cos(math.radians(rot)), -math.sin(math.radians(rot))],
                   [math.sin(math.radians(rot)), math.cos(math.radians(rot))]])
    lm = np.zeros((21, 2))
    for tip, (mcp, pip, dip) in FING.items():
        c = curl.get(tip, curl.get("d", 0.0))
        base = math.radians(ANGLES[tip])
        d = np.array([math.sin(base), -math.cos(base)]) @ rx.T
        m = w + d * 0.42 * scale
        # 卷曲: 方向往掌心折(朝下), 长度缩短
        cd = (d * (1 - c) + np.array([0.0, 0.85]) * c)
        cd = cd / (np.linalg.norm(cd) + 1e-9)
        p2 = m + cd * 0.23 * scale * (1 - c * 0.30)
        p3 = p2 + cd * 0.21 * scale * (1 - c * 0.45)
        lm[mcp], lm[pip], lm[dip], lm[tip] = m, p2, p2, p3
    # 拇指(粗): cmc 1 / mcp 2 / tip 4 简易链
    thumb_dir = {"natural": np.array([0.32, 0.10]), "inward": np.array([0.06, 0.34]),
                 "together": np.array([0.32, 0.34])}[thumb_mode] @ rx.T
    lm[1] = w + np.array([0.14, 0.06]) @ rx.T
    lm[2] = lm[1] + np.array([0.12, 0.05]) @ rx.T
    lm[4] = lm[2] + thumb_dir * 0.16
    if d48 is not None:
        v = np.array([1.0, 0.35]); v = v / np.linalg.norm(v)
        lm[4] = lm[8] + v * d48
    if noise:
        lm = lm + np.random.normal(0, noise, (21, 2))
    class Pt:
        def __init__(self, p): self.x, self.y = float(p[0]), float(p[1])
    class H:
        def __init__(self, m):
            self.landmark = [Pt(m[i]) for i in range(21)]
    return H(lm)


# ---- head-exec 加载被测函数 ----
def load_module():
    src = open(r"D:\DS\工具\gesture-command.py", encoding="utf-8").read()
    head = src.split('print("摄像头启动中')[0]
    ns = {"__file__": r"D:\DS\工具\gesture-command.py"}
    exec(compile(head, "<gesture-head>", "exec"), ns)
    return ns["finger_stats"], ns["classify"]


finger_stats, classify = load_module()

HAND_SPECS = {
    "open": ({"d": 0.0}, 5),
    "fist": ({"d": 1.0}, 0),
    "v":    ({"d": 1.0, 8: 0.0, 12: 0.0}, 2),      # 食中伸其余卷 = 2 指伸 + 拇指不算
    "one":  ({"d": 1.0, 8: 0.0}, 1),
    "ok":   ({"d": 0.25, 8: 0.15, 12: 0.1}, "ok"),
    "hair": ({"d": 0.4, 8: 0.25, 12: 0.5, 16: 0.7, 20: 0.8}, None),  # 反例：撩发(不触发)
}


def run():
    total, ok = 0, 0
    print(f"{'类':6} {'正确':>5} {'总':>4} {'结果'}")
    for name, (curl, expect) in HAND_SPECS.items():
        n_pass, n_tot = 0, 0
        for rot in (0, 12, -15):
            for scale in (0.9, 1.0, 1.1):
                for noise in (0.0, 0.008, 0.016):
                    n_tot += 1
                    hand = make_hand(curl, thumb_mode="together" if name == "ok" else
                                     ("inward" if name == "hair" else "natural"),
                                     d48=0.03 if name == "ok" else None,
                                     rot=rot, scale=scale, noise=noise)
                    got, _ = classify(hand)
                    want = expect
                    got_ok = (got == want) if want is not None else (got != "ok")
                    if got_ok:
                        n_pass += 1
                    else:
                        print(f"  [x] {name} rot={rot} s={scale} n={noise}: got={got} want={want}")
        total += n_tot; ok += n_pass
        print(f"{name:8} {n_pass:>4}/{n_tot:>4}  {'PASS' if n_pass == n_tot else 'FAIL'}")
    rate = ok / total
    print(f"--- 总准确率 {ok}/{total} = {rate:.1%} {'✅ PASS' if rate >= 0.90 else '❌ FAIL(需调)'}")
    return rate >= 0.90


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
