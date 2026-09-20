# -*- coding: utf-8 -*-
"""手势模型训练器（b+ 路线）：21 点合成样本 → sklearn 随机森林 → gesture-model.pkl。
类型: open(张掌)/fist/one(食指)/v/ok/none(减负类)，增强: 旋转±18° / 缩放0.88-1.12 /
噪声0-0.02 / 左手镜像 / 手型乱机(noise)。特征: 标准化 21 点(腕原点)+掌位移+d48+d812+拇指距。
用法: python train_gesture_model.py
"""
import math, os, pickle, sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gesture_tests as gt   # 复用同目录合成手生成器(gt.make_hand)

# 特征函数与 gesture-command.py 同源（head-exec 提取）
def load_features():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture-command.py"),
               encoding="utf-8").read()
    head = src.split('print("摄像头启动中')[0]
    ns = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture-command.py")}
    exec(compile(head, "<gc>", "exec"), ns)
    return ns["to_features"]

to_features = load_features()

rng = np.random.default_rng(20260825)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture-model.pkl")

SPECS = {
    "open": ({"d": 0.0}, None),
    "fist": ({"d": 1.0}, None),
    "v": ({"d": 1.0, 8: 0.0, 12: 0.0}, None),
    "one": ({"d": 1.0, 8: 0.0}, None),
    "ok": ({"d": 0.25, 8: 0.15, 12: 0.1}, "together"),
}


def to_features(hand):
    lm = np.array([[p.x, p.y] for p in hand.landmark])
    wrist = lm[0]
    lm = lm - wrist                       # 腕为原点
    span = np.max([np.linalg.norm(lm[i]) for i in range(21)]) + 1e-9
    lm = lm / span                        # 手部跨度归一
    palm = np.mean([lm[i] for i in (0, 1, 2, 5, 9, 13, 17)], axis=0) if False else \
        np.mean([lm[i] for i in (0, 1, 2, 5, 9, 13, 17)], axis=0)
    kv = dict(d48=np.linalg.norm(lm[4] - lm[8]) / span if False else 0.0,
              thumb_out=np.linalg.norm(lm[4] - palm) / span if False else 0.0)
    # 上面划线保留(span 已归一,直接用归一坐标)
    d48 = float(np.linalg.norm(lm[4] - lm[8]))
    d812 = float(np.linalg.norm(lm[8] - lm[12]))
    thumb_out = float(np.linalg.norm(lm[4] - palm))
    feats = list(lm.flatten()) + [d48, d812, thumb_out, float(palm[0]), float(palm[1])]
    return feats


def gen_all():
    X, y = [], []
    def add(hand, label):
        # 左手镜像 + 标准化特征
        X.append(to_features(hand))
        y.append(label)
    N = 4000
    for label, (curl, thumb) in SPECS.items():
        for _ in range(N):
            rot = rng.uniform(-18, 18); scale = rng.uniform(0.88, 1.12)
            noise = rng.uniform(0.0, 0.02)
            tm = thumb or rng.choice(["natural", "inward"])
            hand = gt.make_hand(curl, thumb_mode=tm,
                                d48=rng.uniform(0.015, 0.09) if label == "ok" else None,   # 实况校准: 真机OK分0.67偏软,扩捏合松紧样本
                                rot=rot, scale=scale, noise=noise)
            add(hand, label)
            if rng.random() < 0.5:      # 镜像(左手)
                h2 = gt.make_hand(curl, thumb_mode=tm,
                                  d48=None, rot=-rot, scale=scale, noise=noise)
                add(h2, label)
        print(f"generated {label}: {len(X)}")
    # none 类: 随机乱手(任意卷度+拇指乱摆) × 8000
    for _ in range(8000):
        curl = {8: rng.uniform(0, 1), 12: rng.uniform(0, 1),
                16: rng.uniform(0, 1), 20: rng.uniform(0, 1)}
        hand = gt.make_hand(curl, thumb_mode=rng.choice(["inward", "natural"]),
                            rot=rng.uniform(-20, 20), scale=rng.uniform(0.85, 1.15),
                            noise=rng.uniform(0, 0.025))
        add(hand, "none")
    print("generated none:", len(X))
    return np.array(X), np.array(y)


def main():
    X, y = gen_all()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=7)
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         random_state=7)   # 直方图梯度提升: 推断~几ms, RF 300棵实测90ms太慢
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"--- 模型测试集准确率: {acc:.2%} | 训练 {len(Xtr)} 测试 {len(Xte)}")
    print("混淆矩阵(行=真实 列=预测):")
    lab = ["none", "v", "one", "fist", "ok", "open"]
    print(confusion_matrix(yte, clf.predict(Xte), labels=lab))
    joblib.dump(clf, OUT)
    print(f"SAVED: {OUT}")


if __name__ == "__main__":
    main()
