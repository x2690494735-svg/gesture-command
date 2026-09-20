# gesture-command

摄像头隔空比手势 + 语音，直接往编辑器窗口打字并回车。手不动键鼠。

## 技术

MediaPipe GestureRecognizer → 自研规则链（方向向量法 cos > 0.55；OK 捏合双条件）
→ sklearn HistGradientBoosting 分类（注释里记了取舍：随机森林 300 棵实测 90ms 太慢，弃用）
→ faster-whisper 转写语音 → ctypes.windll 控制目标窗口。

## 文件

| 文件 | 说明 |
|---|---|
| `gesture-command.py` | 主程序：手势识别 + 打字 |
| `gesture-voice.py` | 语音通道 |
| `train_gesture_model.py` | 训练脚本 |
| `gesture_tests.py` | 测试 |
| `gesture-model.pkl` | 训练好的模型（1.6MB） |
| `gesture-debug.csv` | 实时判定写盘，用于误判取证（不靠肉眼调参） |

6 类手势，抗抖用 6 帧投票，迭代了 7 版。

MIT
