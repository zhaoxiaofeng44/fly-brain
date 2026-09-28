# 果蝇全脑 × 具身游乐场（Fly Brain × Embodied Playground）

在 Apple Silicon 上本地运行的真实果蝇全脑仿真，并让这颗大脑实时驱动一只有身体、
有腿、有翅膀的果蝇（NeuroMechFly v2 / MuJoCo），完成栖息、飞行、觅食的完整行为环。

```
┌─────────────────────────┐   尖峰读出    ┌──────────────────────────────┐
│  Metal GPU 全脑仿真器     │ ───────────▶ │  具身身体 (flygym/MuJoCo)     │
│  138,639 神经元           │   P9 → CPG   │  42 关节 CPG 步态 + 引导飞行   │
│  15,091,983 突触          │ ◀─────────── │  糖 GRN / P9 刺激表达          │
│  3.6–6.4× 实时 (M5 Pro)   │   感官刺激    └──────────────────────────────┘
└─────────────────────────┘      50ms 闭环 tick
```

基于 [Shiu et al. 2023](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1)
的 FlyWire 连接组 LIF 模型（上游说明见 [README_upstream.md](README_upstream.md)）。

## 快速开始

前置：macOS + Apple Silicon、[uv](https://docs.astral.sh/uv/)、Xcode 命令行工具；
连接组数据 `data/2025_Connectivity_783.parquet` 需已就位（随上游项目分发）。

```bash
./deploy/setup_all.sh                    # 三个环境一次装好
./deploy/run_headless_verify.sh 40       # 无头冒烟：完整脑驱动行为环
./deploy/run_playground.sh               # 交互窗口（默认飞行游乐场）
./deploy/run_brain_sim.sh validate       # 单独验证 Metal 脑仿真
```

交互键位（点击 MuJoCo 窗口）：**C** 投糖果 **S** 挥棍驱赶 **F** 强制飞行 **Esc** 退出。

## 行为诚实分层

| 层 | 来源 |
|---|---|
| 速度 / 食欲 | 真连接组：P9 放电率 → CPG 频率；进食触发糖 GRN → P9 抑制（饱食） |
| 起飞 / 归巢决策 | 饥饿（内部模型，连接组无饥饿环路）表达为 P9 刺激；**转换只认实测脑输出**：唤醒度 >47Hz 出击、饱食静默 <23Hz 归巢，无定时器 |
| 转向 | 嗅觉反射算意图 → 左右 P9 差分刺激 → CPG 转向从实测 L/R 放电差反推 |
| 飞行 | 脚本化代理（连接组无飞行环路、翅膀为装饰）：贝塞尔引导力 + 运动学姿态/翅膀；何时飞由脑决定，怎么飞不是 |
| 逃跑 / 翻正 | 硬反射（棍子威胁、腹面着地弹跳） |

## 仓库结构

```
code/            上游 CPU 全脑仿真（Brian2 基准，README_upstream.md）
metal_sim/       Metal GPU 全脑仿真器（自研）
  brain.metal / metal_sim.mm    Metal 内核 + C++ 编排器
  brain_bridge.py               Python 管道桥（run / rate / spikes 协议）
  export_connectivity.py        连接组 → CSR 二进制（生成 data/，~116MB，不入库）
  validate_metal.py             与 CPU 模型对账的统计验证
  brain_viewer.py               实时栅格图查看器
embodied/        具身游乐场（演化路径：v1 投糖/挥棍 → nature 自主觅食
                 → flight 树冠飞行 + 脑驱动决策）
  brain_body_coupling.py        脑-身耦合最小演示（P9 → CPG 离线回放）
deploy/          环境部署与运行脚本
docs/            架构说明与踩坑笔记
data/            FlyWire 连接组数据（2025_Completeness / Connectivity）
```

## 已验证指标（M5 Pro）

- Metal 全脑：3.6–6.4× 实时，与 CPU 模型放电统计对账通过
- 脑驱动行为环：无头 110s 模拟内 4 次唤醒出击 / 4 次饱食归巢 / 吃 3 颗糖，
  周期约 22.5s，全程无定时器
- 飞行姿态：全程 2–8° 水平（运动学姿态保持），0 翻正 / 0 摔落
