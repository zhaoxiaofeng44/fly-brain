# 系统架构

## 三层结构

```
                    每 50ms 一个闭环 tick
┌────────────────────────────────────────────────────────────────┐
│ ① 感官表达层（身体 → 脑，刺激）                                   │
│    hunger(0→1, 30s) → P9 双侧刺激 8–95Hz                        │
│    转向意图 u → 左右 P9 差分 (base·(1∓0.8u))                     │
│    进食 → 糖 GRN 200Hz（连接组里真实的饱食通路）                   │
│    逃跑 → P9 150Hz 冲刺驱动                                      │
├────────────────────────────────────────────────────────────────┤
│ ② Metal 全脑仿真（每 tick 500 步 × 0.1ms）                        │
│    LIF 神经元 + CSR 稀疏突触，Metal Performance Shaders          │
│    协议：run <steps> / rate <group> <hz> / spikes / reset        │
│    群组：p9_left(1) p9_right(1) sugar(21)                        │
├────────────────────────────────────────────────────────────────┤
│ ③ 身体读出层（脑 → 身体）                                        │
│    p9 EMA → drive → CPG 频率 0.4–1.6×                            │
│    实测 L/R 放电差 → CPG 左右振幅不对称 → 转向                    │
│    实测唤醒度门控状态机：>47Hz 出击，饱食静默 <23Hz 归巢            │
└────────────────────────────────────────────────────────────────┘
```

## 模块

| 模块 | 文件 | 职责 |
|---|---|---|
| Metal 内核 | `metal_sim/brain.metal` | LIF 更新 + 稀疏突触传播（shared memory 环形缓冲） |
| 编排器 | `metal_sim/metal_sim.mm` | stdio 协议循环、刺激群组、尖峰流输出 |
| Python 桥 | `metal_sim/brain_bridge.py` | `MetalBrain`：Popen 封装（set_rate/run/drain_spikes） |
| 导出 | `metal_sim/export_connectivity.py` | parquet → CSR（w.f32/post.i32）+ meta.json |
| 验证 | `metal_sim/validate_metal.py` | 与 CPU Brian2 模型的放电率/相关系数对账 |
| 游乐场 | `embodied/interactive_playground_*.py` | MuJoCo 场景 + 50ms 闭环 + 键盘交互 |
| 部署 | `deploy/*.sh` | 环境一键装、交互/无头运行 |

## 演化路径（embodied/）

1. `interactive_playground.py` — 首个交互层：C 投糖（进食姿态）、S 挥棍（不对称振幅转身逃跑）
2. `interactive_playground_nature.py` — 自然场景（树/石/预置糖）+ 自主寻食，头测 6/6 吃完
3. `interactive_playground_flight.py` — 树冠栖息 + 树间/落地飞行 + **脑驱动决策**（当前主入口），
   `HEADLESS=1 SIM_LIMIT=n` 可无头复跑，`FLYDBG=1` 出姿态/接触调试
4. `brain_body_coupling.py` / `interactive_coupling.py` — 脑-身桥的离线/实时最小演示
