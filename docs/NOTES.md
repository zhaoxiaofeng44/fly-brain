# 踩坑笔记（macOS / Apple Silicon）

按主题整理的本项目实战坑与结论，修问题前先查这里。

## 环境

- **两套 venv 互不相通**：CPU 脑仿真 `.venv`（py3.11 + **numpy<2**，Brian2 硬约束）；
  具身 `.venv-body`（py3.12 + flygym 2.1.0）。读连接组 parquet 都要手动补 pyarrow。
- **mjpython 双坑**（GUI 必须 `mjpython` 跑）：① uv 装的 Python 要加
  `DYLD_FALLBACK_LIBRARY_PATH=<uv python prefix>/lib`，否则 app bundle 找不到
  libpython3.12.dylib；② 从无人值守的 shell 启动时，显示器睡眠会让 GLFW
  `_glfwGetVideoModeCocoa` 段错误（exit 139）或 glfwInit 挂死——先 `caffeinate -u -t 5`
  唤醒屏幕，或直接在用户终端里跑。
- 后台跑脚本必须 `PYTHONUNBUFFERED=1`（否则 stdout 块缓冲，日志全憋着）。
- 渲染用 macOS 默认 GL 后端，**别设 MUJOCO_GL=egl**（那是 Linux 的）。

## flygym / MuJoCo（模型单位 = mm、mg）

- **MjSpec 吃 body**：`add_fly` 的 attach+compile 会吞掉 worldbody 下第一个 body——
  先建一个牺牲 anchor，再建所有场景 body，最后 add_fly。
- **碰撞全靠显式 pair**：flygym 苍蝇几何 `conaffinity=0`，只跟显式 contact pair 碰。
  场景（树/石/停机坪）要仿照其地面写法逐对 `add_pair(苍蝇 geom × 场景 geom)`，
  否则全程穿模。
- **质量必须按名字过滤**：`model.body_mass.sum()` 含整片场景（4.3e6），
  苍蝇本体只有 1.019e-3——引导力按总质量算会力爆 → MuJoCo **静默自动 reset**
  （data.time 倒回、瞬移回出生点）。
- **自由关节寻址**：场景 body 排在苍蝇前面，`qvel[:3]` 不是自由关节；
  用 `model.jnt_dofadr[mj_name2id(..., "live_fly")]` 找 dofadr。
- **模型单位陷阱**：m·g≈10（不是 0.01）。引导力施加在胸段质心（xfrc 力作用点 =
  body 质心 xipos，不是 frame 原点），r×F 寄生力矩 ≈3.6 vs 果蝇转动惯量 ≈2.6e-3
  → 悬空 0.1s 翻滚到 ω≈140 rad/s。**力矩 PD 赢不了，飞行姿态用运动学保持**
  （每物理步直接积分自由关节四元数，与翅膀同属诚实标注的运动学代理）。
- 引导力要按**物理步**（1e-4s）重算 PD，50ms 一次会震荡。
- 翅膀铰链是后加的：翅膀 body mass/inertia 置 1e-12 消反作用力。
- mocap 道具位置要在 `sim.warmup()` **之后**写，否则 reset 会重置。
- 翻身弹跳：qvel z=350 + 角速度 ±40 才有效（v=30 在 g=9810 下只跳 0.05mm）。
- 栖息要 amp=0 + 六脚粘附常 True（相位式粘附会松爪滑落）；树冠顶加平停机坪
  （穹顶太滑）。

## Metal 脑仿真

- 单神经元 Poisson 噪声很大：对称刺激下 L/R 读数可差 ±30Hz——读出侧必须
  EMA 平滑 + 死区。
- P9 读出率与刺激率有抖动（50ms 窗口），EMA 0.6/0.4 后再进门控。
- `metal_sim/data/` 全部由 `export_connectivity.py` 从 parquet 生成（~116MB），
  可随时重建，不入库。

## 验证方法论（从翻车中学到的）

- 最小模型探针定坐标系：旋转 90° 的单体测 `mj_objectVelocity`（flg_local=0 是
  世界系）与 xfrc 力矩帧；偏心质量单体测 xfrc 力作用点（= xipos）；
  自由关节 qvel 角速度是 **body 系**。
- 高频现象会被低频日志混叠：0.05s 采样把 140rad/s 翻滚看成"稳定 70°"。
  怀疑姿态问题时用无头跑 + 每步 tilt 日志。
