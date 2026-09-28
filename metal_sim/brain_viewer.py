"""Live fly-brain spike visualizer (Metal GPU brain, real-time).

Left: scrolling spike raster heatmap — 138,639 neurons compressed into 512
      rows by index; each column is 50ms of simulated time.
Right: whole-brain firing rate (Hz) scrolling trace.
Bottom: P9 command-neuron rate bar + stimulation state.

Keys: S = toggle sugar-GRN stimulation (200Hz), P = toggle P9 stimulation
      (150Hz), R = reset brain state, Esc = quit.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path.home() / "fly-brain" / "metal_sim"))
from brain_bridge import MetalBrain  # noqa: E402

BINS = 512
COLS = 240          # 50ms per column -> 12s window
TICK_STEPS = 500    # 50ms sim per frame (brain dt = 0.1ms)
FRAME_MS = 50
DRIVE_NORM_MAX = 105.0  # P9 Hz at full stimulation (measured)

brain = MetalBrain()
print("GPU brain online:", brain.n, "neurons,", brain.n_syn, "synapses")

bin_of_neuron = (np.arange(brain.n, dtype=np.int64) * BINS // brain.n)
p9_idx = np.concatenate([brain.groups["p9_left"], brain.groups["p9_right"]])
sugar_bins = np.unique(bin_of_neuron[brain.groups["sugar"]])
p9_bins = np.unique(bin_of_neuron[p9_idx])

raster = np.zeros((BINS, COLS), dtype=np.float32)
rate_hist = np.zeros(COLS, dtype=np.float32)
p9_hist = np.zeros(COLS, dtype=np.float32)
sugar_on = p9_on = False

pygame.init()
W, H = 1120, 660
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption(f"Fly Brain Live — {brain.n:,} neurons (Metal GPU)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("menlo", 15)
font_b = pygame.font.SysFont("menlo", 18, bold=True)

RASTER = pygame.Rect(20, 60, 700, 430)
RATE = pygame.Rect(740, 60, 360, 430)
BAR = pygame.Rect(740, 380, 340, 24)


def colormap(v: np.ndarray) -> np.ndarray:
    x = np.clip(v, 0, None)
    r = 24 + 231 * np.minimum(x, 1.0)
    g = 18 + 190 * np.minimum(x / 3.0, 1.0)
    b = 32 + 120 * np.minimum(x / 8.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def draw_panel(rect: pygame.Rect, border=(90, 90, 110)) -> None:
    pygame.draw.rect(screen, (14, 12, 20), rect)
    pygame.draw.rect(screen, border, rect, 1)


running = True
while running:
    loop_t0 = time.perf_counter()

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_s:
                sugar_on = not sugar_on
                brain.set_rate("sugar", 200.0 if sugar_on else 0.0)
            elif ev.key == pygame.K_p:
                p9_on = not p9_on
                brain.set_rate("p9_left", 150.0 if p9_on else 0.0)
                brain.set_rate("p9_right", 150.0 if p9_on else 0.0)
            elif ev.key == pygame.K_r:
                brain.reset()

    # ---- advance the brain by one 50ms tick ----
    brain.run(steps=TICK_STEPS)
    sp_t, sp_n = brain.drain_spikes()

    raster = np.roll(raster, -1, axis=1)
    rate_hist = np.roll(rate_hist, -1)
    p9_hist = np.roll(p9_hist, -1)
    new_col = np.zeros(BINS, dtype=np.float32)
    if len(sp_n):
        np.add.at(new_col, bin_of_neuron[sp_n], 1.0)
    raster[:, -1] = new_col
    rate_hist[-1] = len(sp_n) / 0.05
    p9_hist[-1] = np.isin(sp_n, p9_idx).sum() / 2.0 / 0.05

    # ---- render ----
    screen.fill((8, 8, 12))
    draw_panel(RASTER)
    img = colormap(raster.T)  # (COLS, BINS, 3), time on x
    surf = pygame.surfarray.make_surface(img[:, ::-1, :])
    surf = pygame.transform.scale(surf, (RASTER.w, RASTER.h))
    screen.blit(surf, RASTER.topleft)
    ybin2y = lambda b: RASTER.bottom - int((b + 1) * RASTER.h / BINS)
    for b in sugar_bins:
        pygame.draw.rect(screen, (255, 170, 40), (RASTER.left, ybin2y(b), 5, max(1, RASTER.h // BINS)))
    for b in p9_bins:
        pygame.draw.rect(screen, (80, 220, 255), (RASTER.left + 6, ybin2y(b), 5, max(1, RASTER.h // BINS)))

    draw_panel(RATE)
    peak = max(60.0, float(rate_hist.max()))
    pts = [(RATE.left + int(i * RATE.w / COLS),
            RATE.bottom - int(min(rate_hist[i] / peak, 1.0) * (RATE.h - 8)))
           for i in range(COLS)]
    if len(pts) > 1:
        pygame.draw.lines(screen, (255, 120, 90), False, pts, 2)
    pygame.draw.line(screen, (60, 60, 80), (RATE.left, RATE.bottom), (RATE.right, RATE.bottom), 1)
    txt = font.render(f"whole-brain rate  peak {peak:.0f} Hz", True, (200, 200, 210))
    screen.blit(txt, (RATE.left + 6, RATE.top + 4))

    # P9 bar
    pygame.draw.rect(screen, (24, 24, 34), BAR)
    frac = min(p9_hist[-1] / DRIVE_NORM_MAX, 1.0)
    pygame.draw.rect(screen, (80, 220, 255), (BAR.x, BAR.y, int(BAR.w * frac), BAR.h))
    pygame.draw.rect(screen, (110, 110, 130), BAR, 1)
    screen.blit(font_b.render(f"P9 {p9_hist[-1]:5.1f} Hz", True, (80, 220, 255)), (BAR.x, BAR.bottom + 6))
    screen.blit(font_b.render(f"sugar {'ON' if sugar_on else 'OFF'}", True,
                              (255, 170, 40) if sugar_on else (120, 120, 130)), (BAR.x + 220, BAR.bottom + 6))
    screen.blit(font_b.render(f"p9    {'ON' if p9_on else 'OFF'}", True,
                              (80, 220, 255) if p9_on else (120, 120, 130)), (BAR.x + 380, BAR.bottom + 6))

    screen.blit(font_b.render("spike raster (512 neuron-bins x 12s, orange=sugar GRNs, blue=P9)",
                              True, (210, 210, 220)), (RASTER.left, RASTER.top - 26))
    screen.blit(font.render("S sugar | P p9 | R reset | Esc quit", True, (150, 150, 160)),
                (RASTER.left, H - 28))
    pygame.display.flip()

    # ---- real-time pacing (1x) ----
    elapsed = time.perf_counter() - loop_t0
    if elapsed < FRAME_MS / 1000.0:
        pygame.time.delay(int((FRAME_MS / 1000.0 - elapsed) * 1000))
    clock.tick(20)

brain.close()
pygame.quit()
