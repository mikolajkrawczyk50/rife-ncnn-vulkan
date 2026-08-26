#!/usr/bin/env python3
"""
Benchmark RIFE v4.6 on GT 730 and Intel HD 4600 with default vs 4:4:4 threading configuration.
"""

import os
import sys
import time
import shutil
import subprocess
import numpy as np
from PIL import Image

REPO_ROOT = "/home/user/repos/rife-ncnn-vulkan"
GLVK_BUILD = "/home/user/repos/glvk/build"
STANDALONE_BIN = os.path.join(REPO_ROOT, "build", "rife-ncnn-vulkan")
MASTER_BIN = os.path.join(REPO_ROOT, "build", "rife-ncnn-vulkan-master")
WORKER_BIN = os.path.join(REPO_ROOT, "build", "rife-ncnn-vulkan-worker")
MODEL_V46 = os.path.join(REPO_ROOT, "models", "rife-v4.6")
INPUTS_DIR = os.path.join(REPO_ROOT, "inputs")
OUTPUT_BASE = os.path.join(REPO_ROOT, "outputs", "bench_444")

ENV_BASE = os.environ.copy()
ENV_BASE["LD_LIBRARY_PATH"] = f"{GLVK_BUILD}:{ENV_BASE.get('LD_LIBRARY_PATH', '')}"

def get_gpu_env(dev_id):
    env = ENV_BASE.copy()
    if dev_id >= 0:
        env["GLVK_DEVICE"] = str(dev_id)
    return env

def stop_process(p):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

def prepare_input_slice(num_frames=20):
    seq_dir = os.path.join(OUTPUT_BASE, "input_slice")
    shutil.rmtree(seq_dir, ignore_errors=True)
    os.makedirs(seq_dir, exist_ok=True)
    src_files = sorted(os.listdir(INPUTS_DIR))[:num_frames]
    for i, fname in enumerate(src_files):
        shutil.copyfile(os.path.join(INPUTS_DIR, fname), os.path.join(seq_dir, f"{i:08d}.png"))
    return seq_dir

def run_standalone_test(label, gpu_dev, thread_opt, seq_indir, num_frames=24):
    out_dir = os.path.join(OUTPUT_BASE, f"standalone_{label.replace(' ', '_')}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    
    ncnn_gpu = "-1" if gpu_dev == -1 else "0"
    cmd = [
        STANDALONE_BIN,
        "-m", MODEL_V46,
        "-i", seq_indir,
        "-o", out_dir,
        "-n", str(num_frames),
        "-g", ncnn_gpu,
        "-j", thread_opt
    ]
    
    env = get_gpu_env(gpu_dev)
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True)
    elapsed = time.time() - t0
    
    assert res.returncode == 0, f"Run {label} failed: {res.stderr.decode(errors='replace')}"
    out_files = os.listdir(out_dir)
    assert len(out_files) == num_frames, f"Expected {num_frames} frames, got {len(out_files)}"
    
    fps = num_frames / elapsed
    ms_frame = (elapsed / num_frames) * 1000.0
    return {
        "time": elapsed,
        "fps": fps,
        "ms_frame": ms_frame,
        "frames": num_frames
    }

def run_network_test(label, thread_opt_save, worker_threads, seq_indir, num_frames=24):
    out_dir = os.path.join(OUTPUT_BASE, f"net_{label.replace(' ', '_')}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    
    port = 19400 + int(time.time() % 500)
    master_cmd = [
        MASTER_BIN,
        "-p", str(port),
        "-m", MODEL_V46,
        "-i", seq_indir,
        "-o", out_dir,
        "-n", str(num_frames),
        "-j", str(thread_opt_save)
    ]
    
    master_p = subprocess.Popen(master_cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.3)
    
    w1_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0", "-t", str(worker_threads)],
                            env=get_gpu_env(0), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    w2_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0", "-t", str(worker_threads)],
                            env=get_gpu_env(1), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    t0 = time.time()
    m_out, m_err = master_p.communicate(timeout=120)
    elapsed = time.time() - t0
    stop_process(w1_p)
    stop_process(w2_p)
    
    assert master_p.returncode == 0, f"Network cluster {label} failed: {m_err.decode(errors='replace')}"
    out_files = os.listdir(out_dir)
    assert len(out_files) == num_frames, f"Expected {num_frames} frames, got {len(out_files)}"
    
    fps = num_frames / elapsed
    ms_frame = (elapsed / num_frames) * 1000.0
    return {
        "time": elapsed,
        "fps": fps,
        "ms_frame": ms_frame,
        "frames": num_frames
    }

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    num_frames = 24
    seq_indir = prepare_input_slice(num_frames=16)
    
    print("="*85)
    print(f"  RIFE v4.6 PERFORMANCE BENCHMARK: DEFAULT (1:2:2) vs UNBOTTLENECKED (4:4:4)")
    print(f"  Test dataset: {num_frames} frames from inputs/ @ 640x360")
    print("="*85)
    
    results = {}
    
    # 1. CPU Reference
    print("[1/7] Running CPU Reference (-j 4:4:4)...")
    results["CPU (-j 4:4:4)"] = run_standalone_test("CPU_444", -1, "4:4:4", seq_indir, num_frames)
    
    # 2. GT 730 (GLVK) Default vs 4:4:4
    print("[2/7] Running Solo GT 730 Default (-j 1:2:2)...")
    results["Solo GT 730 (-j 1:2:2)"] = run_standalone_test("GT730_default", 0, "1:2:2", seq_indir, num_frames)
    
    print("[3/7] Running Solo GT 730 Unbottlenecked (-j 4:4:4)...")
    results["Solo GT 730 (-j 4:4:4)"] = run_standalone_test("GT730_444", 0, "4:4:4", seq_indir, num_frames)
    
    # 3. HD 4600 (GLVK) Default vs 4:4:4
    print("[4/7] Running Solo HD 4600 Default (-j 1:2:2)...")
    results["Solo HD 4600 (-j 1:2:2)"] = run_standalone_test("HD4600_default", 1, "1:2:2", seq_indir, num_frames)
    
    print("[5/7] Running Solo HD 4600 Unbottlenecked (-j 4:4:4)...")
    results["Solo HD 4600 (-j 4:4:4)"] = run_standalone_test("HD4600_444", 1, "4:4:4", seq_indir, num_frames)
    
    # 4. Dual-GPU Network Cluster Default (save=2, worker_threads=1) vs Unbottlenecked (save=4, worker_threads=4)
    print("[6/7] Running Dual-GPU Network Cluster Default (save=2, threads=1)...")
    results["Dual GPU Cluster (Default 2:1)"] = run_network_test("Dual_Default", 2, 1, seq_indir, num_frames)
    
    print("[7/7] Running Dual-GPU Network Cluster Unbottlenecked (save=4, threads=4)...")
    results["Dual GPU Cluster (4:4:4 Network)"] = run_network_test("Dual_444", 4, 4, seq_indir, num_frames)
    
    print("\n" + "="*85)
    print(f"{'Configuration':<36} | {'Total Time':<11} | {'Throughput':<11} | {'Frame Latency':<16}")
    print("="*85)
    for label, r in results.items():
        print(f"{label:<36} | {r['time']:>7.2f} s   | {r['fps']:>7.2f} FPS | {r['ms_frame']:>9.2f} ms/f")
    print("="*85)
    
    gt_speedup = results["Solo GT 730 (-j 1:2:2)"]["time"] / results["Solo GT 730 (-j 4:4:4)"]["time"]
    hd_speedup = results["Solo HD 4600 (-j 1:2:2)"]["time"] / results["Solo HD 4600 (-j 4:4:4)"]["time"]
    cluster_speedup = results["Solo GT 730 (-j 1:2:2)"]["time"] / results["Dual GPU Cluster (4:4:4 Network)"]["time"]
    
    print(f"\n[ANALYSIS & OBSERVATIONS]")
    print(f"  - GT 730 Speedup (4:4:4 vs 1:2:2)         : {gt_speedup:.2f}x")
    print(f"  - HD 4600 Speedup (4:4:4 vs 1:2:2)        : {hd_speedup:.2f}x")
    print(f"  - Dual-GPU Cluster Speedup over solo GT730 : {cluster_speedup:.2f}x")

if __name__ == "__main__":
    main()
