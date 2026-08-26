#!/usr/bin/env python3
"""
Comprehensive Network Processing Test Suite for rife-ncnn-vulkan with GLVK on GT 730 & Intel HD 4600.
"""

import os
import sys
import time
import shutil
import signal
import subprocess
import numpy as np
from PIL import Image

REPO_ROOT = "/home/user/repos/rife-ncnn-vulkan"
GLVK_BUILD = "/home/user/repos/glvk/build"
MASTER_BIN = os.path.join(REPO_ROOT, "build", "rife-ncnn-vulkan-master")
WORKER_BIN = os.path.join(REPO_ROOT, "build", "rife-ncnn-vulkan-worker")
MODEL_V46 = os.path.join(REPO_ROOT, "models", "rife-v4.6")
OUTPUT_BASE = os.path.join(REPO_ROOT, "outputs", "network_tests")

ENV_BASE = os.environ.copy()
ENV_BASE["LD_LIBRARY_PATH"] = f"{GLVK_BUILD}:{ENV_BASE.get('LD_LIBRARY_PATH', '')}"

def get_gpu_env(dev_id):
    env = ENV_BASE.copy()
    env["GLVK_DEVICE"] = str(dev_id)
    return env

def compute_psnr(img_a, img_b):
    arr_a = np.array(img_a).astype(float)
    arr_b = np.array(img_b).astype(float)
    mse = np.mean((arr_a - arr_b) ** 2)
    if mse == 0:
        return float('inf'), 0.0, 0.0
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    max_diff = np.max(np.abs(arr_a - arr_b))
    mean_diff = np.mean(np.abs(arr_a - arr_b))
    return psnr, max_diff, mean_diff

def stop_process(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

def prepare_test_dataset(num_frames=10, width=960, height=540):
    indir = os.path.join(REPO_ROOT, "outputs", "test_seq_input")
    os.makedirs(indir, exist_ok=True)
    
    src_files = sorted(os.listdir(os.path.join(REPO_ROOT, "inputs")))[:num_frames]
    for i, fname in enumerate(src_files):
        src_path = os.path.join(REPO_ROOT, "inputs", fname)
        dst_path = os.path.join(indir, f"{i:08d}.png")
        if not os.path.exists(dst_path):
            im = Image.open(src_path).convert("RGB")
            if im.size != (width, height):
                im = im.resize((width, height), Image.Resampling.BILINEAR)
            im.save(dst_path)
    return indir

def test1_single_worker_gt730():
    print("\n" + "="*70)
    print("TEST 1: Single Frame Network Dispatch over TCP -> GT 730 Worker (GLVK)")
    print("="*70)
    out_dir = os.path.join(OUTPUT_BASE, "test1_gt730")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "out_0.png")
    
    port = 19101
    master_cmd = [
        MASTER_BIN,
        "-p", str(port),
        "-m", MODEL_V46,
        "-0", os.path.join(REPO_ROOT, "images", "0.png"),
        "-1", os.path.join(REPO_ROOT, "images", "1.png"),
        "-o", out_file,
        "-v"
    ]
    
    worker_cmd = [
        WORKER_BIN,
        "-c", f"127.0.0.1:{port}",
        "-g", "0"
    ]
    
    master_p = subprocess.Popen(master_cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.3)
    worker_p = subprocess.Popen(worker_cmd, env=get_gpu_env(0), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    m_out, m_err = master_p.communicate(timeout=30)
    stop_process(worker_p)
    
    assert master_p.returncode == 0, f"Master failed: {m_err.decode()}"
    assert os.path.exists(out_file), f"Output file missing: {out_file}"
    
    im = Image.open(out_file)
    print(f"[PASS] Output generated: {out_file}, Resolution: {im.size}")
    return True

def test2_single_worker_hd4600():
    print("\n" + "="*70)
    print("TEST 2: Single Frame Network Dispatch over TCP -> HD 4600 Worker (GLVK)")
    print("="*70)
    out_dir = os.path.join(OUTPUT_BASE, "test2_hd4600")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "out_0.png")
    
    port = 19102
    master_cmd = [
        MASTER_BIN,
        "-p", str(port),
        "-m", MODEL_V46,
        "-0", os.path.join(REPO_ROOT, "images", "0.png"),
        "-1", os.path.join(REPO_ROOT, "images", "1.png"),
        "-o", out_file,
        "-v"
    ]
    
    worker_cmd = [
        WORKER_BIN,
        "-c", f"127.0.0.1:{port}",
        "-g", "0"
    ]
    
    master_p = subprocess.Popen(master_cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.3)
    worker_p = subprocess.Popen(worker_cmd, env=get_gpu_env(1), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    m_out, m_err = master_p.communicate(timeout=30)
    stop_process(worker_p)
    
    assert master_p.returncode == 0, f"Master failed: {m_err.decode()}"
    assert os.path.exists(out_file), f"Output file missing: {out_file}"
    
    im = Image.open(out_file)
    print(f"[PASS] Output generated: {out_file}, Resolution: {im.size}")
    
    # Compare Test 1 vs Test 2 outputs
    im1 = Image.open(os.path.join(OUTPUT_BASE, "test1_gt730", "out_0.png"))
    psnr, max_d, mean_d = compute_psnr(im1, im)
    print(f"[VERIFY] GT730 vs HD4600 over Network: PSNR={psnr:.2f}dB, MaxDiff={max_d}, MeanDiff={mean_d:.6f}")
    assert max_d <= 1.0, f"Excessive difference between GPUs: {max_d}"
    return True

def test3_dual_gpu_cluster(seq_indir):
    print("\n" + "="*70)
    print("TEST 3: Multi-Frame Distributed Processing with Dual-GPU Cluster (GT 730 + HD 4600)")
    print("="*70)
    out_dir = os.path.join(OUTPUT_BASE, "test3_dual_gpu")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    
    port = 19103
    num_frames = 16
    master_cmd = [
        MASTER_BIN,
        "-p", str(port),
        "-m", MODEL_V46,
        "-i", seq_indir,
        "-o", out_dir,
        "-n", str(num_frames),
        "-v"
    ]
    
    master_p = subprocess.Popen(master_cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.4)
    
    w1_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0"],
                            env=get_gpu_env(0), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    w2_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0"],
                            env=get_gpu_env(1), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    start_t = time.time()
    m_out, m_err = master_p.communicate(timeout=60)
    elapsed = time.time() - start_t
    stop_process(w1_p)
    stop_process(w2_p)
    
    assert master_p.returncode == 0, f"Master failed: {m_err.decode()}"
    
    out_files = sorted(os.listdir(out_dir))
    print(f"[PASS] Master completed in {elapsed:.2f}s ({num_frames / elapsed:.2f} FPS). Generated {len(out_files)}/{num_frames} frames.")
    assert len(out_files) == num_frames, f"Expected {num_frames} frames, got {len(out_files)}"
    
    for fname in out_files:
        p = os.path.join(out_dir, fname)
        im = Image.open(p)
        assert im.size == (960, 540)
    print(f"[VERIFY] All {num_frames} frames verified successfully!")
    return True

def test4_fault_tolerance_requeue(seq_indir):
    print("\n" + "="*70)
    print("TEST 4: Fault Tolerance & In-Flight Re-Queue on Worker Crash (Kill GT 730 mid-flight)")
    print("="*70)
    out_dir = os.path.join(OUTPUT_BASE, "test4_fault_tolerance")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    
    port = 19104
    num_frames = 20
    master_cmd = [
        MASTER_BIN,
        "-p", str(port),
        "-m", MODEL_V46,
        "-i", seq_indir,
        "-o", out_dir,
        "-n", str(num_frames),
        "-v"
    ]
    
    master_p = subprocess.Popen(master_cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.4)
    
    w1_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0"],
                            env=get_gpu_env(0), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    w2_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0"],
                            env=get_gpu_env(1), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(1.5)
    print("[CHAOS] Simulating sudden crash of Worker 1 (GT 730) with SIGKILL...")
    w1_p.kill()
    w1_p.wait()
    
    m_out, m_err = master_p.communicate(timeout=60)
    stop_process(w2_p)
    
    err_str = m_err.decode()
    print("Master log excerpt:")
    for line in err_str.splitlines():
        if "worker" in line.lower() or "re-queu" in line.lower() or "complete" in line.lower():
            print(f"  {line}")
            
    assert "re-queuing" in err_str.lower() or "disconnected" in err_str.lower(), "Expected master to detect worker disconnect & requeue"
    assert master_p.returncode == 0, f"Master failed to recover: {err_str}"
    
    out_files = sorted(os.listdir(out_dir))
    print(f"[PASS] Recovered completely! Generated {len(out_files)}/{num_frames} frames despite worker failure.")
    assert len(out_files) == num_frames, f"Expected {num_frames} frames, got {len(out_files)}"
    return True

def test5_dynamic_join(seq_indir):
    print("\n" + "="*70)
    print("TEST 5: Dynamic Worker Join (HD 4600 joins cluster mid-render)")
    print("="*70)
    out_dir = os.path.join(OUTPUT_BASE, "test5_dynamic_join")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    
    port = 19105
    num_frames = 20
    master_cmd = [
        MASTER_BIN,
        "-p", str(port),
        "-m", MODEL_V46,
        "-i", seq_indir,
        "-o", out_dir,
        "-n", str(num_frames),
        "-v"
    ]
    
    master_p = subprocess.Popen(master_cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.4)
    
    w1_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0"],
                            env=get_gpu_env(0), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2.0)
    print("[SCALE-UP] Dynamically attaching Worker 2 (HD 4600) to live render session...")
    w2_p = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0"],
                            env=get_gpu_env(1), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    m_out, m_err = master_p.communicate(timeout=60)
    stop_process(w1_p)
    stop_process(w2_p)
    
    assert master_p.returncode == 0, f"Master failed: {m_err.decode()}"
    out_files = sorted(os.listdir(out_dir))
    print(f"[PASS] Scale-up successful! Generated {len(out_files)}/{num_frames} frames.")
    assert len(out_files) == num_frames
    return True

def test6_benchmarks(seq_indir):
    print("\n" + "="*70)
    print("TEST 6: Throughput & Scaling Benchmark Comparison (GT 730 vs HD 4600 vs Dual-GPU)")
    print("="*70)
    num_frames = 16
    results = {}
    
    configs = [
        ("Solo GT 730", [0], 19106),
        ("Solo HD 4600", [1], 19107),
        ("Dual GPU (GT 730 + HD 4600)", [0, 1], 19108)
    ]
    
    for label, gpus, port in configs:
        out_dir = os.path.join(OUTPUT_BASE, f"bench_{port}")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        
        master_cmd = [
            MASTER_BIN,
            "-p", str(port),
            "-m", MODEL_V46,
            "-i", seq_indir,
            "-o", out_dir,
            "-n", str(num_frames)
        ]
        
        master_p = subprocess.Popen(master_cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.3)
        
        workers = []
        for dev in gpus:
            wp = subprocess.Popen([WORKER_BIN, "-c", f"127.0.0.1:{port}", "-g", "0"],
                                  env=get_gpu_env(dev), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            workers.append(wp)
            
        start_t = time.time()
        m_out, m_err = master_p.communicate(timeout=60)
        elapsed = time.time() - start_t
        for wp in workers:
            stop_process(wp)
            
        fps = num_frames / elapsed
        sec_per_frame = elapsed / num_frames
        results[label] = {
            "elapsed": elapsed,
            "fps": fps,
            "sec_per_frame": sec_per_frame
        }
        print(f"  {label:<30}: {elapsed:.2f}s total | {fps:.2f} FPS | {sec_per_frame:.3f} s/frame")
        
    speedup = results["Solo GT 730"]["elapsed"] / results["Dual GPU (GT 730 + HD 4600)"]["elapsed"]
    print(f"\n[BENCHMARK RESULT] Dual-GPU Cluster Speedup over solo GT 730: {speedup:.2f}x")
    return results

def main():
    print("Initializing comprehensive rife-ncnn-vulkan network test suite...")
    print(f"Model: {MODEL_V46}")
    print(f"GLVK:  {GLVK_BUILD}")
    
    seq_indir = prepare_test_dataset(num_frames=10, width=960, height=540)
    
    t1 = test1_single_worker_gt730()
    t2 = test2_single_worker_hd4600()
    t3 = test3_dual_gpu_cluster(seq_indir)
    t4 = test4_fault_tolerance_requeue(seq_indir)
    t5 = test5_dynamic_join(seq_indir)
    bench_results = test6_benchmarks(seq_indir)
    
    print("\n" + "="*70)
    print("                   ALL NETWORK TESTS PASSED!")
    print("="*70)
    print(f"Test 1 (GT 730 Network Single-Pair)     : PASSED")
    print(f"Test 2 (HD 4600 Network Single-Pair)    : PASSED")
    print(f"Test 3 (Dual-GPU Multi-Frame Cluster)   : PASSED")
    print(f"Test 4 (Fault Tolerance & Re-queue)     : PASSED")
    print(f"Test 5 (Dynamic Scale-Up Worker Join)   : PASSED")
    print(f"Test 6 (Performance Benchmarks)         : PASSED")
    print("="*70)

if __name__ == "__main__":
    main()
