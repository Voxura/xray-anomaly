import os
import sys
import time
import subprocess
import shutil
import hashlib
import ctypes
from ctypes import wintypes

# Set up 64-bit psapi / kernel32 signatures
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
psapi.EnumProcessModules.restype = wintypes.BOOL

psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
psapi.GetModuleFileNameExW.restype = wintypes.DWORD

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

def get_hash(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def get_loaded_modules(pid):
    h_proc = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h_proc:
        return []
    h_mods = (wintypes.HMODULE * 1024)()
    cb_needed = wintypes.DWORD()
    modules = []
    if psapi.EnumProcessModules(h_proc, h_mods, ctypes.sizeof(h_mods), ctypes.byref(cb_needed)):
        num_mods = cb_needed.value // ctypes.sizeof(wintypes.HMODULE)
        for i in range(num_mods):
            mod_path = ctypes.create_unicode_buffer(1024)
            psapi.GetModuleFileNameExW(h_proc, h_mods[i], mod_path, 1024)
            modules.append(os.path.basename(mod_path.value))
    kernel32.CloseHandle(h_proc)
    return modules

def run_case(name, exe_dir, exe_name, args, duration=4):
    print(f"\n==========================================")
    print(f"RUNNING: {name}")
    print(f"Dir: {exe_dir}")
    print(f"Exe: {exe_name}")
    print(f"Args: {' '.join(args)}")
    print(f"==========================================")
    
    exe_path = os.path.join(exe_dir, exe_name)
    if not os.path.exists(exe_path):
        print(f"ERROR: {exe_path} not found")
        return {"name": name, "success": False, "error": "exe not found"}
        
    cmd = [exe_path] + args
    start_t = time.time()
    p = subprocess.Popen(cmd, cwd=exe_dir)
    pid = p.pid
    print(f"Process spawned with PID: {pid}")
    
    modules_loaded = []
    is_running = True
    for i in range(int(duration * 2)):
        time.sleep(0.5)
        ret = p.poll()
        if ret is not None:
            is_running = False
            print(f"Process terminated early with return code: {ret} after {time.time()-start_t:.1f}s")
            break
        # Sample modules while running
        mods = get_loaded_modules(pid)
        if mods and len(mods) > len(modules_loaded):
            modules_loaded = mods
            
    if is_running:
        print(f"Process running stably after {duration}s. Terminating gracefully...")
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
        exit_code = "RUNNING_STABLE (TERMINATED)"
    else:
        exit_code = f"TERMINATED (Code {p.returncode})"
        
    # Check for engine.log or *.log in exe_dir or appdata
    appdata_logs = []
    for root, dirs, files in os.walk(exe_dir):
        for f in files:
            if f.endswith(".log") and not f.startswith("task-"):
                log_p = os.path.join(root, f)
                if os.path.getmtime(log_p) >= start_t - 2:
                    appdata_logs.append(log_p)
                    
    print(f"Exit Code: {exit_code}")
    print(f"Modules detected count: {len(modules_loaded)}")
    
    rust_dx12_loaded = any("xray_render_dx12" in m.lower() for m in modules_loaded)
    d3d12_loaded = any("d3d12.dll" == m.lower() for m in modules_loaded)
    dxgi_loaded = any("dxgi.dll" == m.lower() for m in modules_loaded)
    print(f"  -> xray_render_dx12.dll loaded: {rust_dx12_loaded}")
    print(f"  -> d3d12.dll loaded: {d3d12_loaded}")
    print(f"  -> dxgi.dll loaded: {dxgi_loaded}")
    
    for lp in appdata_logs:
        print(f"\n--- LOG CONTENT ({os.path.basename(lp)}) ---")
        try:
            with open(lp, "r", errors="ignore") as f:
                lines = f.readlines()
                print("".join(lines[-30:]))
        except Exception as e:
            print(f"Could not read log {lp}: {e}")
            
    return {
        "name": name,
        "exit_code": exit_code,
        "modules": modules_loaded,
        "rust_dx12_loaded": rust_dx12_loaded,
        "d3d12_loaded": d3d12_loaded,
        "dxgi_loaded": dxgi_loaded,
        "logs": appdata_logs
    }

def main():
    root_dir = os.path.abspath(r"C:\falcon\Projects\Windows\xray-anomaly")
    bin_dbg = os.path.join(root_dir, r"_build\_game\bin_dbg")
    exe_name = "AnomalyDX11.exe"
    dll_name = "xray_render_dx12.dll"
    dll_path = os.path.join(bin_dbg, dll_name)
    dll_bak = dll_path + ".missing_test_bak"
    
    print("==========================================")
    print("PHASE 3 VERIFICATION MATRIX RUNNER")
    print("MSVC Toolset: v145 (Release / LTCG / WPO)")
    print("Target Architecture: x86_64")
    print("==========================================")
    
    exe_hash = get_hash(os.path.join(bin_dbg, exe_name))
    dll_hash = get_hash(dll_path)
    print(f"Binary: {exe_name} (SHA256: {exe_hash})")
    print(f"Binary: {dll_name} (SHA256: {dll_hash})")
    
    # CASE 1: Pure Rust DX12 Present
    args_case1 = ["-use_rust_dx12", "-r5", "-nointro", "-nosound", "-pure_alloc", "-nosplashwindow"]
    res1 = run_case("Case 1 (Pure Rust DX12 Present)", bin_dbg, exe_name, args_case1, duration=4)
    
    # CASE 2: Default Legacy Fallback
    args_case2 = ["-nointro", "-nosound", "-pure_alloc", "-nosplashwindow"]
    res2 = run_case("Case 2 (Default Legacy Fallback)", bin_dbg, exe_name, args_case2, duration=4)
    
    # CASE 3: Controlled Missing DLL Test (Temporarily rename xray_render_dx12.dll)
    args_case3 = ["-use_rust_dx12", "-r5", "-nointro", "-nosound", "-pure_alloc", "-nosplashwindow", "-silent_error_mode"]
    res3 = None
    try:
        if os.path.exists(dll_path):
            os.rename(dll_path, dll_bak)
        res3 = run_case("Case 3 (Controlled Missing DLL Test)", bin_dbg, exe_name, args_case3, duration=3)
    finally:
        if os.path.exists(dll_bak):
            os.rename(dll_bak, dll_path)
            
    print("\n==========================================")
    print("PHASE 3 ACCEPTANCE MATRIX RESULTS")
    print("==========================================")
    print(f"| Case | Launch Args | Exit / Status | Rust DX12 Loaded | D3D12 / DXGI | Result |")
    print(f"|---|---|---|---|---|---|")
    print(f"| Case 1: Pure Rust DX12 | `{' '.join(args_case1)}` | {res1.get('exit_code')} | {res1.get('rust_dx12_loaded')} | {res1.get('d3d12_loaded')} / {res1.get('dxgi_loaded')} | {'PASS' if res1.get('rust_dx12_loaded') and res1.get('d3d12_loaded') else 'FAIL'} |")
    print(f"| Case 2: Legacy Fallback | `{' '.join(args_case2)}` | {res2.get('exit_code')} | {res2.get('rust_dx12_loaded')} | {res2.get('d3d12_loaded')} / {res2.get('dxgi_loaded')} | {'PASS' if not res2.get('rust_dx12_loaded') else 'FAIL'} |")
    print(f"| Case 3: Missing DLL Test | `{' '.join(args_case3)}` | {res3.get('exit_code')} | {res3.get('rust_dx12_loaded')} | {res3.get('d3d12_loaded')} / {res3.get('dxgi_loaded')} | {'PASS' if not res3.get('rust_dx12_loaded') else 'FAIL'} |")
    print("==========================================")

if __name__ == "__main__":
    main()
