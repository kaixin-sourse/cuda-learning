# Windows Profiling Commands

## Nsight Systems

Use the full path on this machine:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe" profile `
  --trace=cuda,osrt `
  --output week7_reduction_trace `
  .\cmake-build-debug\week7_profile_reduction.exe
```

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe" profile `
  --trace=cuda,osrt `
  --output week7_matmul_trace `
  .\cmake-build-debug\week7_profile_matmul.exe
```

## Nsight Compute

`ncu` is already available in PATH on this machine:

```powershell
ncu --set full .\cmake-build-debug\week7_profile_reduction.exe
```

```powershell
ncu --set full .\cmake-build-debug\week7_profile_matmul.exe
```

If you only want a lighter first pass:

```powershell
ncu --section LaunchStats --section SpeedOfLight .\cmake-build-debug\week7_profile_reduction.exe
```
