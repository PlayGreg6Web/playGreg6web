# Run: powershell -File tests/test_update_prebuilt_stderr.ps1
# tools/update_prebuilt.ps1 must not die on a native stderr line (python progress such as "[plan] ...") when its output is redirected
# (*>, 2>&1) or it runs as a background job (Windows PowerShell 5.1 turns such lines into terminating errors under
# $ErrorActionPreference = "Stop"), while a real failure (non-zero exit) must still be fatal. A fake python.cmd (stderr line, exit code
# from $env:FAKE_RC) stands in for the build tools; no game or build is touched.
$ErrorActionPreference = "Stop"
$script = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\tools\update_prebuilt.ps1"))
$root = Join-Path ([IO.Path]::GetTempPath()) "upstderr-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force "$root\fake" | Out-Null
Set-Content -Encoding ascii "$root\fake\python.cmd" @'
@echo off
echo [plan] fake native stderr line 1>&2
if defined FAKE_TMP if not exist "%FAKE_TMP%" mkdir "%FAKE_TMP%"
exit /b %FAKE_RC%
'@
Set-Content "$root\pack.wgsl.pack" "x"
$oldPath = $env:PATH
function Check($ok, $msg) { if (-not $ok) { throw "FAIL: $msg" }; Write-Host "ok: $msg" }

# Runs update_prebuilt in the given mode; returns $true when it finished without error and exit code 0.
function Invoke-Case([string]$mode, [bool]$web, [int]$rc, [string]$name) {
    $dir = Join-Path $root $name
    $env:FAKE_RC = "$rc"; $env:FAKE_TMP = "$dir.new"; $env:EMSDK = Join-Path $root "no-emsdk"
    $h = @{ Dir = $dir; Exe = "x"; GameDir = "y"; SkipSmoke = $true }   # named-parameter splat (an array would bind positionally)
    $a = @("-Dir", $dir, "-Exe", "x", "-GameDir", "y", "-SkipSmoke")
    if ($web) { $h.Web = $true; $h.ShaderPack = "$root\pack.wgsl.pack"; $a += @("-Web", "-ShaderPack", "$root\pack.wgsl.pack") }
    $log = Join-Path $root "$name.log"
    switch ($mode) {
        "redirect" {  # *> to a file inside this process
            try { & $script @h *> $log; return ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) } catch { Write-Host "ERR: $_"; return $false }
        }
        "merge" {  # 2>&1 into the pipeline
            try { $null = & $script @h 2>&1; return ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) } catch { Write-Host "ERR: $_"; return $false }
        }
        "job" {  # background job
            $j = Start-Job -ArgumentList $script, $h -ScriptBlock { param($s, $x) & $s @x }
            $null = Wait-Job $j; $ok = ($j.State -eq "Completed")
            $null = Receive-Job $j -ErrorAction SilentlyContinue 2>&1; Remove-Job $j -Force
            return $ok
        }
        "child" {  # separate powershell with stderr redirected by the caller
            # this test runs with "Stop" on purpose: the script must not put anything on stderr that a caller could trip over
            if ($rc -ne 0) { $ErrorActionPreference = "Continue" }   # a real failure legitimately writes its error to stderr
            try { $null = & powershell -NoProfile -File $script @a 2>&1 } finally { $ErrorActionPreference = "Stop" }
            return ($LASTEXITCODE -eq 0)
        }
    }
}
try {
    $env:PATH = "$root\fake;$oldPath"
    foreach ($web in @($false, $true)) {
        $kind = if ($web) { "web" } else { "native" }
        foreach ($mode in @("redirect", "merge", "job", "child")) {
            Check (Invoke-Case $mode $web 0 "$kind-$mode-ok") "${kind}/${mode}: stderr line with exit 0 is not fatal"
            Check (-not (Invoke-Case $mode $web 1 "$kind-$mode-fail")) "${kind}/${mode}: exit 1 is still fatal"
        }
    }
    # -DryRun prints the plan under redirection / as a job, in both paths
    foreach ($web in @($false, $true)) {
        $h = @{ Dir = "$root\dry"; Exe = "x"; GameDir = "y"; DryRun = $true }
        if ($web) { $h.Web = $true; $h.ShaderPack = "$root\pack.wgsl.pack" }
        $env:EMSDK = Join-Path $root "no-emsdk"
        & $script @h *> "$root\dry.log"
        Check (Select-String -Path "$root\dry.log" -Pattern "dry-run" -Quiet) "dry-run (web=$web) under *>"
        $j = Start-Job -ArgumentList $script, $h -ScriptBlock { param($s, $x) & $s @x }
        $null = Wait-Job $j; $st = $j.State; Remove-Job $j -Force
        Check ($st -eq "Completed") "dry-run (web=$web) as a job"
    }
    Write-Host "ALL PASS"
} finally {
    $env:PATH = $oldPath
    Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue
}
