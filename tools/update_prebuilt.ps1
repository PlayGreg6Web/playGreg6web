<#
.SYNOPSIS
  Rebuild the ready-to-play native build and swap it in atomically, only when the build succeeded.
  The build goes into a temp dir next to the target (using the object/lift/recomp caches); the previous
  playable copy is renamed aside and only deleted after the new one is in place, so a failed build never
  breaks it. Writes prebuilt.json (source commit) into the build. Play it with: .\tools\play_native.ps1 -Prebuilt
  With -Web: the same for the browser build (tools/play_local.ps1 -Prebuilt); dir -Dir, else $env:PLAYGTA6_PREBUILT_WEB_DIR,
  else %LOCALAPPDATA%\playgta6\prebuilt-web. Also needs a shader pack: -ShaderPack, else derived from the std pack (-StdPack / PLAYGTA6_NATIVE_SHADER_PACK, tools/subset_wgsl_pack.py), else PLAYGTA6_SHADER_PACK; and emsdk;
  reuses the lift cache and the previous object cache.
  Smoke gate (native): after the build, one headless ~75 s menu-flow boot (--live-executor, no window) runs from the new copy; it is
  swapped in only if it reached presents > 0 and ended on no fault or a known hard stop (tools/prebuilt_smoke.py). Otherwise the old
  prebuilt stays, the new one is left as <dir>.new, and the stop line and run dir are printed. -SkipSmoke opts out; -DryRun prints the plan.
.EXAMPLE
  .\tools\update_prebuilt.ps1            # builds the checked-out commit
  .\tools\update_prebuilt.ps1 -Dir H:\pg6-play
.NOTES
  Target: -Dir, else $env:PLAYGTA6_PREBUILT_DIR, else %LOCALAPPDATA%\playgta6\prebuilt. Needs PLAYGTA6_EXE and GTA6_GAME_DIR
  (or -Exe/-GameDir); the shader pack is only needed at play time. Intermediates not needed to run are removed.
#>
param(
    [string]$Dir = $env:PLAYGTA6_PREBUILT_DIR,
    [string]$Exe = $env:PLAYGTA6_EXE,
    [string]$GameDir = $env:GTA6_GAME_DIR,
    [switch]$Web,
    [string]$ShaderPack = $env:PLAYGTA6_SHADER_PACK,
    [string]$StdPack = $env:PLAYGTA6_NATIVE_SHADER_PACK,
    [switch]$SkipSmoke,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($Web) {
    # Windows PowerShell 5.1 turns any native stderr line (python progress such as "[plan] ...") into a terminating error under "Stop";
    # every native call below is checked via $LASTEXITCODE, so do not stop on stderr.
    $ErrorActionPreference = "Continue"
    $PrebuiltDir = if ($PSBoundParameters.ContainsKey("Dir")) { $Dir } else { $env:PLAYGTA6_PREBUILT_WEB_DIR }
    if (-not $PrebuiltDir) { $PrebuiltDir = Join-Path $env:LOCALAPPDATA "playgta6\prebuilt-web" }
    if (-not $Exe -or -not $GameDir) { throw "Pass -Exe and -GameDir (or set PLAYGTA6_EXE / GTA6_GAME_DIR)." }
    # Browser pack: explicit -ShaderPack wins. Else derive one from the full native (std) pack: same stages, v1 (the web
    # preloader reads v1 only and holds the pack in memory), per-shader cap 256 KiB, total cap 1.6 GB, stages of the
    # PLAYGTA6_SHADER_PACK pack kept first. Cached in work/web-pack/ (rebuilt when the std pack changes).
    if (-not $PSBoundParameters.ContainsKey("ShaderPack") -and $StdPack -and (Test-Path -LiteralPath $StdPack)) {
        $outDir = Join-Path $repo "work/web-pack"
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        $derived = Join-Path $outDir "splash.wgsl.pack"
        $stamp = "$derived.src"
        $srcInfo = (Get-Item -LiteralPath $StdPack)
        $srcKey = "$($srcInfo.FullName)|$($srcInfo.Length)|$($srcInfo.LastWriteTimeUtc.Ticks)"
        if (-not (Test-Path -LiteralPath $derived) -or -not (Test-Path -LiteralPath $stamp) -or (Get-Content -LiteralPath $stamp -Raw).Trim() -ne $srcKey) {
            $subsetArgs = @((Join-Path $PSScriptRoot "subset_wgsl_pack.py"), $StdPack, "--out", $derived)
            if ($ShaderPack -and (Test-Path -LiteralPath $ShaderPack)) { $subsetArgs += @("--priority-pack", $ShaderPack) }
            & python @subsetArgs
            if ($LASTEXITCODE -ne 0) { throw "subset_wgsl_pack failed" }
            Set-Content -LiteralPath $stamp -Value $srcKey
        }
        $ShaderPack = $derived
    }
    if (-not $ShaderPack) {
        $default = Join-Path $repo "work/splash.wgsl.pack"
        if (Test-Path -LiteralPath $default) { $ShaderPack = $default }
    }
    if (-not $ShaderPack -or -not (Test-Path -LiteralPath $ShaderPack)) { throw "No shader pack: pass -ShaderPack, or set PLAYGTA6_NATIVE_SHADER_PACK (std pack; a browser pack is derived) / PLAYGTA6_SHADER_PACK." }
    if (-not (Get-Command emcc -ErrorAction SilentlyContinue) -and -not $env:EMSDK) {
        & python (Join-Path $PSScriptRoot "emsdk_locator.py")
        if ($LASTEXITCODE -ne 0) { throw "No Emscripten found (emcc not on PATH, EMSDK unset, no work/**/emsdk)." }
    }
    if (-not (Get-Command emcmake -ErrorAction SilentlyContinue) -and $env:EMSDK) {
        $env:PATH = (Join-Path $env:EMSDK "upstream\emscripten") + [System.IO.Path]::PathSeparator + $env:PATH
    }
    $target = [System.IO.Path]::GetFullPath($PrebuiltDir)
    $tmp = "$target.new"
    if ($DryRun) {
        Write-Host "[dry-run] web prebuilt: $target (build in $tmp, swap on success; no headless smoke boot for the browser build)"
        return
    }
    if (Test-Path -LiteralPath $tmp) { Remove-Item -Recurse -Force -LiteralPath $tmp }
    $seed = @()
    if (Test-Path -LiteralPath $target) { $seed = @("--seed-from", $target) }
    & python (Join-Path $PSScriptRoot "prebuilt_web.py") prepare --repo $repo --dest $tmp @seed
    if ($LASTEXITCODE -ne 0) { throw "prepare failed" }
    Push-Location $repo
    try {
        & python tools/run_browser_boot.py --build-only --exe $Exe --game-dir $GameDir --shader-pack $ShaderPack `
            --build-dir (Join-Path $tmp "build") --boot-dir (Join-Path $tmp "boot") `
            --out-dir (Join-Path $tmp "web\dist") --web-dir (Join-Path $tmp "web")
        if ($LASTEXITCODE -ne 0) { throw "build failed (existing prebuilt left untouched; partial build in $tmp)" }
    } finally { Pop-Location }
    & python (Join-Path $PSScriptRoot "prebuilt_web.py") stamp --repo $repo --dest $tmp --shader-pack (Join-Path $tmp "build\staged-shaders\$(Split-Path -Leaf $ShaderPack)")
    if ($LASTEXITCODE -ne 0) { throw "stamp failed" }
    & python (Join-Path $PSScriptRoot "prebuilt_web.py") swap --new $tmp --target $target
    if ($LASTEXITCODE -ne 0) { throw "swap failed" }
    & python (Join-Path $PSScriptRoot "prebuilt_web.py") info --prebuilt $target --repo $repo
    return
}
if (-not $Dir) { $Dir = Join-Path $env:LOCALAPPDATA "playgta6\prebuilt" }
if (-not $Exe -or -not $GameDir) { throw "Pass -Exe and -GameDir (or set PLAYGTA6_EXE / GTA6_GAME_DIR)." }
$Dir = [IO.Path]::GetFullPath($Dir)
$tmp = "$Dir.new"; $old = "$Dir.old"
if ($DryRun) {
    Write-Host "[dry-run] native prebuilt: $Dir (build in $tmp, old copy kept as $old until the swap succeeded)"
    Write-Host "[dry-run] smoke gate: $(if ($SkipSmoke) { 'skipped (-SkipSmoke)' } else { "python tools/run_native_boot.py --preset menu-flow --name prebuilt-smoke-<yyyyMMdd-HHmmss>-<commit> --live-executor --trust-build --build-dir $tmp --runs-dir $tmp-smoke-runs$(if ($StdPack) { " --shader-pack $StdPack" }) --max-ms 75000; then python tools/prebuilt_smoke.py <run dir> --expect-name <that name> (only that runs report is read); keeps the last 3 smoke runs" })"
    exit 0
}
if (Test-Path -LiteralPath $tmp) { Remove-Item -Recurse -Force -LiteralPath $tmp }
if (Test-Path -LiteralPath $old) { Remove-Item -Recurse -Force -LiteralPath $old }
$parent = Split-Path -Parent $Dir
if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$commit = (& git -C $repo rev-parse HEAD).Trim()
$dirty = [bool](& git -C $repo status --porcelain --untracked-files=no)
Push-Location $repo
try {
    & python tools/run_native_boot.py --preset long --name prebuilt-build --play --build-only `
        --exe $Exe --game-dir $GameDir --build-dir $tmp
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE); the existing prebuilt copy is untouched." }
} catch {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -Recurse -Force -LiteralPath $tmp }
    throw
} finally { Pop-Location }
@{ commit = $commit; dirty = $dirty; built_utc = (Get-Date).ToUniversalTime().ToString("o") } |
    ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $tmp "prebuilt.json")
if (-not $SkipSmoke) {
    $smokeName = "prebuilt-smoke-$((Get-Date).ToString('yyyyMMdd-HHmmss'))-$($commit.Substring(0, 9))"  # unique per gate run: never reuse (or read) an older run's dir
    $smokeRuns = "$tmp-smoke-runs"  # outside the prebuilt (it is read-only for the runners)
    $smokeRun = Join-Path $smokeRuns $smokeName
    Write-Host "[prebuilt] smoke boot from $tmp (headless, ~75 s)..."
    $smokePack = @(); if ($StdPack -and (Test-Path -LiteralPath $StdPack)) { $smokePack = @("--shader-pack", $StdPack) }  # PLAYGTA6_NATIVE_SHADER_PACK
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    Push-Location $repo
    try {
        & python tools/run_native_boot.py --preset menu-flow --name $smokeName --live-executor --trust-build `
            --exe $Exe --game-dir $GameDir --build-dir $tmp --runs-dir $smokeRuns @smokePack --max-ms 75000 --watchdog-ms 105000
        & python (Join-Path $PSScriptRoot "prebuilt_smoke.py") $smokeRun --expect-name $smokeName
        $smokeOk = ($LASTEXITCODE -eq 0)
    } finally { Pop-Location; $ErrorActionPreference = $eap }
    & python (Join-Path $PSScriptRoot "prebuilt_smoke.py") --prune $smokeRuns --keep 3  # only inside the smoke-runs dir, never a prebuilt
    if (-not $smokeOk) {
        Write-Host "[prebuilt] smoke boot FAILED: the existing prebuilt at $Dir is kept; the new build is left at $tmp (run dir: $smokeRun). Use -SkipSmoke to swap anyway."
        exit 3
    }
}
# drop what is not needed to run (per-run outputs and object/IR intermediates)
foreach ($n in @("runs", "program\native\harness-build", "program\native\bin\playgta6_boot.pdb")) { $p = Join-Path $tmp $n; if (Test-Path -LiteralPath $p) { Remove-Item -Recurse -Force -LiteralPath $p } }
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path -like "$Dir\*" })
if ($busy) {
    Write-Host "[prebuilt] $Dir is in use (pid $($busy.Id -join ',')); new build kept at $tmp. Close the game, then: Rename-Item '$Dir' '$(Split-Path -Leaf $old)'; Rename-Item '$tmp' '$(Split-Path -Leaf $Dir)'; Remove-Item -Recurse -Force '$old'"
    exit 2
}
if (Test-Path -LiteralPath $Dir) { Rename-Item -LiteralPath $Dir -NewName (Split-Path -Leaf $old) }
try { Rename-Item -LiteralPath $tmp -NewName (Split-Path -Leaf $Dir) }
catch { if (Test-Path -LiteralPath $old) { Rename-Item -LiteralPath $old -NewName (Split-Path -Leaf $Dir) }; throw }
if (Test-Path -LiteralPath $old) { Remove-Item -Recurse -Force -LiteralPath $old }
Write-Host "[prebuilt] updated $Dir from $commit. Play: `$env:PLAYGTA6_PREBUILT_DIR='$Dir'; .\tools\play_native.ps1 -Prebuilt"
