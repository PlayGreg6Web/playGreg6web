<#
.SYNOPSIS
  Rebuild the ready-to-play native build and swap it in atomically, only when the build succeeded.
  The build goes into a temp dir next to the target (using the object/lift/recomp caches); the previous
  playable copy is renamed aside and only deleted after the new one is in place, so a failed build never
  breaks it. Writes prebuilt.json (source commit) into the build. Play it with: .\tools\play_native.ps1 -Prebuilt
  With -Web: the same for the browser build (tools/play_local.ps1 -Prebuilt); dir -Dir, else $env:PLAYGTA6_PREBUILT_WEB_DIR,
  else %LOCALAPPDATA%\playgta6\prebuilt-web. Also needs a shader pack (-ShaderPack / PLAYGTA6_SHADER_PACK) and emsdk;
  reuses the lift cache and the previous object cache.
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
    [string]$ShaderPack = $env:PLAYGTA6_SHADER_PACK
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($Web) {
    $PrebuiltDir = if ($PSBoundParameters.ContainsKey("Dir")) { $Dir } else { $env:PLAYGTA6_PREBUILT_WEB_DIR }
    if (-not $PrebuiltDir) { $PrebuiltDir = Join-Path $env:LOCALAPPDATA "playgta6\prebuilt-web" }
    if (-not $Exe -or -not $GameDir) { throw "Pass -Exe and -GameDir (or set PLAYGTA6_EXE / GTA6_GAME_DIR)." }
    if (-not $ShaderPack) {
        $default = Join-Path $repo "work/splash.wgsl.pack"
        if (Test-Path -LiteralPath $default) { $ShaderPack = $default }
    }
    if (-not $ShaderPack -or -not (Test-Path -LiteralPath $ShaderPack)) { throw "No shader pack: pass -ShaderPack or set PLAYGTA6_SHADER_PACK." }
    if (-not (Get-Command emcc -ErrorAction SilentlyContinue) -and -not $env:EMSDK) {
        & python (Join-Path $PSScriptRoot "emsdk_locator.py")
        if ($LASTEXITCODE -ne 0) { throw "No Emscripten found (emcc not on PATH, EMSDK unset, no work/**/emsdk)." }
    }
    if (-not (Get-Command emcmake -ErrorAction SilentlyContinue) -and $env:EMSDK) {
        $env:PATH = (Join-Path $env:EMSDK "upstream\emscripten") + [System.IO.Path]::PathSeparator + $env:PATH
    }
    $target = [System.IO.Path]::GetFullPath($PrebuiltDir)
    $tmp = "$target.new"
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
    & python (Join-Path $PSScriptRoot "prebuilt_web.py") stamp --repo $repo --dest $tmp
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
if (Test-Path -LiteralPath $tmp) { Remove-Item -Recurse -Force -LiteralPath $tmp }
if (Test-Path -LiteralPath $old) { Remove-Item -Recurse -Force -LiteralPath $old }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dir) | Out-Null
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
# drop what is not needed to run (per-run outputs and object/IR intermediates)
foreach ($n in @("runs", "program\native\harness-build", "program\native\bin\playgta6_boot.pdb")) { $p = Join-Path $tmp $n; if (Test-Path -LiteralPath $p) { Remove-Item -Recurse -Force -LiteralPath $p } }
@{ commit = $commit; dirty = $dirty; built_utc = (Get-Date).ToUniversalTime().ToString("o") } |
    ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $tmp "prebuilt.json")
if (Test-Path -LiteralPath $Dir) { Rename-Item -LiteralPath $Dir -NewName (Split-Path -Leaf $old) }
try { Rename-Item -LiteralPath $tmp -NewName (Split-Path -Leaf $Dir) }
catch { if (Test-Path -LiteralPath $old) { Rename-Item -LiteralPath $old -NewName (Split-Path -Leaf $Dir) }; throw }
if (Test-Path -LiteralPath $old) { Remove-Item -Recurse -Force -LiteralPath $old }
Write-Host "[prebuilt] updated $Dir from $commit. Play: `$env:PLAYGTA6_PREBUILT_DIR='$Dir'; .\tools\play_native.ps1 -Prebuilt"
