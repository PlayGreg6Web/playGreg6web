<#
.SYNOPSIS
  Rebuild the ready-to-play native build and swap it in atomically, only when the build succeeded.
  The build goes into a temp dir next to the target (using the object/lift/recomp caches); the previous
  playable copy is renamed aside and only deleted after the new one is in place, so a failed build never
  breaks it. Writes prebuilt.json (source commit) into the build. Play it with: .\tools\play_native.ps1 -Prebuilt
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
    [string]$GameDir = $env:GTA6_GAME_DIR
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
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
