<#
.SYNOPSIS
  Build (if stale, with the Dawn native executor on) and play natively: a live window, audio out, keyboard/pad.
  Thin wrapper around tools/run_native_boot.py --play (uses the object/lift/recomp caches). At the end of every run the boot summary (tools/boot_summary.py) is printed.
.EXAMPLE
  .\tools\play_native.ps1 -MaxMs 120000
.NOTES
  Shader pack: -ShaderPack, env PLAYGTA6_NATIVE_SHADER_PACK, else the newest *.wgsl.pack under work/ (or
  env PLAYGTA6_SHADER_PACK_DIR); the pick is printed. Saves persist under %LOCALAPPDATA%\playgta6\saves;
  -Record writes your real keyboard/pad input to %LOCALAPPDATA%/playgta6/recordings/<name>.txt (-RecordFile: another
  path); replay it with run_native_boot.py --input-script <file>. -FreshSetup uses a new empty save dir so the setup screens appear. F11 toggles fullscreen. -PresentMode auto|fifo|mailbox|immediate picks the window present mode (auto: mailbox, immediate, fifo; unsupported falls back, logged); -MaxFps N caps the host present rate (0 = off, guest pacing unchanged). -ConsoleTiming honors the title's sync interval (30 fps menus). Stop: close the window or Ctrl+C;
  without -MaxMs the run lasts 2 h. -Update: if the checkout has no uncommitted tracked changes, switch to main when detached and fast-forward to origin/main (never resets; otherwise prints why and continues); every start warns when behind origin/main. See docs/PLAY_LOCALLY.md. Game files stay outside the repo.
  -Prebuilt: play the ready-made build in $env:PLAYGTA6_PREBUILT_DIR (default %LOCALAPPDATA%\playgta6\prebuilt, made by tools/update_prebuilt.ps1) with -TrustBuild; never builds, prints its source commit and how far behind origin/main it is.
  -Keymap <file> rebinds keys/mouse (default %LOCALAPPDATA%\playgta6\keymap.ini, written on first run; see docs).
#>
param(
    [string]$Exe = $env:PLAYGTA6_EXE,
    [string]$GameDir = $env:GTA6_GAME_DIR,
    [string]$ShaderPack = $env:PLAYGTA6_NATIVE_SHADER_PACK,
    [string]$SaveDir = "",
    [ValidateSet("none", "start-screen", "menu-flow", "menu-flow-2", "menu-flow-3", "menu-flow-4", "menu-to-ingame")]
    [string]$InputScript = "none",
    [string]$BuildDir = "work/native-boot",
    [string]$Name = "",
    [int]$MaxMs = 7200000,
    [ValidateSet("auto", "fifo", "mailbox", "immediate")]
    [string]$PresentMode = "auto",
    [int]$MaxFps = 0,
    [switch]$ConsoleTiming,
    [switch]$FreshSetup,
    [switch]$Record,
    [string]$RecordFile = "",
    [string]$Keymap = "",
    [switch]$NoBuild,
    [switch]$TrustBuild,
    [switch]$Update,
    [switch]$Prebuilt,
    [switch]$DryRun,
    [switch]$Help
)
$ErrorActionPreference = "Stop"
if ($Help) {
    @"
play_native.ps1: build (if stale) and play natively.
  Setup once (new PowerShell window after): PLAYGTA6_EXE, GTA6_GAME_DIR, PLAYGTA6_NATIVE_SHADER_PACK
  Play:     .\tools\play_native.ps1 -Update
  -Update        fast-forward to origin/main first (never discards local changes)
  -FreshSetup    empty save dir so the setup screens appear
  -PresentMode   auto|fifo|mailbox|immediate     -MaxFps N  (0 = off)
  -ConsoleTiming honor the title's sync interval (menus at the console's 30 fps instead of 60)
  -Record        save real input to %LOCALAPPDATA%\playgta6\recordings   -Keymap <file>  rebind keys
  -Prebuilt      play the ready-made build (no compile; refresh it with tools\update_prebuilt.ps1)
  -NoBuild / -TrustBuild / -DryRun / -Name <run>
  Keys: WASD stick, Enter/Space A, Backspace B, Tab View, Esc Menu, F11 fullscreen, F12 screenshot (PNG).
  Mouse look: keymap.ini [mouse] look = on; F8 releases/re-captures the cursor, a click re-captures.
  Window hotkeys (rebind in keymap.ini [hotkeys]; duplicates are a load error): fullscreen F11, screenshot F12,
  mouse_release F8, overlay F3; --show-target only: target_mip Ctrl+F7, target_layer Ctrl+F8, target_prev Ctrl+F9, target_next Ctrl+F10.
  Docs: docs/PLAY_LOCALLY.md
"@ | Write-Host
    exit 0
}
$repo = Split-Path -Parent $PSScriptRoot
if ($Prebuilt) {
    if ($Update) { throw "-Prebuilt never rebuilds or updates; run tools\update_prebuilt.ps1 to refresh the build." }
    $pdir = if ($env:PLAYGTA6_PREBUILT_DIR) { $env:PLAYGTA6_PREBUILT_DIR } else { Join-Path $env:LOCALAPPDATA "playgta6\prebuilt" }
    $pjson = Join-Path $pdir "prebuilt.json"
    if (-not (Test-Path -LiteralPath $pjson)) { throw "No prebuilt build at $pdir (prebuilt.json missing). Create it with tools\update_prebuilt.ps1." }
    $BuildDir = $pdir; $TrustBuild = $true
    $pc = (Get-Content -LiteralPath $pjson -Raw | ConvertFrom-Json).commit
    $behind = "unknown"
    try {
        & git -C $repo fetch --quiet origin main 2>$null
        $n = (& git -C $repo rev-list --count "$pc..origin/main" 2>$null)
        if ($LASTEXITCODE -eq 0 -and "$n" -match '^\d+$') { $behind = $n }
    } catch { }
    Write-Host "[prebuilt] $pdir built from $pc ($behind commits behind origin/main). Refresh: .\tools\update_prebuilt.ps1"
    if ($DryRun) { Write-Host "[dry-run] python tools/prebuilt_stale.py --prebuilt $pdir --repo $repo" }
    try { & python (Join-Path $PSScriptRoot "prebuilt_stale.py") --prebuilt $pdir --repo $repo } catch { }
}
if ($Update) { & python (Join-Path $PSScriptRoot "repo_update.py") update --repo $repo }
if (-not $Prebuilt) { & python (Join-Path $PSScriptRoot "repo_update.py") check --repo $repo }
if (-not $Exe -or -not $GameDir) { throw "Pass -Exe and -GameDir (or set PLAYGTA6_EXE / GTA6_GAME_DIR)." }
if ($ShaderPack -and -not (Test-Path -LiteralPath $ShaderPack)) { throw "Shader pack not found: $ShaderPack" }
if (-not $Name) { $Name = "play-native-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
$preset = if ($InputScript -eq "none") { "long" } else { $InputScript }
$argv = @("tools/run_native_boot.py", "--preset", $preset, "--name", $Name, "--play",
          "--exe", $Exe, "--game-dir", $GameDir, "--build-dir", $BuildDir,
          "--max-ms", $MaxMs, "--watchdog-ms", ($MaxMs + 30000))
# Without -ShaderPack run_native_boot.py picks one (env var, then the newest *.wgsl.pack under work/) and prints it.
if ($ShaderPack) { $argv += @("--shader-pack", $ShaderPack) }
if ($Record -or $RecordFile) { $argv += "--record-input"; if ($RecordFile) { $argv += $RecordFile } }
if ($Keymap) { $argv += @("--keymap", $Keymap) }
if ($PresentMode -ne "auto") { $argv += @("--present-mode", $PresentMode) }
if ($ConsoleTiming) { $argv += @("--present-interval", "honor") }
if ($MaxFps -gt 0) { $argv += @("--max-fps", $MaxFps) }
if ($FreshSetup) { $argv += "--fresh-setup" }
if ($NoBuild) { $argv += "--no-build" }
if ($TrustBuild) { $argv += "--trust-build" }
if ($SaveDir) { $argv += @("--save-dir", $SaveDir) }
if ($DryRun) {
    Write-Host "[dry-run] build dir: $BuildDir; run name: $Name; input script: $InputScript (preset $preset); shader pack: $(if ($ShaderPack) { $ShaderPack } else { '(auto-picked by run_native_boot.py)' })"
    Write-Host "[dry-run] python $($argv -join ' ')"
    exit 0
}
Push-Location $repo
try { & python @argv; exit $LASTEXITCODE } finally { Pop-Location }
