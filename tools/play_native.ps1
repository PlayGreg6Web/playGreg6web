<#
.SYNOPSIS
  Build (if stale, with the Dawn native executor on) and play natively: a live window, audio out, keyboard/pad.
  Thin wrapper around tools/run_native_boot.py --play (uses the object/lift/recomp caches).
.EXAMPLE
  .\tools\play_native.ps1 -MaxMs 120000
.NOTES
  Shader pack: -ShaderPack, env PLAYGTA6_NATIVE_SHADER_PACK, else the newest *.wgsl.pack under work/ (or
  env PLAYGTA6_SHADER_PACK_DIR); the pick is printed. Saves persist under %LOCALAPPDATA%\playgta6\saves;
  -Record writes your real keyboard/pad input to %LOCALAPPDATA%/playgta6/recordings/<name>.txt (-RecordFile: another
  path); replay it with run_native_boot.py --input-script <file>. -FreshSetup uses a new empty save dir so the setup screens appear. F11 toggles fullscreen. -PresentMode auto|fifo|mailbox|immediate picks the window present mode (auto: mailbox, immediate, fifo; unsupported falls back, logged); -MaxFps N caps the host present rate (0 = off, guest pacing unchanged). Stop: close the window or Ctrl+C;
  without -MaxMs the run lasts 2 h. -Update: if the checkout has no uncommitted tracked changes, switch to main when detached and fast-forward to origin/main (never resets; otherwise prints why and continues); every start warns when behind origin/main. See docs/PLAY_LOCALLY.md. Game files stay outside the repo.
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
    [switch]$FreshSetup,
    [switch]$Record,
    [string]$RecordFile = "",
    [switch]$NoBuild,
    [switch]$TrustBuild,
    [switch]$Update,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($Update) { & python (Join-Path $PSScriptRoot "repo_update.py") update --repo $repo }
& python (Join-Path $PSScriptRoot "repo_update.py") check --repo $repo
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
if ($PresentMode -ne "auto") { $argv += @("--present-mode", $PresentMode) }
if ($MaxFps -gt 0) { $argv += @("--max-fps", $MaxFps) }
if ($FreshSetup) { $argv += "--fresh-setup" }
if ($NoBuild) { $argv += "--no-build" }
if ($TrustBuild) { $argv += "--trust-build" }
if ($DryRun) { $argv += "--dry-run" }
if ($SaveDir) { $argv += @("--save-dir", $SaveDir) }
Push-Location $repo
try { & python @argv; exit $LASTEXITCODE } finally { Pop-Location }
