<#
.SYNOPSIS
  Build (if stale) and watch the browser build in a headed Chrome, locally.
  Thin wrapper around tools/run_browser_boot.py (local COOP/COEP server + Chrome with its own profile).
.EXAMPLE
  .\tools\play_local.ps1 -Exe D:\game\game.exe -GameDir D:\game\package
.NOTES
  Stop: close the Chrome window or press Ctrl+C here; the server stops with this script.
  -Prebuilt: play the build made by tools/update_prebuilt.ps1 -Web (dir: -PrebuiltDir, else $env:PLAYGTA6_PREBUILT_WEB_DIR, else %LOCALAPPDATA%\playgta6\prebuilt-web; -BuildDir is ignored). Never builds (-TrustBuild semantics); prints its source commit and how far behind origin/main it is.
  -StreamShaders [-StreamPack file]: serve the FULL std pack (WGPK v2 + .reflect.json; -StreamPack, else $env:PLAYGTA6_NATIVE_SHADER_PACK) with HTTP Range and open the page with ?streamShaders=1 (staged by hard link into a separate dir). With -Prebuilt this is the default when that pack exists; -NoStreamShaders forces the staged v1 subset.
  -Update: if the checkout has no uncommitted tracked changes, switch to main when detached and fast-forward to origin/main (never resets; otherwise prints why and continues). Every start warns when behind origin/main.
  See docs/PLAY_LOCALLY.md. Game files stay outside the repo.
#>
param(
    [string]$Exe = $env:PLAYGTA6_EXE,
    [string]$GameDir = $env:GTA6_GAME_DIR,
    [int]$Port = 8141,
    [string]$BuildDir = "work/browser-boot",
    [string]$Chrome = $env:CHROME_PATH,
    [ValidateSet("none", "start-screen", "menu-flow", "calibration-skip", "intro-watch", "legal-notices", "chrome-menu-flow", "chrome-menu-flow-2", "chrome-menu-flow-4", "chrome-menu-flow-5", "chrome-full-boot", "menu-to-ingame", "chrome-to-ingame")]
    [string]$InputScript = "calibration-skip",
    [string]$ShaderPack = $env:PLAYGTA6_SHADER_PACK,
    [string]$Name = "",
    [int]$DurationS = 7200,
    [switch]$NoBuild,
    [switch]$TrustBuild,
    [switch]$Prebuilt,
    [switch]$StreamShaders,
    [switch]$NoStreamShaders,
    [string]$StreamPack = $env:PLAYGTA6_NATIVE_SHADER_PACK,
    [string]$PrebuiltDir = $env:PLAYGTA6_PREBUILT_WEB_DIR,
    [switch]$Update,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($Update) { & python (Join-Path $PSScriptRoot "repo_update.py") update --repo $repo }
& python (Join-Path $PSScriptRoot "repo_update.py") check --repo $repo
if ($Prebuilt) {
    if (-not $PrebuiltDir) { $PrebuiltDir = Join-Path $env:LOCALAPPDATA "playgta6\prebuilt-web" }
    $PrebuiltDir = [System.IO.Path]::GetFullPath($PrebuiltDir)
    if (-not (Test-Path -LiteralPath (Join-Path $PrebuiltDir "prebuilt.json"))) {
        throw "No prebuilt build in $PrebuiltDir. Create it with: .\tools\update_prebuilt.ps1 -Web"
    }
    $TrustBuild = $true
    & python (Join-Path $PSScriptRoot "prebuilt_web.py") info --prebuilt $PrebuiltDir --repo $repo
    try { & python (Join-Path $PSScriptRoot "prebuilt_stale.py") --prebuilt $PrebuiltDir --repo $repo --web } catch { }
    # Full std pack (WGPK v2) streamed with HTTP Range (?streamShaders=1); the staged v1 subset is the fallback.
    if (-not $NoStreamShaders -and -not $ShaderPack -and -not $StreamShaders) {
        $found = (& python (Join-Path $PSScriptRoot "prebuilt_web.py") stream-pack --env "$StreamPack") | Select-Object -Last 1
        if ($found) { $StreamShaders = $true; $StreamPack = $found }
    }
    if (-not $ShaderPack -and -not $StreamShaders) {
        $staged = Get-ChildItem -LiteralPath (Join-Path $PrebuiltDir "build\staged-shaders") -Filter *.pack -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($staged) { $ShaderPack = $staged.FullName }
    }
}
if ($StreamShaders -and $NoStreamShaders) { throw "-StreamShaders and -NoStreamShaders are exclusive." }
if ($StreamShaders) {
    if (-not $StreamPack) { throw "-StreamShaders needs a v2 pack: pass -StreamPack <file> or set PLAYGTA6_NATIVE_SHADER_PACK." }
    $ShaderPack = (& python (Join-Path $PSScriptRoot "prebuilt_web.py") stream-pack --explicit "$StreamPack") | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or -not $ShaderPack) { throw "-StreamShaders: no usable WGPK v2 pack ($StreamPack)." }
    Write-Host "Streaming the full shader pack $ShaderPack (WGPK v2, HTTP Range, ?streamShaders=1)."
}
if (-not $Exe -or -not $GameDir) { throw "Pass -Exe and -GameDir (or set PLAYGTA6_EXE / GTA6_GAME_DIR)." }
if (-not $ShaderPack) {
    $default = Join-Path $repo "work/splash.wgsl.pack"
    if (Test-Path -LiteralPath $default) { $ShaderPack = $default }
}
if (-not $ShaderPack -or -not (Test-Path -LiteralPath $ShaderPack)) {
    $msg = "No shader pack found (without one the page stays on its loading artwork). Build one:`n" +
           "  cargo run --release -p playgta6-shaders -- pack-rdna2 <input-dir> --out work/splash.wgsl.pack`n" +
           "then pass -ShaderPack <file> or set PLAYGTA6_SHADER_PACK (work/splash.wgsl.pack is picked up automatically)."
    if ($DryRun) { Write-Warning $msg } else { throw $msg }
}
if (-not $NoBuild -and -not $TrustBuild -and -not (Get-Command emcc -ErrorAction SilentlyContinue) -and -not $env:EMSDK) {
    & python (Join-Path $PSScriptRoot "emsdk_locator.py")
    if ($LASTEXITCODE -ne 0) {
        $m = "No Emscripten found (emcc not on PATH, EMSDK unset, no work/**/emsdk). Fix: git clone https://github.com/emscripten-core/emsdk work/emsdk; cd work/emsdk; .\emsdk install 7.0.1; .\emsdk activate 7.0.1 (or activate an existing emsdk in this shell)."
        if ($DryRun) { Write-Warning $m } else { throw $m }
    }
}
if (-not $Name) { $Name = "play-local-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
$argv = @("tools/run_browser_boot.py", "--preset", "long", "--name", $Name, "--headed",
          "--exe", $Exe, "--game-dir", $GameDir, "--port", $Port, "--build-dir", $BuildDir,
          "--duration-s", $DurationS, "--report-to-log")
if ($Chrome) { $argv += @("--chrome", $Chrome) }
if ($ShaderPack) { $argv += @("--shader-pack", $ShaderPack) }
if ($InputScript -and $InputScript -ne "none") { $argv += @("--input-script", $InputScript) }
if ($StreamShaders) {
    $argv += @("--stream-shaders", "--shader-stage-dir", (Join-Path $repo "work\staged-shaders-stream"))
    if ($Prebuilt) { $argv[$argv.IndexOf("--shader-stage-dir") + 1] = (Join-Path $PrebuiltDir "build\staged-shaders-stream") }
}
if ($Prebuilt) {
    $argv[$argv.IndexOf("--build-dir") + 1] = (Join-Path $PrebuiltDir "build")
    $argv += @("--boot-dir", (Join-Path $PrebuiltDir "boot"), "--out-dir", (Join-Path $PrebuiltDir "web\dist"), "--web-dir", (Join-Path $PrebuiltDir "web"))
}
if ($NoBuild) { $argv += "--no-build" }
if ($TrustBuild) { $argv += "--trust-build" }
if ($DryRun) { $argv += "--dry-run" }
Write-Host "Serving on http://127.0.0.1:$Port/ ; stop with Ctrl+C or by closing Chrome (run lasts up to $DurationS s)."
Push-Location $repo
try { & python @argv; exit $LASTEXITCODE } finally { Pop-Location }
