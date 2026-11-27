# Dot-sourced by update_prebuilt.ps1 (and tests/test_prebuilt_swap.ps1): swap a freshly built dir into place with bounded retries.

# Run a native command with its stderr lines forwarded to the host (information stream) instead of the error stream, so a progress
# line on stderr can never become a terminating error or NativeCommandError noise, whatever $ErrorActionPreference the caller or a
# redirection (*>, 2>&1, Start-Job, a parent that merges our stderr) has. The exit code stays in $LASTEXITCODE: callers must check it.
function Invoke-Native {
    $exe, $rest = $args
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        & $exe @rest 2>&1 | ForEach-Object { Write-Host ($(if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" })) }
    } finally { $ErrorActionPreference = $eap }
}

function Invoke-WithRetry {
    param([scriptblock]$Action, [int]$Tries = 5, [int]$DelayMs = 500)
    for ($i = 1; $i -le $Tries; $i++) {
        try { & $Action; return }
        catch {
            if ($i -ge $Tries) { throw }
            Start-Sleep -Milliseconds ($DelayMs * [math]::Pow(2, $i - 1))   # 0.5, 1, 2, 4 s ~ 7.5 s total at the defaults
        }
    }
}

# Rename $Dir -> $Old, then $Tmp -> $Dir; each rename is retried (transient AV/indexer/just-exited-process handles give
# "Access denied"). If the second step still fails, $Old is renamed back so $Dir is never left missing. $Old is removed at the end
# (a failure there is only a warning: the new build is already in place).
function Invoke-PrebuiltSwap {
    param([string]$Dir, [string]$Tmp, [string]$Old, [int]$Tries = 5, [int]$DelayMs = 500)
    $dirLeaf = Split-Path -Leaf $Dir; $oldLeaf = Split-Path -Leaf $Old
    $hadDir = Test-Path -LiteralPath $Dir
    if ($hadDir) {
        try { Invoke-WithRetry -Tries $Tries -DelayMs $DelayMs -Action { Rename-Item -LiteralPath $Dir -NewName $oldLeaf -ErrorAction Stop } }
        catch {
            throw "swap failed: cannot rename '$Dir' to '$oldLeaf' after $Tries tries ($($_.Exception.Message)). Something still holds a handle inside it (game, AV, indexer, an open shell/Explorer window). Nothing was changed: '$Dir' is intact and the new build is at '$Tmp'."
        }
    }
    try { Invoke-WithRetry -Tries $Tries -DelayMs $DelayMs -Action { Rename-Item -LiteralPath $Tmp -NewName $dirLeaf -ErrorAction Stop } }
    catch {
        $why = $_.Exception.Message
        if ($hadDir) {
            try { Invoke-WithRetry -Tries $Tries -DelayMs $DelayMs -Action { Rename-Item -LiteralPath $Old -NewName $dirLeaf -ErrorAction Stop } }
            catch { throw "swap failed AND rollback failed: '$Dir' is missing; the previous build is at '$Old', the new one at '$Tmp'. Rename '$Old' back to '$dirLeaf' by hand. ($why; rollback: $($_.Exception.Message))" }
        }
        throw "swap failed: cannot rename '$Tmp' to '$dirLeaf' after $Tries tries ($why); a handle is held inside the new build. Rolled back: '$Dir' is the previous build again, the new one stays at '$Tmp'."
    }
    if (Test-Path -LiteralPath $Old) {
        try { Invoke-WithRetry -Tries $Tries -DelayMs $DelayMs -Action { Remove-Item -Recurse -Force -LiteralPath $Old -ErrorAction Stop } }
        catch { Write-Warning "new build is in place, but could not delete the old copy '$Old' ($($_.Exception.Message)); delete it by hand." }
    }
}
