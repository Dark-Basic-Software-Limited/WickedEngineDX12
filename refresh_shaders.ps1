# =============================================================================
# refresh_shaders.ps1 - Force stale engine shader .cso files to recompile.
#
# WHY THIS EXISTS
#   Engine shaders (WickedEngine/shaders/*.hlsl) are compiled lazily AT RUNTIME
#   on first launch. wiRenderer::LoadShader() only recompiles a .cso when
#   wi::shadercompiler::IsShaderOutdated() says it is stale. That check trusts a
#   .cso as "fresh" when (a) its .wishadermeta is missing, or (b) the meta's
#   recorded dependency paths (stored RELATIVE to the deploy shaders folder)
#   don't resolve on this machine - unresolved deps are silently skipped. Both
#   cases leave a STALE .cso in use. That silently broke grass on 2026-07-23: a
#   stale hairparticle_simulateCS.cso, compiled against an OLD constant layout
#   (ShaderInterop_HairParticle.h), read the per-strand filters from the wrong
#   offsets and zeroed every strand (grass simulated ~59ms GPU but drew nothing).
#
# WHAT THIS DOES (no fragile relative-path meta tracking - a plain "is the
# compiled output older than its source?" freshness check that CANNOT silently
# fail, resolving each shader's REAL #include chain from disk):
#   For each deploy .cso that has a matching .wishadermeta (i.e. an ENGINE shader
#   the runtime knows how to recompile - this automatically EXCLUDES the GG*/test
#   game shaders, which are built by the game's MSBuild FxCompile, have no meta,
#   and would vanish if deleted), delete the .cso (+ its .wishadermeta) when the
#   compiled output is OLDER than its source .hlsl OR any header that .hlsl
#   transitively #includes (globals.hlsli, ShaderInterop_*.h, ...). A deleted
#   .cso is missing, so LoadShader's existence check forces a clean recompile
#   from the CURRENT source on next launch. This is selective (only shaders whose
#   own dependency chain actually changed) and converges (a freshly recompiled
#   .cso is newer than its sources, so it is never re-flagged). Worst case is a
#   harmless extra recompile - never a stale shader.
#
# USAGE
#   powershell -NoProfile -ExecutionPolicy Bypass -File refresh_shaders.ps1
#   ...             -All      wipe every engine (has-meta) .cso, force full recompile
#   ...             -DryRun   print what would be deleted, delete nothing
#   ...             -DeployDir <path>  -SourceDir <path>   override defaults
#
# Called automatically from build_wicked.bat (engine) and the game build.bat.
# =============================================================================
param(
    [string]$DeployDir = "D:\DEV\BUILD\GameGuru Wicked MAX Build Area\Max\shaders",
    [string]$SourceDir = "D:\max\WickedEngineDX12\WickedEngine\shaders",
    [switch]$All,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DeployDir)) {
    Write-Host "refresh_shaders: deploy shaders folder not found, nothing to do:`n  $DeployDir"
    exit 0
}
if (-not (Test-Path -LiteralPath $SourceDir)) {
    Write-Host "refresh_shaders: engine shader source folder not found, nothing to do:`n  $SourceDir"
    exit 0
}
$SourceDirFull = (Resolve-Path -LiteralPath $SourceDir).Path

# --- Index every source file (.hlsl/.hlsli/.h) by lowercase base name so a cso
#     can be mapped back to its top-level .hlsl even through permutation suffixes. ---
$srcByBase = @{}
foreach ($f in Get-ChildItem -LiteralPath $SourceDirFull -Recurse -Include *.hlsl,*.hlsli,*.h -File) {
    $key = $f.BaseName.ToLowerInvariant()
    if (-not $srcByBase.ContainsKey($key)) { $srcByBase[$key] = $f.FullName }
}

# --- Memoized transitive-include freshness: newest write-time of a source file
#     and everything it #includes (recursively, cycle-guarded). ---
$depTimeCache = @{}
function Get-DepTime([string]$path, $stack) {
    $full = [System.IO.Path]::GetFullPath($path)
    $ck = $full.ToLowerInvariant()
    if ($depTimeCache.ContainsKey($ck)) { return $depTimeCache[$ck] }
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { return 0L }
    if ($stack.Contains($ck)) { return 0L }         # include cycle
    [void]$stack.Add($ck)

    $item = Get-Item -LiteralPath $full
    $newest = $item.LastWriteTimeUtc.Ticks
    $dir = $item.DirectoryName
    foreach ($line in [System.IO.File]::ReadAllLines($full)) {
        # Only follow project includes: #include "path". Angle-bracket includes
        # (<utility>, <string>, ... in the shared CPU/GPU ShaderInterop headers)
        # are system headers and irrelevant to shader staleness.
        $m = [regex]::Match($line, '^\s*#\s*include\s*"([^"]+)"')
        if (-not $m.Success) { continue }
        $inc = $m.Groups[1].Value
        # Resolve the include against the including file's dir, then the shaders root.
        $cand = Join-Path $dir $inc
        if (-not (Test-Path -LiteralPath $cand -PathType Leaf)) { $cand = Join-Path $SourceDirFull $inc }
        if (-not (Test-Path -LiteralPath $cand -PathType Leaf)) { continue }  # header outside the shader tree
        $t = Get-DepTime $cand $stack
        if ($t -gt $newest) { $newest = $t }
    }
    [void]$stack.Remove($ck)
    $depTimeCache[$ck] = $newest
    return $newest
}

# Map a (possibly permutation-suffixed) cso base name to its top-level source
# .hlsl. Permutations append "_DEFINE"; strip trailing _segments until a real
# source name matches. Returns $null if none found.
function Get-SourceHlsl([string]$csoBase) {
    $name = $csoBase.ToLowerInvariant()
    while ($true) {
        if ($srcByBase.ContainsKey($name)) {
            $p = $srcByBase[$name]
            if ([System.IO.Path]::GetExtension($p) -ieq ".hlsl") { return $p }
        }
        $idx = $name.LastIndexOf('_')
        if ($idx -lt 0) { return $null }
        $name = $name.Substring(0, $idx)
    }
}

$deleted = 0
$skippedGame = 0
foreach ($cso in Get-ChildItem -LiteralPath $DeployDir -Filter *.cso -File) {
    $meta = [System.IO.Path]::ChangeExtension($cso.FullName, "wishadermeta")
    if (-not (Test-Path -LiteralPath $meta)) { $skippedGame++; continue }  # game shader - never touch

    $stale = [bool]$All
    if (-not $stale) {
        $srcHlsl = Get-SourceHlsl $cso.BaseName
        if ($srcHlsl) {
            $srcTicks = Get-DepTime $srcHlsl (New-Object 'System.Collections.Generic.HashSet[string]')
            if ($cso.LastWriteTimeUtc.Ticks -lt $srcTicks) { $stale = $true }
        }
    }
    if (-not $stale) { continue }

    if ($DryRun) {
        Write-Host "  would delete: $($cso.Name)"
    } else {
        Remove-Item -LiteralPath $cso.FullName -Force
        Remove-Item -LiteralPath $meta -Force
    }
    $deleted++
}

$verb = if ($DryRun) { "would refresh" } else { "refreshed" }
Write-Host "refresh_shaders: $verb $deleted stale engine shader .cso (recompile on next launch); left $skippedGame game shaders untouched."
exit 0
