<#
    run_bench.ps1 — Benchmark reproductible d'ORACLE (Windows / MinGW64)

    Construit oracle_bench puis le lance sur une liste de modèles, et
    assemble les fragments Markdown dans benchmark/results.md.

    Règle automatiquement le piège PATH/DLL : sans C:\msys64\mingw64\bin dans
    le PATH, l'exe MinGW échoue au lancement avec 0xC000007B (DLL runtime
    introuvable). Ce script prepend ce bin avant build ET run.

    USAGE :
      .\benchmark\run_bench.ps1
      .\benchmark\run_bench.ps1 -Models @("models/qwen2.5-7b-instruct-q4km.gguf") `
                                -Prompt 128 -Gen 128 -Runs 5 -Threads 8
#>
param(
    [string[]]$Models  = @("models/qwen2.5-7b-instruct-q4km.gguf",
                           "models/llama-3.2-3b-q4k.gguf"),
    [int]$Prompt  = 128,
    [int]$Gen     = 128,
    [int]$Runs    = 5,
    [int]$Threads = 0
)

$ErrorActionPreference = "Stop"
$Root     = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build_x64"
$Exe      = Join-Path $BuildDir "oracle_bench.exe"
$OutMd    = Join-Path $PSScriptRoot "results.md"

# --- Toolchain MinGW64 : indispensable au build ET au run -------------------
$MingwBin = $null
foreach ($cand in @("C:\msys64\mingw64\bin", "C:\winlibs\mingw64\bin")) {
    if (Test-Path (Join-Path $cand "g++.exe")) { $MingwBin = $cand; break }
}
if (-not $MingwBin) { Write-Error "MinGW64 introuvable (C:\msys64\mingw64\bin)." }
$env:PATH = "$MingwBin;$env:PATH"

# --- Build de la cible bench ------------------------------------------------
Write-Host "[bench] Build oracle_bench..." -ForegroundColor Cyan
& (Join-Path $Root "build.ps1") "oracle_bench" | Out-Null
if (-not (Test-Path $Exe)) { Write-Error "oracle_bench.exe non produit." }

# --- Runs -------------------------------------------------------------------
$fragments = @()
foreach ($m in $Models) {
    $full = Join-Path $Root $m
    if (-not (Test-Path $full)) { Write-Warning "modèle absent, ignoré : $m"; continue }

    $name    = [IO.Path]::GetFileNameWithoutExtension($m)
    $mdFrag  = Join-Path $PSScriptRoot "_frag_$name.md"
    $jsonOut = Join-Path $PSScriptRoot "result_$name.json"

    Write-Host "[bench] $m  (p=$Prompt g=$Gen r=$Runs t=$Threads)" -ForegroundColor Green
    $bargs = @("-m", $m, "-p", $Prompt, "-g", $Gen, "-r", $Runs,
               "--md", $mdFrag, "--json", $jsonOut)
    if ($Threads -gt 0) { $bargs += @("-t", $Threads) }

    Push-Location $Root
    & $Exe @bargs
    Pop-Location

    if (Test-Path $mdFrag) { $fragments += (Get-Content $mdFrag -Raw); Remove-Item $mdFrag }
}

# --- Assemble results.md ----------------------------------------------------
$header = "# Benchmark results`n`n_Généré par ``benchmark/run_bench.ps1`` le $(Get-Date -Format 'yyyy-MM-dd HH:mm')._`n"
Set-Content -Path $OutMd -Value ($header + "`n" + ($fragments -join "`n")) -Encoding utf8
Write-Host "[bench] Résultats agrégés : $OutMd" -ForegroundColor Cyan
