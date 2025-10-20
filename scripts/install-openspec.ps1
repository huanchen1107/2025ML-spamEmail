<#
Installs Fission-AI/OpenSpec into "C:\\Program Files\\OpenSpec" and makes it
globally available via an `openspec` command (CMD + PowerShell shims).

Usage (run in elevated PowerShell):
  .\scripts\install-openspec.ps1 -RepoUrl https://github.com/Fission-AI/OpenSpec.git

Notes:
- Requires admin rights (writes to Program Files and system PATH).
- Requires Git. Node.js/npm are recommended (repo is likely JS/TS).
- The script clones the repo, installs/builds (Node-first), copies to Program Files,
  creates `openspec.cmd` and `openspec.ps1` shims, and updates PATH.
#>

param(
  [Parameter(Mandatory = $false)]
  [string]$RepoUrl = "https://github.com/Fission-AI/OpenSpec.git",

  [Parameter(Mandatory = $false)]
  [string]$Ref,  # branch, tag, or commit

  [Parameter(Mandatory = $false)]
  [string]$InstallDir = "C:\\Program Files\\OpenSpec",

  [Parameter(Mandatory = $false)]
  [string]$BinBaseName = "openspec",  # produces openspec.cmd / openspec.ps1

  [switch]$Force  # overwrite existing install dir
)

# --- Helpers ---
$ErrorActionPreference = 'Stop'

function Require-Admin() {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $p  = [Security.Principal.WindowsPrincipal]::new($id)
  if (-not $p.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    throw "Please run this script in an elevated PowerShell (Run as administrator)."
  }
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
  }
}

function Add-ToSystemPath([string]$Dir) {
  $existing = [Environment]::GetEnvironmentVariable('Path','Machine')
  $parts = ($existing -split ';' | ForEach-Object { $_.Trim() }) | Where-Object { $_ }
  if ($parts -notcontains $Dir) {
    $new = ($parts + $Dir) -join ';'
    [Environment]::SetEnvironmentVariable('Path', $new, 'Machine')
    Write-Host "Added to system PATH: $Dir"
    Write-Host "Open a new terminal to pick up the updated PATH."
  } else {
    Write-Host "Directory already in system PATH: $Dir"
  }
}

function Has-Command([string]$Name) {
  return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Run-InDir([string]$Dir, [scriptblock]$Block) {
  Push-Location $Dir
  try { & $Block } finally { Pop-Location }
}

function Parse-PackageJson([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  try { return (Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json) } catch { return $null }
}

function Detect-Node-Entry([object]$Pkg, [string]$DefaultBaseName) {
  if ($null -eq $Pkg) { return $null }
  if ($Pkg.bin -is [string]) { return $Pkg.bin }
  if ($Pkg.bin) {
    $names = $Pkg.bin.PSObject.Properties.Name
    if ($names -contains $DefaultBaseName) { return $Pkg.bin.$DefaultBaseName }
    if ($names.Count -eq 1) { $single = $names[0]; return $Pkg.bin.$single }
  }
  if ($Pkg.main) { return $Pkg.main }
  return $null
}

function New-CmdShim([string]$ShimPath, [string]$TargetJsRel) {
  $content = @'
@echo off
setlocal
node "%~dp0app\__ENTRY__" %*
endlocal
'@
  $content = $content.Replace('__ENTRY__', $TargetJsRel)
  Set-Content -LiteralPath $ShimPath -Value $content -Encoding ASCII -Force
}

function New-PwshShim([string]$ShimPath, [string]$TargetJsRel) {
  $content = @"
param([Parameter(ValueFromRemainingArguments = $true)] [string[]]$Args)
$script = Join-Path $PSScriptRoot "app/$TargetJsRel"
& node $script @Args
"@
  Set-Content -LiteralPath $ShimPath -Value $content -Encoding UTF8 -Force
}

# --- Main ---
Require-Admin

Write-Host "Installing OpenSpec from: $RepoUrl" -ForegroundColor Yellow
if ($Ref) { Write-Host "Ref: $Ref" -ForegroundColor Yellow }

if ((Test-Path -LiteralPath $InstallDir) -and -not $Force) {
  throw "Install directory already exists: $InstallDir. Use -Force to overwrite."
}

$tmpRoot = Join-Path $env:TEMP ("openspec_install_" + [IO.Path]::GetFileNameWithoutExtension([IO.Path]::GetRandomFileName()))
Ensure-Dir $tmpRoot
$tmpClone = Join-Path $tmpRoot "repo"

if (-not (Has-Command git)) { throw "Git is required (not found in PATH)." }

# Clone repo
Write-Host "Cloning repository..." -ForegroundColor Cyan
Run-InDir $tmpRoot { git clone --depth 1 $RepoUrl repo }
if ($Ref) {
  Run-InDir $tmpClone { git fetch origin $Ref --depth 1 }
  Run-InDir $tmpClone { git checkout $Ref }
}

# Detect project type
$isNode = Test-Path -LiteralPath (Join-Path $tmpClone 'package.json')
$isCargo = Test-Path -LiteralPath (Join-Path $tmpClone 'Cargo.toml')
$isGo    = Test-Path -LiteralPath (Join-Path $tmpClone 'go.mod')
$isPy    = (Test-Path -LiteralPath (Join-Path $tmpClone 'pyproject.toml')) -or (Test-Path -LiteralPath (Join-Path $tmpClone 'setup.py'))

# Prepare install dir
if (Test-Path -LiteralPath $InstallDir) { Remove-Item -Recurse -Force -LiteralPath $InstallDir }
Ensure-Dir $InstallDir

if ($isNode) {
  Write-Host "Detected Node.js project" -ForegroundColor Green
  if (-not (Has-Command node)) { throw "Node.js is required (node not found)." }
  if (-not (Has-Command npm))  { throw "npm is required (npm not found)." }

  # Install deps (prefer CI-friendly path)
  Run-InDir $tmpClone {
    if (Test-Path -LiteralPath 'package-lock.json') { npm ci }
    else { npm install }
  }
  # Optional build
  $pkg = Parse-PackageJson (Join-Path $tmpClone 'package.json')
  $hasBuild = $false
  if ($pkg -and $pkg.scripts -and $pkg.scripts.build) { $hasBuild = $true }
  if ($hasBuild) {
    Write-Host "Running build script" -ForegroundColor Cyan
    Run-InDir $tmpClone { npm run build }
  }

  # Copy project into Program Files
  $appDir = Join-Path $InstallDir 'app'
  Ensure-Dir $appDir
  Write-Host "Copying project to $appDir" -ForegroundColor Cyan
  # Copy-Item sometimes struggles with node_modules symlinks; use robocopy if available
  if (Has-Command robocopy) {
    $null = robocopy $tmpClone $appDir /MIR /NFL /NDL /NJH /NJS /NP /XF .git /XD .git
    if ($LASTEXITCODE -gt 8) { throw "robocopy failed with code $LASTEXITCODE" }
  } else {
    Copy-Item -Path (Join-Path $tmpClone '*') -Destination $appDir -Recurse -Force -ErrorAction Stop
  }

  # Determine CLI entry
  $entry = Detect-Node-Entry $pkg $BinBaseName
  if (-not $entry) {
    throw "Unable to determine CLI entry from package.json (bin/main not found)."
  }
  $entryRel = $entry -replace "^\\.\\\\", '' -replace "^\./", ''

  # Create shims
  $cmdShim = Join-Path $InstallDir ("$BinBaseName.cmd")
  $psShim  = Join-Path $InstallDir ("$BinBaseName.ps1")
  New-CmdShim -ShimPath $cmdShim -TargetJsRel $entryRel
  New-PwshShim -ShimPath $psShim -TargetJsRel $entryRel
  Write-Host "Created shims: $cmdShim, $psShim" -ForegroundColor Green

} elseif ($isCargo) {
  Write-Host "Detected Rust project" -ForegroundColor Green
  if (-not (Has-Command cargo)) { throw "Rust toolchain (cargo) is required." }
  Run-InDir $tmpClone { cargo build --release }
  $exe = Get-ChildItem -LiteralPath (Join-Path $tmpClone 'target\release') -Filter *.exe -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match $BinBaseName } | Select-Object -First 1
  if (-not $exe) { $exe = Get-ChildItem -LiteralPath (Join-Path $tmpClone 'target\release') -Filter *.exe -File | Select-Object -First 1 }
  if (-not $exe) { throw "Built executable not found in target\\release" }
  Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $InstallDir "$BinBaseName.exe") -Force
  Write-Host "Installed: " (Join-Path $InstallDir "$BinBaseName.exe") -ForegroundColor Green

} elseif ($isGo) {
  Write-Host "Detected Go project" -ForegroundColor Green
  if (-not (Has-Command go)) { throw "Go toolchain is required (go not found)." }
  $out = Join-Path $InstallDir ("$BinBaseName.exe")
  Run-InDir $tmpClone { go build -o "$out" . }
  Write-Host "Installed: $out" -ForegroundColor Green

} elseif ($isPy) {
  Write-Host "Detected Python project" -ForegroundColor Green
  if (-not (Has-Command python)) { throw "Python is required (python not found)." }
  if (-not (Has-Command pip))    { throw "pip is required (pip not found)." }
  $venvDir = Join-Path $InstallDir 'venv'
  Run-InDir $InstallDir { python -m venv venv }
  $pipExe = Join-Path $venvDir 'Scripts\pip.exe'
  Run-InDir $tmpClone { & $pipExe install . }
  # Try to locate console_script matching BinBaseName in venv Scripts
  $candidate = Get-ChildItem -LiteralPath (Join-Path $venvDir 'Scripts') -Filter "$BinBaseName*.exe" -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($candidate) {
    Copy-Item -LiteralPath $candidate.FullName -Destination (Join-Path $InstallDir ("$BinBaseName.exe")) -Force
  } else {
    # Fallback shim
    $psShim = Join-Path $InstallDir ("$BinBaseName.ps1")
    $content = @"
param([Parameter(ValueFromRemainingArguments = $true)] [string[]]$Args)
& "$(Join-Path $venvDir 'Scripts\python.exe')" -m $BinBaseName @Args
"@
    Set-Content -LiteralPath $psShim -Value $content -Encoding UTF8 -Force
  }
  Write-Host "Installed Python CLI into $InstallDir" -ForegroundColor Green

} else {
  throw "Could not detect project type (no package.json, Cargo.toml, go.mod, or Python config)."
}

# Add to PATH
Add-ToSystemPath -Dir $InstallDir

Write-Host "Done. Open a new terminal and run '$BinBaseName --help'." -ForegroundColor Green
