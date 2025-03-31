# Script para verificar la configuration de R en el sistema

# Solicitar la ruta de installation de R (opcional)
$userRPath = Read-Host "Ingresa la ruta donde esta instalado R (presiona Enter para omitir)"

# Verificar si R esta en el PATH
$rInPath = $null
try {
    $rInPath = Get-Command R -ErrorAction SilentlyContinue
    if ($rInPath) {
        Write-Host "R encontrado en el PATH" -ForegroundColor Green
        Write-Host $rInPath.Source -ForegroundColor Green
    } else {
        Write-Host "R no esta en el PATH del sistema" -ForegroundColor Red
    }
} catch {
    Write-Host "R no esta en el PATH del sistema" -ForegroundColor Red
}

# Verificar variable de entorno R_HOME
$rHome = [Environment]::GetEnvironmentVariable("R_HOME", "Machine")
if ($rHome) {
    Write-Host "Variable R_HOME encontrada" -ForegroundColor Green
    Write-Host $rHome -ForegroundColor Green
    
    # Verificar si la ruta existe
    if (Test-Path $rHome) {
        Write-Host "La ruta de R_HOME existe" -ForegroundColor Green
    } else {
        Write-Host "La ruta de R_HOME no existe" -ForegroundColor Red
    }
} else {
    Write-Host "Variable R_HOME no esta definida" -ForegroundColor Red
}

# Buscar posibles instalaciones de R
Write-Host "`nBuscando posibles instalaciones de R..." -ForegroundColor Cyan
$possibleRPaths = @(
    "C:\Program Files\R",
    "C:\Program Files (x86)\R"
)

# Agregar la ruta proporcionada por el usuario si se dio
if ($userRPath -and (Test-Path $userRPath)) {
    $possibleRPaths += $userRPath
}

$foundR = $false
foreach ($path in $possibleRPaths) {
    if (Test-Path $path) {
        $rVersions = Get-ChildItem $path -Directory | Where-Object { $_.Name -match "R-\d" }
        if ($rVersions) {
            $foundR = $true
            Write-Host "Instalaciones de R encontradas en" -ForegroundColor Green
            Write-Host "$path" -ForegroundColor Green
            
            foreach ($version in $rVersions) {
                Write-Host "   - $($version.Name)" -ForegroundColor Green
                Write-Host "     Ruta completa: $($version.FullName)" -ForegroundColor Green
                
                # Sugerir configuracion para esta version
                $binPath = Join-Path $version.FullName "bin"
                Write-Host "`n   Para configurar esta version:" -ForegroundColor Yellow
                Write-Host "   1. Agregar al PATH: $binPath" -ForegroundColor Yellow
                Write-Host "   2. Establecer R_HOME: $($version.FullName)" -ForegroundColor Yellow
                
                # Preguntar si desea configurar esta version
                $configureThis = Read-Host "`n   Deseas configurar esta version de R? (s/n)"
                if ($configureThis -eq "s") {
                    # Configurar PATH
                    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
                    if (-not $currentPath.Contains($binPath)) {
                        [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$binPath", "Machine")
                        Write-Host "   PATH actualizado con $binPath" -ForegroundColor Green
                    } else {
                        Write-Host "   El PATH ya contiene esta ruta" -ForegroundColor Yellow
                    }
                    
                    # Configurar R_HOME
                    [Environment]::SetEnvironmentVariable("R_HOME", $version.FullName, "Machine")
                    Write-Host "   R_HOME establecido como $($version.FullName)" -ForegroundColor Green
                    
                    Write-Host "`n   Configuracion completada. Reinicia tu terminal o IDE para aplicar los cambios." -ForegroundColor Green
                    Write-Host "   Despues intenta instalar rpy2 con: pip install rpy2" -ForegroundColor Green
                    break
                }
            }
        }
    }
}

if (-not $foundR) {
    Write-Host "No se encontraron instalaciones de R en las rutas estandar" -ForegroundColor Red
    
    # Si el usuario proporciono una ruta pero no se encontro R
    if ($userRPath -and -not (Test-Path $userRPath)) {
        Write-Host "La ruta que proporcionaste no existe: $userRPath" -ForegroundColor Red
    }
}