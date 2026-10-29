@echo off
rem Lance la plateforme AIDGE : le worker (entrainements) et l'interface, dans le venv.
rem Double-cliquer sur ce fichier. Ne pas melanger avec Docker (bases differentes).
rem Fichier volontairement en ASCII : pas d'accents, pour eviter les problemes d'encodage.
setlocal
set "PYTHONIOENCODING=utf-8"

rem Adresse de l'interface et duree maximale d'attente (secondes, sonde toutes les 2 s)
set "URL=http://127.0.0.1:7860"
if not defined AIDGE_ATTENTE_MAX set "AIDGE_ATTENTE_MAX=120"

rem Racine du depot = dossier parent de scripts\ (marche avec espaces et apostrophes)
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" goto novenv

echo.
echo  AIDGE IN SPACE - demarrage de la plateforme
echo  Dossier : %ROOT%
echo.

start "AIDGE - worker" /D "%ROOT%" cmd /k ""%PY%" -m aidge_in_space.worker"
start "AIDGE - interface" /D "%ROOT%" cmd /k ""%PY%" -m aidge_in_space.app"

echo  Deux fenetres viennent de s'ouvrir : "AIDGE - worker" et "AIDGE - interface".
echo  Laissez-les ouvertes : les fermer arrete la plateforme.
echo  Attente de l'interface sur %URL% (au plus %AIDGE_ATTENTE_MAX% s)...

rem Sonde : curl.exe (livre avec Windows 10/11), sinon PowerShell
set "SONDE=ps"
where curl.exe >nul 2>&1
if not errorlevel 1 set "SONDE=curl"
set /a ELAPSED=0

:sonde
if "%SONDE%"=="curl" (curl.exe -s -o nul --max-time 2 "%URL%") else (powershell -NoProfile -NonInteractive -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%' | Out-Null; exit 0 } catch { exit 1 }")
if not errorlevel 1 goto pret
if %ELAPSED% GEQ %AIDGE_ATTENTE_MAX% goto trop_long
rem Pause de 2 s (ping est utilisable meme sans console interactive)
ping -n 3 127.0.0.1 >nul
set /a ELAPSED+=2
goto sonde

:pret
echo  Interface prete, ouverture du navigateur.
start "" "%URL%"
echo.
echo  Vous pouvez fermer cette fenetre-ci.
timeout /t 5 >nul
endlocal
exit /b 0

:trop_long
echo.
echo  ATTENTION : l'interface ne repond toujours pas apres %AIDGE_ATTENTE_MAX% secondes.
echo  Regardez la fenetre "AIDGE - interface" pour voir l'erreur affichee.
echo  Quand elle indique que l'interface tourne, ouvrez a la main l'adresse :
echo    %URL%
echo.
pause
endlocal
exit /b 2

:novenv
echo.
echo  ERREUR : l'environnement Python (.venv) n'existe pas encore.
echo.
echo  Il faut d'abord l'installer, une seule fois :
echo    1. Ouvrez le dossier "scripts" du projet
echo    2. Clic droit sur setup_venv.ps1, puis "Executer avec PowerShell"
echo    3. Attendez la fin, puis relancez ce fichier.
echo.
pause
endlocal
exit /b 1
