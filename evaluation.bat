@echo off
setlocal

REM Directory containing the .txt files
set "dir=Solutions/cvrp"

REM Output file
for /f %%i in ('powershell -command "[int][double]::Parse((Get-Date -UFormat %%s))"') do set epoch=%%i
set output=Results\results_%epoch%.txt
REM set "output=results.txt"

REM Clear the output file
if exist %output% del %output%

REM Iterate over each .txt file in the specified directory
for %%f in (%dir%\*.txt) do (
    REM Extract the base name of the file (without extension)
    set "filename=%%~nf"
    
    REM Run the Python command and append the output to the results file
    python evaluator.py cvrp %%~nf %%f >> %output%
)

endlocal
