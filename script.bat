@echo off
setlocal enabledelayedexpansion

REM Check if the Solutions\cvrp directory exists, if not, create it
if not exist "Solutions\cvrp" (
    mkdir "Solutions\cvrp"
)

REM Read instances.txt file line by line
for /f "tokens=*" %%i in (instances.txt) do (
    REM Run main.py with the current line as an argument and redirect output to Solutions\cvrp\<line>.txt
    echo Running main.py with argument %%i
    python main.py %%i > Solutions\cvrp\%%i.txt
)

echo All instances processed.
