@echo off
setlocal enabledelayedexpansion

for /l %%i in (1,1,21) do (
    echo Loop iteration %%i
    soln_gen.bat && evaluation.bat
)

endlocal
