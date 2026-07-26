@echo off
echo ==== py2 project backup: G:\py2 -^> E:\py2_backup ====
echo Incremental copy (only new/updated files; never deletes anything).
echo First run copies ~6.5 GB and may take a few minutes.
echo.
robocopy G:\py2 E:\py2_backup /E /XO /XJ /R:1 /W:1 /NFL /NDL
echo.
echo Backup finished. (robocopy exit codes 0-7 all mean success)
pause
