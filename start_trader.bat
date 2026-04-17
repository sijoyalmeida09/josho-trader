@echo off
REM JoSho Trader — Auto-start script for Windows
REM Add to: shell:startup (Win+R -> shell:startup -> paste shortcut)
REM Or use Task Scheduler to run at login

echo Starting JoSho Trader services...
cd /d C:\Sijoy_2.0
pm2 resurrect
echo Done. Check status with: pm2 list
