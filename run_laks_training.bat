@echo off
echo ================================================================================
echo Launching SatQuery Division 4 Training on RTX 4060 GPU
echo ================================================================================
".venv\Scripts\python.exe" laks_you_should_run_this_only.py --data-dir "D:/official_whu_opt_sar_dataset/official_whu_opt_sar_100scenes" %*
pause
