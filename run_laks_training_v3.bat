@echo off
echo ================================================================================
echo Launching SatQuery Division 4 Balanced V3 Training on RTX 4060 GPU
echo (Median-Frequency Balanced Loss, Meaningful Minority Sampler, Isolated Checkpoint)
echo ================================================================================
".venv\Scripts\python.exe" laks_run_balanced_v3.py --data-dir "D:/official_whu_opt_sar_dataset/official_whu_opt_sar" %*
pause
