@echo off
REM Windows batch script to run LED parameter tuning

REM Activate conda environment
call conda activate ps_llm

REM Set project paths
set PROJECT_DIR=%~dp0..
set DATA_PATH=%PROJECT_DIR%\dataset\mimic-iv-note-ext-di-bhc\dataset
set MODEL_NAME=led-large-16384
set RUN_DIR=mimic-iv-note-di-bhc_led-large-16384_4000_600_chars_100_valid

REM Create models directory if not exists
if not exist "%PROJECT_DIR%\models\%MODEL_NAME%\%RUN_DIR%" mkdir "%PROJECT_DIR%\models\%MODEL_NAME%\%RUN_DIR%"

REM Set device
set DEVICE=cuda

REM Experiment parameters
set MAX_STEPS=200000
set SAVE_LOGGING_STEPS=20000
set BATCH_SIZE=1

REM Loop through parameters
for %%d in (0.05 0.1 0.2) do (
    for %%l in (5e-4 1e-5 5e-5 1e-6 5e-6) do (
        set DROPOUT=%%d
        set LEARNING_RATE=%%l
        set FOLDER_NAME=dropout_%%d_learning_rate_%%l
        set EXPERIMENT_PATH=%PROJECT_DIR%\models\%MODEL_NAME%\%RUN_DIR%\!FOLDER_NAME!
        
        if not exist "!EXPERIMENT_PATH!" (
            echo Starting experiment: !EXPERIMENT_PATH!
            mkdir "!EXPERIMENT_PATH!"
            
            cd /d "%PROJECT_DIR%"
            
            python summarization\run_summarization_large_long.py ^
                --model_name_or_path allenai/led-large-16384 ^
                --do_train --do_eval --do_predict ^
                --train_file "%DATA_PATH%\train.json" ^
                --validation_file "%DATA_PATH%\valid_last_100.json" ^
                --test_file "%DATA_PATH%\valid_last_100.json" ^
                --output_dir "!EXPERIMENT_PATH!" ^
                --max_steps %MAX_STEPS% ^
                --evaluation_strategy steps ^
                --eval_steps %SAVE_LOGGING_STEPS% ^
                --save_steps %SAVE_LOGGING_STEPS% ^
                --load_best_model_at_end ^
                --per_device_train_batch_size=%BATCH_SIZE% ^
                --per_device_eval_batch_size=%BATCH_SIZE% ^
                --dropout !DROPOUT! ^
                --learning_rate !LEARNING_RATE! ^
                --predict_with_generate ^
                --max_source_length 4096 ^
                --max_target_length 350
        ) else (
            echo X Experiment already exists: !EXPERIMENT_PATH!
        )
    )
)

echo All experiments completed!
pause
