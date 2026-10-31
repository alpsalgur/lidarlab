@echo off
rem Start LidarLab without a console window. Optional argument: a test run folder.
start "" "%~dp0venv\Scripts\lidarlab.exe" %*
