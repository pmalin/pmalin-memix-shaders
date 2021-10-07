:: drag a memix shader onto me to install

@echo off

copy %LOCALAPPDATA%\Beautypi\Memix\shader.txt %LOCALAPPDATA%\Beautypi\Memix\shader.txt.old_%RANDOM%
copy %1 %LOCALAPPDATA%\Beautypi\Memix\shader.txt

:: touch file 
copy /B %LOCALAPPDATA%\Beautypi\Memix\shader.txt+,, %LOCALAPPDATA%\Beautypi\Memix\shader.txt

:: pause