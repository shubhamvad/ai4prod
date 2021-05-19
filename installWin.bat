
@REM IF "%1"=="--cuda" (
@REM         SET cuda=%2
@REM     )

@REM @ECHO OFF
@REM ECHO Congratulations! Your first batch file was executed successfully.
@REM ECHO %cuda%

@REM IF "%cuda%"=="10.2" (
@REM     ECHO Select cuda 10.2
@REM     set fileid="1h1ybEJWrysYJ8PKCdL9ELEfUNk7Nn1Qa"
@REM     )



@REM set cookieFile=cookie.txt
@REM set confirmFile=confirm.txt

@REM set filename="deps.7z"

@REM REM downlaod cooky and message with request for confirmation
@REM wget --quiet --save-cookies "%cookieFile%" --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=%fileid%" -O "%confirmFile%"


@REM REM extract confirmation key from message saved in confirm file and keep in variable resVar
@REM call jrepl ".*confirm=([0-9A-Za-z_]+).*" "$1" /F "confirm.txt" /A /rtn resVar

@REM ECHO %resVar%

@REM REM when jrepl writes to variable, it adds carriage return (CR) (0x0D) and a line feed (LF) (0x0A), so remove these two last characters
@REM SET confirmKey=%resVar%:~0,-2%

@REM ECHO %confirmKey%
@REM REM download the file using cookie and confirmation key
@REM wget --load-cookies "%cookieFile%" -O "%filename%" "https://docs.google.com/uc?export=download&id=%fileid%&confirm=%confirmKey%"

@REM REM clear temporary files 
@REM del %cookieFile%
@REM del %confirmFile%

@REM Unzip deps

set PATH=%PATH%;C:\Program Files\7-Zip\
echo %PATH%
7z deps.7z