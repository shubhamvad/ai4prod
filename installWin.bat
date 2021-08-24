
IF "%1"=="--cuda" (
        SET cuda=%2
    )

@ECHO OFF
ECHO Congratulations! Your first batch file was executed successfully.
ECHO %cuda%

IF "%cuda%"=="11.0" (
    ECHO Select cuda 11.0
    set fileid="1YwUiSmpPTWeAUpWbvubqy57VvAjwM18A"
    )ELSE ( ECHO Cuda version not found)


IF "%1"=="--cpu" (
       ECHO Select CPU
	   set fileid="1m_XtGiWcL0M97uapiEOiLMYauue5DfdG"
    )ELSE ( ECHO Cpu version not found)
	
set cookieFile=cookie.txt
set confirmFile=confirm.txt

set filename="deps.7z"

REM downlaod cooky and message with request for confirmation
wget --quiet --save-cookies "%cookieFile%" --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=%fileid%" -O "%confirmFile%"


REM extract confirmation key from message saved in confirm file and keep in variable resVar
call jrepl ".*confirm=([0-9A-Za-z_]+).*" "$1" /F "confirm.txt" /A /rtn resVar

ECHO ResVar
ECHO %resVar%

REM when jrepl writes to variable, it adds carriage return (CR) (0x0D) and a line feed (LF) (0x0A), so remove these two last characters
SET confirmKey=%resVar%:~0,-2%

ECHO Confirm Key
ECHO %confirmKey%

REM download the file using cookie and confirmation key
wget --load-cookies "%cookieFile%" -O "%filename%" "https://docs.google.com/uc?export=download&id=%fileid%&confirm=%confirmKey%"

REM clear temporary files 
del %cookieFile%
del %confirmFile%

REM Unzip deps. Could take a while

mkdir deps
cd deps
set PATH=%PATH%;C:\Program Files\7-Zip\
echo %PATH%
7z x ..\deps.7z

IF "%cuda%"=="11.0" (
	move tensorrt C:\
)
cd ..

REM Install vcpkg
git clone https://github.com/microsoft/vcpkg
cd vcpkg
git checkout 00d190a039c78cad4aadfc9a9f3ce8b53165ae6b

call bootstrap-vcpkg.bat



