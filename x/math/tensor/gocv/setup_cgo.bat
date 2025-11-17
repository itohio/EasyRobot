@echo off
REM Helper script to set CGO environment variables for building GoCV on Windows
REM Run this before building: .\setup_cgo.bat
REM Or source it: call setup_cgo.bat

set OPENCV_PREFIX=C:\opencv\build\install
set OPENCV_LIB_PATH=%OPENCV_PREFIX%\x64\mingw\lib
set OPENCV_INCLUDE_PATH=%OPENCV_PREFIX%\include

echo Setting CGO environment variables for GoCV...
set CGO_ENABLED=1
set CGO_CXXFLAGS=--std=c++11
set CGO_CPPFLAGS=-I%OPENCV_INCLUDE_PATH%
set CGO_LDFLAGS=-L%OPENCV_LIB_PATH% -lopencv_core4120 -lopencv_imgproc4120 -lopencv_imgcodecs4120 -lopencv_videoio4120 -lopencv_highgui4120 -lopencv_features2d4120 -lopencv_calib3d4120 -lopencv_objdetect4120 -lopencv_dnn4120 -lopencv_video4120

echo.
echo CGO environment variables set:
echo   CGO_ENABLED=%CGO_ENABLED%
echo   CGO_CXXFLAGS=%CGO_CXXFLAGS%
echo   CGO_CPPFLAGS=%CGO_CPPFLAGS%
echo   CGO_LDFLAGS=%CGO_LDFLAGS%
echo.
echo You can now build GoCV projects:
echo   go build ./drivers/lidar
echo.
echo Or use the customenv build tag:
echo   go build -tags customenv ./drivers/lidar

