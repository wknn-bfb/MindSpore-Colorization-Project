@echo off
setlocal enabledelayedexpansion
set "DST=checkpoints"
if not exist "%DST%" mkdir "%DST%"

echo [1/4] colorization_release_v2-9b330a0b.pth
curl -L -o "%DST%/colorization_release_v2-9b330a0b.pth" ^
  https://drive.google.com/uc?id=1AKPjbb9so9H0P9VJ1QrV0U6jIRa_1gI0&export=download

echo [2/4] net_g_40.ckpt
curl -L -o "%DST%/net_g_40.ckpt" ^
  https://drive.google.com/uc?id=1upnFWtAOCmeQ4zl-gtNNCya5thMlG8H7&export=download

echo [3/4] vgg16.ckpt
curl -L -o "%DST%/vgg16.ckpt" ^
  https://drive.google.com/uc?id=11UtAKzEEFtxlu9rzClZU_d8LbhKrLYrD&export=download

echo [4/4] siggraph17-df00044c.pth
curl -L -o "%DST%/siggraph17-df00044c.pth" ^
  https://drive.google.com/uc?id=1JrVvDEjisdiHbwoTgOOUcJFgv8r18_nY&export=download

echo All checkpoints downloaded to %DST%\
pause