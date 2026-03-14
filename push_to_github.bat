@echo off
echo ============================================================
echo Pushing Dog Breed Classifier to GitHub
echo ============================================================
echo.

echo Repository: https://github.com/AashleshaKhanna/DogBreedClassifierWithMixedDetection
echo.

echo Step 1: Checking git status...
git status
echo.

echo Step 2: Adding all files (respecting .gitignore)...
git add .
echo.

echo Step 3: Showing what will be committed...
git status
echo.

set /p CONTINUE="Continue with commit? (y/n): "
if /i not "%CONTINUE%"=="y" (
    echo Aborted.
    pause
    exit /b 0
)

echo.
echo Step 4: Creating commit...
git commit -m "Update: Dog Breed Classifier project with complete infrastructure"
echo.

echo Step 5: Pushing to GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push failed. Trying with master branch...
    git push -u origin master
)

echo.
echo ============================================================
echo Done! Check your repository:
echo https://github.com/AashleshaKhanna/DogBreedClassifierWithMixedDetection
echo ============================================================
echo.
pause
