AutoSWD - Setup
===========================

Instructions:
-------------------------
1. Double-click "Run_Windows.bat".
2. If Python is missing, it will ask to download and install it for you.
   (IMPORTANT: Check "Add to PATH" during installation).
3. The first run will set up necessary libraries.

We provide a sample EDF recording for testing, deliberately selected for its noisy characteristics to highlight the robustness of AutoSWD.

⚠️ Sample EDF and Attention-ResUNet model could not be uploaded on GitHub, but could be downloaded [here](https://drive.google.com/drive/folders/1XRuBrwPgplfdAam1x4VvuIo5aQBUqiul?usp=drive_link).


Warnings:
-------------------
⚠️ Python 3.11.5 must be installed and added to PATH
   - The script will attempt to download Python 3.11.5 if not found
   - Ensure "Add python.exe to PATH" is checked during installation
   - Verify installation: Open Command Prompt and type "python --version"

⚠️ Visual C++ Redistributables must be installed
   - Required for TensorFlow to function properly
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Without this, you may see DLL load errors when launching AutoSWD

⚠️ Internet connection required for pip installs
   - First-time setup downloads all required Python packages
   - Ensure firewall/antivirus allows Python and pip network access

⚠️ Standard user permissions required
   - Script needs permission to create virtual environment and install packages
   - If you encounter permission errors, run Command Prompt as Administrator

⚠️ Software is designed on Windows therefore the GUI may not work as intended on Mac or Linux.
