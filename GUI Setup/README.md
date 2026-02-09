AutoSWD - Universal Package
===========================

⚠️ Sample EDF and Attention-ResUNet model could not be uploaded on GitHub, but could be downloaded [here](https://drive.google.com/drive/folders/1XRuBrwPgplfdAam1x4VvuIo5aQBUqiul?usp=drive_link).

Instructions for Windows:
-------------------------
1. Double-click "Run_Windows.bat".
2. If Python is missing, it will ask to download and install it for you.
   (IMPORTANT: Check "Add to PATH" during installation).
3. The first run will set up necessary libraries.

Instructions for Mac / Linux:
-----------------------------
1. Open Terminal.
2. Navigate to this folder.
   (Tip: Type "cd " and drag this folder into the terminal window).
3. Run the following command to allow execution (only needed once):
   chmod +x Run_Mac_Linux.sh
4. Run the launcher:
   ./Run_Mac_Linux.sh

   *Note: This will launch the optimized 'AutoSWD_Mac.py' script.*

Requirements:
-------------
- Internet connection required for first run (to download libraries).

Warnings (Windows):
-------------------
⚠️ Python 3.11 must be installed and added to PATH
   - The script will attempt to download Python 3.11.9 if not found
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
