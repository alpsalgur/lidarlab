; Inno Setup script for LidarLab: installs dist\LidarLab (PyInstaller output) for the current
; user without administrator rights (all users can be chosen in the setup), adds a Start menu
; entry, optionally a desktop shortcut, and opens .lidarlab workspaces with LidarLab.
; Uninstalling also removes the settings, logs and caches; Documents\LidarLab stays.
; Built by tools\build_release.py:  ISCC /DAppVersion=1.0.0 packaging\installer.iss

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{876DD298-F7A1-4C0B-B479-B3EE3CCEDF49}
AppName=LidarLab
AppVersion={#AppVersion}
AppVerName=LidarLab {#AppVersion}
DefaultDirName={autopf}\LidarLab
DefaultGroupName=LidarLab
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=LidarLab-{#AppVersion}-Setup
SetupIconFile=..\build\lidarlab.ico
UninstallDisplayIcon={app}\LidarLab.exe
UninstallDisplayName=LidarLab {#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
; the compiler is a 32-bit process: compress in Inno Setup's 64-bit helper, or ~570 MB with a
; 64 MB dictionary runs out of memory
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=2
WizardStyle=modern
ChangesAssociations=yes
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\LidarLab\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LidarLab"; Filename: "{app}\LidarLab.exe"
Name: "{group}\Uninstall LidarLab"; Filename: "{uninstallexe}"
Name: "{autodesktop}\LidarLab"; Filename: "{app}\LidarLab.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.lidarlab"; ValueType: string; ValueName: ""; ValueData: "LidarLab.Workspace"; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKA; Subkey: "Software\Classes\LidarLab.Workspace"; ValueType: string; ValueName: ""; ValueData: "LidarLab workspace"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\LidarLab.Workspace\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\LidarLab.exe,0"
Root: HKA; Subkey: "Software\Classes\LidarLab.Workspace\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\LidarLab.exe"" ""%1"""
; the settings LidarLab writes itself (window layout, recent test runs): not created here
Root: HKCU; Subkey: "Software\LidarLab"; Flags: dontcreatekey uninsdeletekey

[UninstallDelete]
; logs and Qt's caches; bookmarks kept here for read-only test runs stay
Type: files; Name: "{localappdata}\LidarLab\*.log"
Type: filesandordirs; Name: "{localappdata}\LidarLab\LidarLab*"
Type: dirifempty; Name: "{localappdata}\LidarLab"

[Run]
Filename: "{app}\LidarLab.exe"; Description: "Start LidarLab"; Flags: nowait postinstall skipifsilent
