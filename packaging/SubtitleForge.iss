#ifndef MyAppVersion
  #define MyAppVersion "1.0.1"
#endif

[Setup]
AppId={{E45DF754-34D9-48D1-88DB-5F624D262897}
AppName=Subtitle Forge
AppVersion={#MyAppVersion}
AppPublisher=xO-ox-ai
AppPublisherURL=https://github.com/xO-ox-ai/subtitle-forge
DefaultDirName={localappdata}\SubtitleForge
DefaultGroupName=Subtitle Forge
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\installer-output
OutputBaseFilename=SubtitleForge-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\SubtitleForge.exe
SetupLogging=yes

[Files]
Source: "..\dist\SubtitleForge.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\portable-assets\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Subtitle Forge"; Filename: "{app}\SubtitleForge.exe"
Name: "{autodesktop}\Subtitle Forge"; Filename: "{app}\SubtitleForge.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："

[Run]
Filename: "{app}\SubtitleForge.exe"; Description: "启动 Subtitle Forge"; Flags: nowait postinstall skipifsilent
