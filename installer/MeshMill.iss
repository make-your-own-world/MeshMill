#ifndef MyAppVersion
  #error MyAppVersion must be supplied by the release build
#endif

#ifndef MyAppName
  #define MyAppName "MeshMill"
#endif
#ifndef MyAppId
  #define MyAppId "{{C8E36B57-B05D-47EE-95B2-45B96E01F7D5}"
#endif
#ifndef MySettingsKey
  #define MySettingsKey "Software\MeshMill"
#endif
#ifndef MyOutputBaseFilename
  #define MyOutputBaseFilename "MeshMill-" + MyAppVersion + "-windows-x64-setup"
#endif
#define MyAppPublisher "MeshMill contributors"
#define MyAppExeName "MeshMill.exe"
#ifndef MySourceDir
  #define MySourceDir "..\dist\MeshMill"
#endif
#define MyAssociationProgId MyAppName + ".STL"
#define MyAssociationExeName MyAppName + ".exe"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/make-your-own-world/MeshMill
AppSupportURL=https://github.com/make-your-own-world/MeshMill/issues
AppUpdatesURL=https://github.com/make-your-own-world/MeshMill/releases/latest
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\release
OutputBaseFilename={#MyOutputBaseFilename}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\meshmill.ico
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=force
RestartApplications=no
SetupLogging=yes
UsePreviousAppDir=yes
UsePreviousGroup=yes
UsePreviousTasks=yes
ChangesAssociations=yes
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=MeshMill installer
VersionInfoProductName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

; Remove the previous PyInstaller payload before copying an upgrade. The Inno
; Setup uninstaller and its data remain intact, while obsolete bundled modules,
; documentation, and locale files cannot survive from an older release.
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\assets"
Type: filesandordirs; Name: "{app}\docs"
Type: filesandordirs; Name: "{app}\locales"
Type: files; Name: "{app}\MeshMill.exe"
Type: files; Name: "{app}\MeshMillCLI.exe"
Type: files; Name: "{app}\LICENSE"
Type: files; Name: "{app}\README.md"
Type: files; Name: "{app}\ROADMAP.md"
Type: files; Name: "{app}\THIRD_PARTY_NOTICES.md"
Type: files; Name: "{app}\CUDA_EXCEPTION.md"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

; QSettings stores MeshMill preferences under this per-user key. An in-place
; upgrade keeps it. A real uninstall removes it with the rest of the app.
[Registry]
Root: HKCU; Subkey: "{#MySettingsKey}"; Flags: uninsdeletekey
; Advertise an optional handler and secondary verb, leaving the default STL handler alone.
Root: HKCU; Subkey: "Software\Classes\{#MyAssociationProgId}"; ValueType: string; ValueData: "STL mesh"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\{#MyAssociationProgId}\DefaultIcon"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"",0"
Root: HKCU; Subkey: "Software\Classes\{#MyAssociationProgId}\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\.stl\OpenWithProgids"; ValueType: string; ValueName: "{#MyAssociationProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAssociationExeName}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAssociationExeName}\SupportedTypes"; ValueType: string; ValueName: ".stl"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAssociationExeName}\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.stl\shell\{#MyAppName}"; ValueType: string; ValueName: "MUIVerb"; ValueData: "Open with {#MyAppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.stl\shell\{#MyAppName}"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#MyAppExeName}"",0"
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.stl\shell\{#MyAppName}\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
Filename: "{app}\{#MyAppExeName}"; Parameters: "{code:GetReopenParameters}"; Flags: nowait; Check: ShouldReopen

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\{#MyAppName}"

[Code]
function OpenOwnedProcess(Access: LongWord; Inherit: Boolean; PID: LongWord): THandle;
  external 'OpenProcess@kernel32.dll stdcall';
function QueryOwnedProcessImage(Process: THandle; Flags: LongWord;
  FileName: String; var Size: LongWord): Boolean;
  external 'QueryFullProcessImageNameW@kernel32.dll stdcall';
function StopOwnedProcess(Process: THandle; ExitCode: LongWord): Boolean;
  external 'TerminateProcess@kernel32.dll stdcall';
function WaitOwnedProcess(Process: THandle; Milliseconds: LongWord): LongWord;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function CloseOwnedHandle(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';

function IsInstalledExecutable(const FileName: String): Boolean;
begin
  Result := (CompareText(FileName, ExpandConstant('{app}\MeshMill.exe')) = 0) or
    (CompareText(FileName, ExpandConstant('{app}\MeshMillCLI.exe')) = 0);
end;

function StopInstalledProcesses(): String;
var
  Locator, Service, Processes, Process: Variant;
  I, Pass: Integer;
  Handle: THandle;
  Image: String;
  Size: LongWord;
  Found, Confirmed: Boolean;
begin
  Result := '';
  Confirmed := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    { Repeat the snapshot to catch workers created while their owner was closing. }
    for Pass := 1 to 4 do begin
      Found := False;
      Processes := Service.ExecQuery(
        'SELECT ProcessId, ExecutablePath FROM Win32_Process WHERE ' +
        'Name = ''MeshMill.exe'' OR Name = ''MeshMillCLI.exe''');
      for I := 0 to Processes.Count - 1 do begin
        Process := Processes.ItemIndex(I);
        if not VarIsNull(Process.ExecutablePath) then
          if IsInstalledExecutable(Process.ExecutablePath) then begin
            Handle := OpenOwnedProcess($00101001, False, Process.ProcessId);
            if Handle <> 0 then begin
              try
                { Recheck the executable through the process handle, not a reusable PID. }
                Size := 32768;
                SetLength(Image, Size);
                if QueryOwnedProcessImage(Handle, 0, Image, Size) then begin
                  SetLength(Image, Size);
                  if IsInstalledExecutable(Image) then begin
                    Found := True;
                    if not Confirmed then begin
                      if SuppressibleMsgBox(
                        'MeshMill is running. Save any work before continuing. ' +
                        'Continuing will close MeshMill and its processing workers ' +
                        'from this installation.', mbConfirmation, MB_OKCANCEL, IDOK) <> IDOK then begin
                        Result := 'Close MeshMill after saving your work, then try again.';
                        Exit;
                      end;
                      Confirmed := True;
                    end;
                    if WaitOwnedProcess(Handle, 0) <> 0 then begin
                      StopOwnedProcess(Handle, 1);
                      if WaitOwnedProcess(Handle, 5000) <> 0 then begin
                        Result := 'MeshMill could not be closed. Close it and try again.';
                        Exit;
                      end;
                    end;
                  end;
                end else if WaitOwnedProcess(Handle, 0) <> 0 then begin
                  Result := 'MeshMill process cleanup could not be verified. Close MeshMill and try again.';
                  Exit;
                end;
              finally
                CloseOwnedHandle(Handle);
              end;
            end else begin
              { A process may have exited between the snapshot and OpenProcess. }
              if DLLGetLastError = 5 then begin
                Result := 'Access to a running MeshMill process was denied. Close it and try again.';
                Exit;
              end;
            end;
          end;
      end;
      if not Found then Exit;
    end;
    Result := 'MeshMill is still running. Close it and try again.';
  except
    Result := 'MeshMill process cleanup could not be verified. Close MeshMill and try again.';
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := StopInstalledProcesses();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Error: String;
begin
  if CurUninstallStep = usUninstall then begin
    Error := StopInstalledProcesses();
    if Error <> '' then RaiseException(Error);
  end;
end;

function GetReopenParameters(Param: String): String;
var
  ReopenFile: String;
begin
  ReopenFile := ExpandConstant('{param:REOPENFILE|}');
  if ReopenFile <> '' then
    Result := '"' + ReopenFile + '"'
  else
    Result := '';
end;

function ShouldReopen(): Boolean;
begin
  Result := ExpandConstant('{param:REOPEN|0}') = '1';
end;
