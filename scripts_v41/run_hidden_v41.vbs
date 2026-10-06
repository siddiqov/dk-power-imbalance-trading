' Nurex V4.1 (2026-10-05): run a .bat from this folder WITHOUT a console window.
' Used by the scheduled tasks:  wscript.exe run_hidden_v41.vbs run_update_v41.bat
' The .bat writes its own log (logs\...), so nothing is lost by hiding the window.
If WScript.Arguments.Count < 1 Then WScript.Quit 1
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
bat = fso.BuildPath(here, WScript.Arguments(0))
Set sh = CreateObject("WScript.Shell")
WScript.Quit sh.Run("cmd /c """ & bat & """", 0, True)
