Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Always start from the project directory, even when Explorer supplies a
' different current working directory.
ProjectDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = ProjectDir

' Keep Unicode startup/log output from crashing under Windows.
WshShell.Environment("Process")("PYTHONIOENCODING") = "utf-8"

WshShell.Run "pythonw.exe """ & FSO.BuildPath(ProjectDir, "webapp\app.py") & """", 0, False
