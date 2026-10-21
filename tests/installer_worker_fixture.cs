// Minimal executable used only by isolated Windows installer regression tests.
using System;
using System.Diagnostics;
using System.Threading;

public class InstallerWorkerFixture
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--orphan")
        {
            Process.Start(new ProcessStartInfo {
                FileName = Process.GetCurrentProcess().MainModule.FileName,
                Arguments = "--worker",
                UseShellExecute = false,
                CreateNoWindow = true
            });
            return;
        }
        Thread.Sleep(Timeout.Infinite);
    }
}
