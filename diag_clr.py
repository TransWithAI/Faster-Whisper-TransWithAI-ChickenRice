import ctypes, os, sys
base = os.path.dirname(sys.executable)
internal = os.path.join(base, "_internal")
p = os.path.join(internal, "clr_loader", "ffi", "dlls", "amd64", "ClrLoader.dll")
f = open(os.path.join(base, "diag_clr.txt"), "w", encoding="utf-8")
f.write(f"frozen={getattr(sys,'frozen',False)}\nmeipass={getattr(sys,'_MEIPASS',None)}\n")
f.write(f"exists={os.path.exists(p)}\npath={p}\n")
try:
    ctypes.WinDLL(p)
    f.write("LoadLibrary OK\n")
except OSError as e:
    f.write(f"FAIL: {e}\n")
import clr
f.write("import clr OK\n")
f.close()
