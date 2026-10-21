"""Build an observational adapter against the installed Fast-Quadric header.

The dependency itself is left untouched. Generated sources stay under build/.
"""
from pathlib import Path
import os
import fast_simplification
from setuptools import Extension, setup

root = Path(__file__).resolve().parents[1]
source = Path(fast_simplification.__file__).parent / "Simplify.h"
header = source.read_text(encoding="utf-8")
start = header.index("\tvoid simplify_mesh(")
end = header.index("\t} //simplify_mesh()", start)
part = header[start:end]
needle = "\t\t\t\tTriangle &t=triangles[i];"
if part.count(needle) != 1:
    raise RuntimeError("Unsupported Fast-Quadric core: progress hook location changed")
part = part.replace(needle, '''
                if ((i & 16383) == 0) {
                    meshmill_progress("Scanning QEM triangles", i, triangles.size());
                }
''' + needle)
part = part.replace("\t\t\t\tupdate_mesh(iteration);", '''
                meshmill_progress("Building QEM connectivity", 0, triangles.size());
                update_mesh(iteration);
                meshmill_progress("Building QEM connectivity", triangles.size(), triangles.size());
''')
part += '\n        meshmill_progress("Reducing mesh topology", deleted_triangles, triangle_count-target_count);\n'
header = header[:start] + part + header[end:]
start = header.index("\tvoid update_mesh(int iteration)\n\t{")
end = header.index("\t// Finally compact mesh", start)
part = header[start:end]
for needle, stage, count in [
    ("Triangle &t=triangles[i];", "Building QEM connectivity", "triangles.size()"),
    ("Vertex &v=vertices[i];", "Checking QEM boundaries", "vertices.size()"),
]:
    part = part.replace(needle, f'''if ((i & 16383) == 0)
                    meshmill_progress("{stage}", i, int({count}));
                {needle}''')
header = header[:start] + part + header[end:]
generated = root / "build" / "qem-progress"
generated.mkdir(parents=True, exist_ok=True)
generated_header = generated / "Simplify.h"
if not generated_header.exists() or generated_header.read_text(encoding="utf-8") != header:
    generated_header.write_text(header, encoding="utf-8")
os.chdir(generated)
setup(name="meshmill-qem-progress", script_args=["build_ext", "--build-lib", str(root)],
      ext_modules=[Extension("_meshmill_qem", [str(root / "native" / "qem_progress.cpp")],
                             depends=[str(generated / "Simplify.h")],
                             include_dirs=[str(generated)], language="c++")])
