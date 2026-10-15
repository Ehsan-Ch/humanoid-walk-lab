"""Package the public source, frozen policy, and published experiment artifacts."""
from pathlib import Path
import os,zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'.gitignore','.gitattributes','pyproject.toml','requirements-lock.txt','README.md','LICENSE','ATTRIBUTION.md','VALIDATION.md'}
DIRS={'.github','walk_lab','tests','scripts','artifacts','reports'}

def public_files():
    files=[]
    for folder,dirs,names in os.walk(ROOT):
        if Path(folder)==ROOT:dirs[:]=[d for d in dirs if d in DIRS]
        else:dirs[:]=[d for d in dirs if d not in ('__pycache__','custom')]
        for name in names:
            p=Path(folder)/name;rel=p.relative_to(ROOT)
            if name=='probe.py':continue
            if (len(rel.parts)==1 and name in ROOT_FILES) or len(rel.parts)>1:files.append(p)
    return sorted(files)

def package():
    target=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in public_files():archive.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise ValueError('Archive integrity check failed.')
    print(target)

if __name__=='__main__':package()
