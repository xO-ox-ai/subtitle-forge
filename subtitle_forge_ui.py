from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _dispatch_pipeline_script() -> bool:
    if len(sys.argv) < 2:
        return False
    candidate = Path(sys.argv[1])
    if candidate.suffix.lower() != ".py" or not candidate.is_file():
        return False
    sys.argv = [str(candidate)] + sys.argv[2:]
    script_dir = str(candidate.resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    runpy.run_path(str(candidate), run_name="__main__")
    return True


def main() -> None:
    if _dispatch_pipeline_script():
        return
    from forge_ui.app import run_app

    run_app(smoke_test="--ui-smoke-test" in sys.argv)


if __name__ == "__main__":
    main()
