"""Installed C author generator: python -m polygrad.sdk --help."""

from pathlib import Path
import runpy


def main():
    generator = Path(__file__).with_name('_sdk') / 'extension_bindings.py'
    if not generator.is_file():
        raise RuntimeError('author SDK missing; in a checkout run make extension-sdk')
    runpy.run_path(str(generator), run_name='__main__')


if __name__ == '__main__':
    main()
