import os
from pathlib import Path

here = Path(os.path.join(os.path.dirname(os.path.abspath(__file__))))
metadata_file = os.path.join(here, 'pandoc.yaml')

for f in here.glob('./**/index.md'):
    directory = os.path.dirname(f)
    src_file = os.path.join(directory, 'index.md')
    dst_file = os.path.join(directory, 'index.html')

    if not os.path.exists(dst_file) or os.path.getmtime(dst_file) < max(os.path.getmtime(src_file), os.path.getmtime(metadata_file), os.path.getmtime(__file__)):
        print(f'Building {directory}...')
        os.system(f'pandoc --fail-if-warnings --metadata-file={metadata_file} --listings --katex --toc --toc-depth=2 -s {src_file} -o {dst_file}')
    else:
        print(f'Skipping {directory}...')
