""" Fix the format of GitHub-generated release notes. """

import os.path

if __name__ == "__main__":

    import re

    root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    with open(os.path.join(root_dir, 'CHANGELOG.rst'), 'r') as f:
        content = f.read().split('/n')


    with open(os.path.join(root_dir,'CHANGELOG_new.rst'), 'w') as f:
        for line in content:
            new_line = re.sub(r'by @.+ in https://github\.com/silx-kit/silx/pull/(\d+)', r'(PR #\1)', line)
            f.write(new_line)
            f.write('\n')
