"""Fix the format of GitHub-generated release notes."""

import argparse
import re
from pathlib import Path

GH_PR_REGEXP = re.compile(
    r"\* (.*) by @.+ in https://github\.com/silx-kit/silx/pull/(\d+)"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "file", nargs=1, metavar="FILE", type=str, help="github changelog file"
)
options = parser.parse_args()

content = Path(options.file[0]).read_text().split("\n")

pr_descriptions = tuple(
    GH_PR_REGEXP.sub(r"\1 (PR #\2)", line)
    for line in content
    if GH_PR_REGEXP.match(line)
)

changelog_entries = {}
for pr_description in pr_descriptions:
    topic_and_description = tuple(text.strip() for text in pr_description.split(":", 1))
    if len(topic_and_description) == 2:
        topic, description = topic_and_description
    else:
        topic = pr_description
        description = ""

    if topic not in changelog_entries:
        changelog_entries[topic] = []

    if description:
        changelog_entries[topic].append(description)

for topic, descriptions in sorted(changelog_entries.items()):
    if len(descriptions) == 0:
        print(f"* {topic}")
    elif len(descriptions) == 1:
        print(f"* {topic}: {descriptions[0]}")
    else:
        print(f"* {topic}:")
        print("")
        for description in descriptions:
            print(f"  * {description}")
        print("")
