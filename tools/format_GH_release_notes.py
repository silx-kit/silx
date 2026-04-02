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

pr_descriptions = sorted(
    GH_PR_REGEXP.sub(r"\1 (PR #\2)", line)
    for line in content
    if GH_PR_REGEXP.match(line)
)

current_topic = None
current_descriptions = []

for index in range(len(pr_descriptions)):
    topic_and_description = tuple(
        text.strip() for text in pr_descriptions[index].split(":", 1)
    )
    if len(topic_and_description) == 2:
        new_topic, new_description = topic_and_description
    else:
        new_topic = pr_descriptions[index]
        new_description = ""

    if new_topic == current_topic:
        current_descriptions.append(new_description)
    else:
        if current_topic is not None:
            if len(current_descriptions) == 1:
                if not current_descriptions[0]:
                    print(f"* {current_topic}")
                else:
                    print(f"* {current_topic}: {current_descriptions[0]}")
            else:
                print(f"* {current_topic}:")
                print("")
                for description in current_descriptions:
                    print(f"  * {description}")
                print("")
        current_topic = new_topic
        current_descriptions = [new_description]
