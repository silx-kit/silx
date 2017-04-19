@echo off
@rem -I: isolate Python from the user's environment (implies -E and -s)
@rem It avoids to be overidded by a path like ./silx
python -I -m silx %*
@echo on
