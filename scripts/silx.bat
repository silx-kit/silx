@echo off
@rem -I: isolate Python from the user's environment (implies -E and -s)
@rem It avoid to be overided my a path like ./silx
python -I -m silx %*
@echo on
