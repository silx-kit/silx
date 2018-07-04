#!/bin/bash
echo $(pwd)

DOWNLOAD_APPSDK=1

if [ -n "$SILX_OPENCL" ]
then
	if [ $SILX_OPENCL = "0" ]
	then
		DOWNLOAD_APPSDK=0
	fi
	if [ $SILX_OPENCL = "False" ]
	then
		DOWNLOAD_APPSDK=0
	fi
fi

if [ $DOWNLOAD_APPSDK = 1 ]
then
	source ./ci/intel_opencl_icd.sh
	ls
else
	echo "OpenCL ICD download skipped"
fi
