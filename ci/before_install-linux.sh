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
	bash ./ci/amd_sdk.sh;
	ls
	tar -xjf AMD-SDK.tar.bz2;
	export AMDAPPSDK=$(pwd)/AMDAPPSDK;
	export OPENCL_VENDOR_PATH=${AMDAPPSDK}/etc/OpenCL/vendors;
	mkdir -p ${OPENCL_VENDOR_PATH};
	sh AMD-APP-SDK*.sh --tar -xf -C ${AMDAPPSDK};
	echo libamdocl64.so > ${OPENCL_VENDOR_PATH}/amdocl64.icd;
	export LD_LIBRARY_PATH=${AMDAPPSDK}/lib/x86_64:${LD_LIBRARY_PATH};
	chmod +x ${AMDAPPSDK}/bin/x86_64/clinfo;
	${AMDAPPSDK}/bin/x86_64/clinfo;
else
	echo "AMDSDK download skipped"
fi
