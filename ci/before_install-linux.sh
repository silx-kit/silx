#!/bin/bash
echo $(pwd)
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
