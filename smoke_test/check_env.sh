#!/bin/sh

# From Intel Cooperation

echo "=====================OS Environment==============="
# check CPU
CPU_NAME=`cat /proc/cpuinfo |grep "model name"|awk -F ':' '{print $2}'|uniq`
echo "-----------------CPU NAME -------------------"
echo "$CPU_NAME"

# check DDR
RAM_SIZE=`free -h`
echo "----------------MEMORY SIZE -----------------"
echo "$RAM_SIZE"

# echo "-------------MEMORY CLOCK SPEED --------------"
# RAM_CLOCK=`dmidecode -t memory|grep -ie HZ`
# echo "$RAM_CLOCK"

# check HT
HT=`lscpu|grep "per core"|awk -F ':' '{print $2}'|sed 's/^ *\| *$//g'`
echo "-------------------  HT ---------------------"
if [[ ${HT} == 1 ]]; then
	echo "HT:OFF"
elif [[ ${HT} > 1 ]]; then
	echo "HT:ON"
else
	echo "HT:UNKHOWN"
fi

# check TURBO
TURBO=`cat /sys/devices/system/cpu/intel_pstate/no_turbo`
echo "-------------------TURBO---------------------"
if [[ ${TURBO} == 1 ]]; then
	echo "TURBO:OFF"
else
	echo "TURBO:On"
fi

# check CPU power governor
echo "-------------------POWER GOVERNOR---------------------"
POWER_GOVERNOR=`cpupower frequency-info | grep \"*\" | awk -F '\"' '{print $2}'`
echo $POWER_GOVERNOR

# check NUMA and MCDRAM configuration
# You should $yum install hwloc
echo "-------------------NUMA and MCDRAM mode---------------------"
CLUSTER_MODE=`hwloc-dump-hwdata |grep -i Cluster\ Mode: | awk -F ' ' '{print $3}'`
echo "Cluster Mode: "$CLUSTER_MODE
MEMORY_MODE=`hwloc-dump-hwdata |grep -i Cluster\ Mode: | awk -F ' ' '{print $(NF)}'` 
echo "Memory Mode: "$MEMORY_MODE

# check OS
echo "-------------------OS VERSION---------------------"
OS_VERSION=`uname -a`
echo $OS_VERSION

echo "===================SOFTWARE SETTINGS===================="

# check threads env
echo "-------------------THREADS ENV---------------------"
MKL_NUM_THREADS=`env |grep MKL_NUM_THREADS`
if [ "${MKL_NUM_THREADS}" = "" ]; then
	echo "MKL_NUM_THREADS:UNSET"
else
	echo "$MKL_NUM_THREADS"
fi

OMP_NUM_THREADS=`env |grep OMP_NUM_THREADS`
if [ "${OMP_NUM_THREADS}" = "" ]; then
	echo "OMP_NUM_THREADS:UNSET"
else
	echo "$OMP_NUM_THREADS"
fi

KMP_AFFINITY=`env |grep KMP_AFFINITY`
if [ "${KMP_AFFINITY}" = "" ]; then
	echo "KMP_AFFINITY:UNSET"
else
	echo "$KMP_AFFINITY"
fi

# check GCC version
echo "-------------------GCC VERSION---------------------"
GCC_VERSION=`gcc --version`
echo $GCC_VERSION


