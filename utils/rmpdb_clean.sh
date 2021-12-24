#!/bin/bash

cd /var/lib/rpm || exit
rm __db.* -rf
rpm --rebuilddb
yum clean all