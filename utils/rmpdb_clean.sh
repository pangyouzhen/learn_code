#!/bin/bash

cd /var/lib/rpm
rm __db.* -rf
rpm --rebuilddb
yum clean all