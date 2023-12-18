#!/bin/bash
unset http_proxy && unset https_proxy
source /etc/network_turbo
echo "Setting up Git user..."
git add .

git commit -m "added"

git pull --rebase

git reset --hard origin/master
