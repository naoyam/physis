#!/usr/bin/env python

import sys
import itertools
import os
import re
import traceback
from optparse import OptionParser

header_path = "source_header.txt"

def debug(s):
    sys.stderr.write("[DEBUG] %s\n" % s)

def error(s):
    sys.stderr.write("[ERROR] %s\n" % s)

def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    return

def add_or_replace(path):
    # query if this path should be handled
    print "Checking %s?" % path
    f = open(path, "r")
    cur_header = ""
    has_empty_line = False
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith("//"):
            cur_header += line
            continue
        has_empty_line = line.strip() == ""
        rem = line + f.read()
        f.close()        
        break
    header_file = open(header_path, "r")
    new_header = header_file.read()
    if cur_header == new_header:
        print "Header not changed; nothing done"
        return
    else:
        sys.stdout.write("Current header: " + cur_header)
    sys.stdout.write("Modifying %s? (y/n): " % path)
    response = sys.stdin.readline()
    response = response.lower().strip()
    if not (response == "y" or response == "yes"):
        sys.stderr.write("Ignoring %s\n" % path)
        return
    f = open(path, "w")
    f.write(new_header)
    if not has_empty_line:
        f.write("\n") 
    f.write(rem)
    f.close()
    return

def apply_recursively(top):
    for dirName, subdirList, fileList in os.walk(top):
        print dirName
        if ".git" in dirName or "examples" in dirName \
                or "gmock" in dirName:
            print "Ignoring"
            continue
        for f in fileList:
            if not (f.endswith("cc") or f.endswith(".h")):
                continue
            add_or_replace(os.path.join(dirName, f))
    return

def main():
    top_path = sys.argv[1]
    apply_recursively(top_path)
    
if __name__ == "__main__":
    main()
