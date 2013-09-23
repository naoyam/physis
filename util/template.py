#!/usr/bin/env python

import sys
import itertools
import os
import re
def debug(s):
    sys.stderr.write("[DEBUG] %s\n" % s)

def error(s):
    sys.stderr.write("[ERROR] %s\n" % s)

TEMPLATE_PATH = ["."]    

class Macro:
    def __init__(self, name, params, macro=""):
        self.name = name
        self.params = params
        self.macro = macro

    def __str__(self):
        return "%s(%s): %s" % (self.name, ", ".join(self.params), self.macro)


def parse_macro_define_line(line):
    line = line.strip()
    name, params = re.match("#define (\w+)\((.*)\)", line).groups()
    params = [s.strip() for s in params.split(",")]
    return Macro(name, params)

def find_template_file(relative_path):
    for p in TEMPLATE_PATH:
        actual_path = os.path.join(p, relative_path)
        if os.path.exists(actual_path):
            f = file(actual_path)
            return f
    else:
        error("No such template file found: " + relative_path)
        sys.exit(1)
            
    
def read_template_file(path, macros):
    f = find_template_file(path)
    macro=None
    for line in f:
        if line.startswith("#define "):
            macro = parse_macro_define_line(line)
        elif line.strip() == "#end":
            macros[macro.name] = macro
            debug("Defining " + macro.name)
            macro = None
        elif macro is not None:
            macro.macro += line
        else:
            if line.strip() != "":
                debug("Ignoring " + line)

def find_macro(name, macros):
    m = macros.get(name)
    return m

def expand_macro(m, args):
    c = m.macro
    for p, a in zip(m.params, args):
        c = re.sub("\${" + p + "\}", a, c)
    return c
    
def process_command(line, output_file, macros):
    tokens = [t.strip() for t in line.split()]
    if tokens[0] == "#use":
        read_template_file(tokens[1], macros)
        return
    macro_name = line[1:line.find("(")]
    m = find_macro(macro_name, macros)
    if not m:
        output_file.write(line)
        return
    line = line.strip()
    args = [s.strip() for s in line[line.find("(")+1:-1].split(",")]
    output_file.write(expand_macro(m, args))
    return
        
def process_source_file(source_file, output_file, macros):
    for line in source_file:
        if line.startswith("#"):
            process_command(line, output_file, macros)
        else:
            output_file.write(line)
    
def main():
    source_file = sys.stdin
    output_file = sys.stdout

    idx = 1
    while idx < len(sys.argv):
        arg = sys.argv[idx]
        if arg.startswith("-I"):
            TEMPLATE_PATH.append(arg[2:])
            idx += 1            
        elif arg == "-o":
            output_file = file(sys.argv[idx+1], 'w')
            idx += 2
        else:
            source_file = file(arg, 'r')
            idx += 1

    macros = {}
    process_source_file(source_file, output_file, macros)
    return
    
if __name__ == "__main__":
    main()

