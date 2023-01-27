#!/usr/bin/env python3.8
import os, sys, re

with open('/Users/sudo/CodeProjects/Tufts/CS0135/ml135_env_su22.yml', 'r') as fi:
    with open('/Users/sudo/CodeProjects/Tufts/CS0135/new_ml135_env_su22.yml', 'w') as fo:
        count_match = 0 # = -> ==
        for line in fi: # add an extra equal sign to first equal sign match
            result = re.match('(    - [a-zA-Z0-9._-]+)(=)(.*)', line)
            if result:
                line = f'{result.group(1)}=={result.group(3)}\n'
                # fo.write(line)
                count_match += 1
            # else:
            fo.write(line)

        print(f'match line count: {count_match}\n')