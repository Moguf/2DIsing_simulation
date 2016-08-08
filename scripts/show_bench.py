#!/usr/bin/env python3
# coding:utf-8

import os
import subprocess

import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self):
        self.rowcol = [[3*2**8*i,3*2**8*i] for i in range(2,16,2)]
        self.filename = "2d_ising.cu"
        
    def clean(self):
        path = "../build/CMakeCache.txt"
        if os.path.isfile(path):
            cmdline = "rm " + path
            subprocess.call(cmdline,shell=True)

    def build(self):
        for rowcol in self.rowcol:
            cmdline = 'sed "s/#define ROW .*/#define ROW %d/ -i ../src/%s' % (rowcol[0],self.filename)
            cmdline2 = 'sed "s/#define COL .*/#define COL %d/ -i ../src/%s' % (rowcol[1],self.filename)
            #subprocess.call(
            print(cmdline)
        
    def run(self):
        print(self.rowcol)


    def show(self):
        pass
    
    def main(self):
        self.clean()
        self.build()
        self.run()
        self.show()
        
if __name__ == "__main__":
    tmp = Benchmark()
    tmp.main()
