#!/usr/bin/env python3
# coding:utf-8

import os
import sys
import subprocess

import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self):
        self.rowcol = [[3*2**8*i,3*2**8*i] for i in range(2,16,2)][:2]
        self.filename = "2d_ising.cu"
        
    def clean(self):
        path = "../build/CMakeCache.txt"
        if os.path.isfile(path):
            cmdline = "rm " + path
            subprocess.call(cmdline,shell=True)

    def build(self):
        for i,rowcol in enumerate(self.rowcol):
            print("size = [row,col] = [%d,%d]" %(rowcol[0],rowcol[1]))
            cmdline = 'sed "s/#define ROW .*/#define ROW %d/" -i ../src/%s' % (rowcol[0],self.filename)
            cmdline2 = 'sed "s/#define COL .*/#define COL %d/" -i ../src/%s' % (rowcol[1],self.filename)
            cmdline3 = 'sed "s/cuda_add_executable.*/cuda_add_executable\(run%d.exe 2d_ising.cu\)/" -i ../src/CMakeLists.txt' % (i)
            build = "(cd ../build;cmake ..;make )"
            subprocess.call(cmdline,shell=True)
            subprocess.call(cmdline2,shell=True)
            subprocess.call(cmdline3,shell=True)
            subprocess.call(build,shell=True)
            
    def run(self):
        f = open("bench.dat",'w')
        #for i in range(len(self.rowcol)):
        for i in range(2):
            cmdline = "(cd ../bin ;nvprof --log-file ../benchmarks/bench%d.dat ./run%d.exe )" % (i,i)
            #proc = subprocess.Popen(cmdline,shell=True,stdout=subprocess.PIPE)
            #line = proc.communicate()[0]
            out = subprocess.check_output(cmdline,shell=True)

            
    def show(self):
        pass
    
    def main(self):
        self.clean()
        #self.build()
        self.run()
        self.show()
        
if __name__ == "__main__":
    tmp = Benchmark()
    tmp.main()
