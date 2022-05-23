from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 10  #length of vectors

#arbitrary example vectors, generated to be evenly divided by the number of
#processes for convenience

x = numpy.linspace(0, 100, n) if comm.rank == 0 else None
y = numpy.linspace(20, 300, n) if comm.rank == 0 else None

print('x:', x)
print('y:', y)

#initialize as numpy arrays
result = numpy.array([0])
local_n = numpy.array([0])

#test for conformability
if rank == 0:
               if (n != y.size):
                               print("vector length mismatch")
                               comm.Abort()


               if(n % size != 0):
                                print("the number of processors   must evenly divide n.")
                                comm.Abort()

               #length of each process's portion of the original  vector
               local_n = numpy.array([n/size])

#communicate local array size to all processes
comm.Bcast(local_n, root=0)

#initialize as numpy arrays
local_x = numpy.zeros(int(local_n))
local_y = numpy.zeros(int(local_n))

#divide up vectors
comm.Scatter(x, local_x, root=0)
comm.Scatter(y, local_y, root=0)

#local computation of the product
local_result = numpy.array([numpy.multiply(local_x,  local_y)])

if (rank == 0):
            print("The final product is", local_result, "computed in parallel")



# When I execute using 2 or 3 or 4 cores, the error pops out as below:
# Traceback (most recent call last):
#   File "MatrixProductParallel.py", line 47, in <module>
#    local_x = numpy.zeros(int(local_n))
# ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.