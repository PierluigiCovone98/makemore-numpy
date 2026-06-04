"""Experimenting how broadcasting works in numpy.
"""

import numpy as np


def main():

    # === Experiment Section ===
    
    # === It's completely random matrix (but easy to manage by hand).
    M = np.arange(12).reshape(4,3)
    
    '''
    Documentation says:
        "By default, these operations (sum, in our case of study) 
         apply to the array as though it were a list of numbers, 
         regardless of its shape. 
         However, by specifying the axis parameter you can apply 
         an operation along the specified axis of an array [...]"

    We know that:
        matrix.shape    =>  (x,y)   where:
                                        x := number of rows
                                        y := number of columnes
    '''

    #   Returns the sum of all element in the matrix.
    M_sum = M.sum()


    '''
    So, what does it meas:
        "[...] by specifying the axes parameter you can
         apply an operation along the specifies axis of
         an array"?
    
    Well, I expect that I specify the axis that I want to
    use as "iterator". Let me explain it better by distinguish
    two cases:

        a.  If I say:    
                sum(axis=0)
            then I want to use rows as iterators.
            
            The final result is a vector where each j-th
            element (where j denotes a column) of the new
            vector, is the sum of the j-th element of each i-th row.

        a.  If I say:    
                sum(axis=1)
            then I want to use columns as iterators.
            
            The final result is a column vector (vector of vectors
            of one element), where each i-th element (where i denotes a row)
            of the new columnv ector, is the sum of the i-th element of each 
            j-th column.
    '''
    M_sum_by_columns = M.sum(axis=0)
    M_sum_by_rows = M.sum(axis=1)
    
    '''
    Notice that, without specify the argument "keepdim=True", even the "column vector"
    become a simple vector. And I do not want it.

    This means that, when I need to work with column vectors, I need to keep the dimension.
    '''

    M_sum_by_columns = M.sum(axis=0)
    M_sum_by_rows = M.sum(axis=1)

    # === Printing Section ====
    
    # === Matrix infos
    print(f"M = {M}\n")
    print(f"M.shape() = {M.shape}\n")

    # === Operation results
    print(f"M.sum() = {M_sum}\n")
    print("Without \"keepdim=True\":\n")
    print(f"\tM.sum(axis=0) [sum-by-column]   = {M_sum_by_columns}")
    print(f"\tM.sum(axis=1) [sum-by-rows]     = {M_sum_by_rows}\n")


    '''
    Notice that, without specify the argument "keepdim=True", even the "column vector"
    become a simple vector. And I do not want it.
    '''

    M_sum_by_columns = M.sum(axis=0, keepdims=True)
    M_sum_by_rows = M.sum(axis=1, keepdims=True)
    print("With \"keepdim=True\":")
    print(f"\tM.sum(axis=0) [sum-by-column]   = {M_sum_by_columns}")
    print(f"\tM.sum(axis=1) [sum-by-rows]     = \n{M_sum_by_rows}\n")


    '''
    Why I do not want to lose the "column vector"?
    Well, because to actually perform the broadcasting operation the vector is "filled"
    such that it matches the matrix with wich is "associated" for the operation (both if
    it is a simple vector or a column vector), if I lose the "column vector-property" then
    it will result in the wrong "padding" (and then in the wrong operation).
    '''

    M1 = M_sum_by_rows
    M2 = M_sum_by_columns

    MxM1 = M*M1
    print(f"M x M1  =\n{MxM1}\n")        # Expected = [ [0, 3, 6], [36, 48, 60], [...], [270, 300, 330] ]

    MxM2 = M*M2
    print(f"M x M2  =\n{MxM2}")        # Expected = [ [0, 22, 52], [54, 88, 150], [...], [162, 220, 286] ]


if __name__ == '__main__':
    main()
