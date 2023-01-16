import numpy as np

#to avoid division by zero
epsilon = -3 * (10 ** -7)

class UnboundedSolutionSetException(Exception):
    pass

def solve(tableau, basic_vars, subtraction_row):
    subtraction_row = np.array(subtraction_row)
    basic_vars = np.array(basic_vars)
    curr_cols = tableau.shape[1]
    curr_rows = tableau.shape[0]
    obj_row = np.zeros((curr_cols))
    for j in range(curr_cols):
        for i in range(curr_rows):
            obj_row[j] += tableau[i,j] * subtraction_row[basic_vars[i]]
        obj_row[j] -= subtraction_row[j]
    mymin = obj_row[1:].min()

    while mymin < 0:
        ind = -1
        for j in range(1,curr_cols):
            if obj_row[j] == mymin:
                ind = j
                break
        pos_val_present = 0
        for i in range(curr_rows):
            if tableau[i,ind] > 0:
                pos_val_present = 1
                break
        if not pos_val_present:
            raise UnboundedSolutionSetException
        replaced_row_ind = -1
        smallest_theta_ratio = float('inf')
        for i in range(curr_rows):
            if tableau[i,ind] > 0:
                theta_ratio = tableau[i,0] / tableau[i,ind]
                if theta_ratio < smallest_theta_ratio:
                    smallest_theta_ratio = theta_ratio
                    replaced_row_ind = i
        basic_vars[replaced_row_ind] = ind
        tempval = tableau[replaced_row_ind, ind]
        for j in range(curr_cols):
            tableau[replaced_row_ind, j] = tableau[replaced_row_ind, j] / tempval

        for i in range(curr_rows):
            if i != replaced_row_ind:
                tempval = tableau[i,ind]
                for j in range(curr_cols):
                    tableau[i,j] = tableau[i,j] - tempval * tableau[replaced_row_ind,j]

        obj_row = np.zeros((curr_cols))
        for j in range(curr_cols):
            for i in range(curr_rows):
                obj_row[j] += tableau[i,j] * subtraction_row[basic_vars[i]]
            obj_row[j] -= subtraction_row[j]
        mymin = obj_row[1:].min()
    
    return tableau, obj_row, basic_vars
    