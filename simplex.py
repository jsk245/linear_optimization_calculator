import numpy as np

class UnboundedSolutionSetException(Exception):
    pass

def solve(tableau, basic_vars, subtraction_row):
    subtraction_row = np.array(subtraction_row)
    multiplication_helper = subtraction_row[basic_vars][:,None]
    basic_vars = np.array(basic_vars)
    obj_row = np.sum(tableau * multiplication_helper, axis=0) - subtraction_row
    mymin = obj_row[1:].min()

    while mymin < 0:
        ind = np.argmin(obj_row[1:], axis=None) + 1
        if tableau[:,ind].max() <= 0:
            raise UnboundedSolutionSetException
        curr_col = tableau[:,ind]
        # TODO: find how out to normalize this/ if normalization is necessary
        theta_ratios = tableau[:,0] / np.where(curr_col != 0, curr_col, np.inf)
        replaced_row_ind = np.where(curr_col > 0, theta_ratios, np.inf).argmin()
        basic_vars[replaced_row_ind] = ind
        tableau[replaced_row_ind,:] = tableau[replaced_row_ind,:] / tableau[replaced_row_ind,ind]

        multiplication_col = np.copy(tableau[:,ind])
        multiplication_col[replaced_row_ind] = 0
        tableau = tableau - multiplication_col[:,None] * tableau[replaced_row_ind,:][None,:]

        multiplication_helper[replaced_row_ind] = subtraction_row[ind]
        obj_row = np.sum(tableau * multiplication_helper, axis=0) - subtraction_row
        mymin = obj_row[1:].min()
    
    return tableau, obj_row, basic_vars