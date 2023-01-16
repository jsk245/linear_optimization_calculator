import numpy as np
import simplex

#to avoid division by zero
epsilon = -3 * (10 ** -7)

class NoFeasibleSolutionException(Exception):
    pass

def solve(tableau, artificial_var_cols, basic_vars, subtraction_row):
    tableau, basic_vars = phase1(tableau, artificial_var_cols, basic_vars, subtraction_row)
    if len(set(basic_vars) & artificial_var_cols):
        return phase2(tableau, basic_vars, subtraction_row, artificial_var_cols)
    else:
        return simplex.solve(tableau, basic_vars, subtraction_row)

def phase1(tableau, artificial_var_cols, basic_vars, subtraction_row):
    phase1_sub_row = np.zeros(subtraction_row.shape)
    np.put(phase1_sub_row, list(artificial_var_cols), -1)

    try:
        tableau, _, basic_vars = simplex.solve(tableau, basic_vars, phase1_sub_row)
    except simplex.UnboundedSolutionSetException:
        raise NoFeasibleSolutionException
    tableau[:,list(artificial_var_cols - set(basic_vars))] = 0

    return tableau, basic_vars

def phase2(tableau, basic_vars, subtraction_row, artificial_var_cols):
    subtraction_row = np.array(subtraction_row)
    multiplication_helper = subtraction_row[basic_vars][:,None]
    basic_vars = np.array(basic_vars)
    artificial_var_col_indices = np.where(np.isin(basic_vars, list(artificial_var_cols)))[0]
    obj_row = np.sum(tableau * multiplication_helper, axis=0) - subtraction_row
    mymin = obj_row[1:].min()

    while mymin < 0:
        ind = np.argmin(obj_row[1:], axis=None) + 1
        if tableau[:,ind].max() <= 0:
            raise UnboundedSolutionSetException
        curr_col = tableau[:,ind]
        if artificial_var_col_indices.shape[0] and curr_col[artificial_var_col_indices].min() < 0:
            replaced_row_ind = curr_col[artificial_var_col_indices].argmin()
        else:
            # TODO: find how out to normalize this/ if normalization is necessary
            theta_ratios = tableau[:,0] / (curr_col - epsilon)
            replaced_row_ind = np.where(curr_col > 0, theta_ratios, np.inf).argmin()
        basic_vars[replaced_row_ind] = ind
        tableau[replaced_row_ind,:] = tableau[replaced_row_ind,:] / tableau[replaced_row_ind,ind]

        multiplication_col = np.copy(tableau[:,ind])
        multiplication_col[replaced_row_ind] = 0
        tableau = tableau - multiplication_col[:,None] * tableau[replaced_row_ind,:][None,:]

        multiplication_helper[replaced_row_ind] = subtraction_row[ind]
        artificial_var_col_indices = np.where(np.isin(basic_vars, list(artificial_var_cols)))[0]
        obj_row = np.sum(tableau * multiplication_helper, axis=0) - subtraction_row
        mymin = obj_row[1:].min()
    
    return tableau, obj_row, basic_vars