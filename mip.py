import numpy as np
import setup_mip
import simplex
import simplex_unvectorized
import two_phase
import branch_and_cut
import random
import timeit

class InvalidSpeedComparisonParamsException(Exception):
    pass

def solve(filename, num_cutting_planes):
    tableau, artificial_var_cols, intvar_cols, unbound_var_names, name_to_col_map, basic_vars, subtraction_row = setup_mip.setup(filename)
    if len(artificial_var_cols):
        tableau, obj_row, basic_vars = two_phase.solve(tableau, artificial_var_cols, basic_vars, subtraction_row)
    else:
        tableau, obj_row, basic_vars = simplex.solve(tableau, basic_vars, subtraction_row)
    if len(intvar_cols):
        tableau, obj_row, basic_vars = branch_and_cut.solve(tableau, basic_vars, intvar_cols, subtraction_row, obj_row, num_cutting_planes)
    return tableau, obj_row, basic_vars, unbound_var_names, name_to_col_map
        

def compare_speeds(min_num_vars, max_num_vars, reps, ignore_np_warnings, print_reps):
    if min_num_vars <= 0 or max_num_vars < min_num_vars or reps <= 0:
        raise InvalidSpeedComparisonParamsException
    unvec_time = 0
    vec_time = 0
    if ignore_np_warnings:
        np.seterr(all="ignore")
    myval = reps // 10
    for i in range(reps):
        if print_reps and not i % myval:
            print(i)
        curr_size = random.randint(min_num_vars, max_num_vars)
        tableau = np.random.randint(low=0, high=20, size=(curr_size, curr_size))
        tableau = np.append(tableau, np.identity(curr_size), axis = 1) 
        subtraction_row = np.zeros((curr_size * 2))
        subtraction_row[:curr_size] = np.random.randint(low=-20, high=20, size=curr_size)
        basic_vars = np.array(list(range(curr_size,curr_size*2)))

        try:
            start_time = timeit.default_timer()
            tableau, obj_row, basic_vars = simplex_unvectorized.solve(tableau, basic_vars, subtraction_row)
            unvec_time += timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            tableau, obj_row, basic_vars = simplex.solve(tableau, basic_vars, subtraction_row)
            vec_time += timeit.default_timer() - start_time
        except simplex_unvectorized.UnboundedSolutionSetException:
            pass
        
    return unvec_time / reps, vec_time / reps