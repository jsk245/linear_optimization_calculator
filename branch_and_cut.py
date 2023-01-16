import numpy as np
import heapq
import two_phase
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    tableau: Any=field(compare=False)
    basic_vars: Any=field(compare=False)

class ValidSolutionException(Exception):
    pass

def solve(tableau, basic_vars, intvar_cols, subtraction_row, obj_row, num_cutting_planes):
    pq = []
    curr_best_sol = PrioritizedItem(float('inf'), np.array([]), np.array([]))

    add_to_pq = 1
    for _ in range(num_cutting_planes):
        try:
            tableau, basic_vars = add_cutting_plane(tableau, basic_vars, intvar_cols)
            try:
                obj_row, tableau, basic_vars = dual_simplex(tableau, basic_vars, subtraction_row)
            except two_phase.NoFeasibleSolutionException:
                add_to_pq = 0
                break
        except ValidSolutionException:
            break
    if add_to_pq:
        heapq.heappush(pq, PrioritizedItem(-1 * obj_row[0], np.copy(tableau), np.copy(basic_vars)))

    while len(pq):
        curr = heapq.heappop(pq)
        if curr.priority >= curr_best_sol.priority:
            break
        try:
            leftbranch, left_basic_vars, rightbranch, right_basic_vars = split_node(curr.tableau, curr.basic_vars, intvar_cols)
            do_cuts = 1
            add_to_pq = 1
            try:
                priority, tableau, basic_vars = dual_simplex(leftbranch, left_basic_vars, subtraction_row)
            except two_phase.NoFeasibleSolutionException:
                do_cuts = 0
                add_to_pq = 0
            if do_cuts:
                for _ in range(num_cutting_planes):
                    try:
                        tableau, basic_vars = add_cutting_plane(tableau, basic_vars, intvar_cols)
                        try:
                            priority, tableau, basic_vars = dual_simplex(tableau, basic_vars, subtraction_row)
                        except two_phase.NoFeasibleSolutionException:
                            add_to_pq = 0
                            break
                    except ValidSolutionException:
                        if -1 * priority[0] < curr_best_sol.priority:
                            curr_best_sol = PrioritizedItem(-1 * priority[0], tableau, basic_vars)
                        add_to_pq = 0
                        break
            if add_to_pq:
                heapq.heappush(pq, PrioritizedItem(-1 * priority[0], tableau, basic_vars))

            do_cuts = 1
            add_to_pq = 1
            try:
                priority, tableau, basic_vars = dual_simplex(rightbranch, right_basic_vars, subtraction_row)
            except two_phase.NoFeasibleSolutionException:
                do_cuts = 0
                add_to_pq = 0
            if do_cuts:
                for _ in range(num_cutting_planes):
                    try:
                        tableau, basic_vars = add_cutting_plane(tableau, basic_vars, intvar_cols)
                        try:
                            priority, tableau, basic_vars = dual_simplex(tableau, basic_vars, subtraction_row)
                        except two_phase.NoFeasibleSolutionException:
                            add_to_pq = 0
                            break
                    except ValidSolutionException:
                        if -1 * priority[0] < curr_best_sol.priority:
                            curr_best_sol = PrioritizedItem(-1 * priority[0], tableau, basic_vars)
                        add_to_pq = 0
                        break
            if add_to_pq:
                heapq.heappush(pq, PrioritizedItem(-1 * priority[0], tableau, basic_vars))

        except ValidSolutionException:
            curr_best_sol = curr

    if curr_best_sol.priority == float('inf'):   
        raise two_phase.NoFeasibleSolutionException
    subtraction_row = np.pad(subtraction_row, (0, curr_best_sol.tableau.shape[1] - subtraction_row.shape[0]))
    multiplication_helper = subtraction_row[curr_best_sol.basic_vars]
    obj_row = curr_best_sol.tableau * multiplication_helper[:,None]
    obj_row = np.sum(obj_row, axis=0)  - subtraction_row
    return curr_best_sol.tableau, obj_row, curr_best_sol.basic_vars

def split_node(tableau, basic_vars, intvar_cols):
    intvar_indices = np.where(np.isin(basic_vars, intvar_cols))[0]
    int_vals = tableau[intvar_indices,0]
    if np.array_equal(int_vals, int_vals // 1):
        raise ValidSolutionException

    diffs = int_vals - int_vals // 1
    diffs_ind = np.abs(diffs - 0.5).argmin()
    target_row_ind = intvar_indices[diffs_ind]
    
    new_row = np.copy(tableau[target_row_ind,:])
    leftbranch = tableau
    rightbranch = np.copy(tableau)
    new_row[basic_vars[target_row_ind]] = 0
    new_row[0] = diffs[diffs_ind]
    new_row = new_row * -1
    leftbranch = np.vstack((leftbranch, new_row))

    new_row = new_row * -1
    new_row[0] = new_row[0] - 1
    rightbranch = np.vstack((rightbranch, new_row))

    left_basic_vars = basic_vars
    left_basic_vars = np.append(left_basic_vars, tableau.shape[1])
    right_basic_vars = np.copy(left_basic_vars)

    leftbranch = np.hstack((leftbranch, np.zeros((leftbranch.shape[0], 1), dtype=leftbranch.dtype)))
    leftbranch[-1,-1] = 1
    rightbranch = np.hstack((rightbranch, np.zeros((rightbranch.shape[0], 1), dtype=rightbranch.dtype)))
    rightbranch[-1,-1] = 1

    return leftbranch, left_basic_vars, rightbranch, right_basic_vars


def add_cutting_plane(tableau, basic_vars, intvar_cols):
    intvar_indices = np.where(np.isin(basic_vars, intvar_cols))[0]
    int_vals = tableau[intvar_indices,0]
    if np.array_equal(int_vals, int_vals // 1):
        raise ValidSolutionException

    diffs = int_vals - int_vals // 1
    target_row_ind = intvar_indices[np.abs(diffs - 0.5).argmin()]

    new_row = np.copy(tableau[target_row_ind,:])
    fractional_part = new_row[0] - new_row[0] // 1
    new_row[0] = fractional_part
    fractional_part_helper = fractional_part / (fractional_part - 1)
    new_row[intvar_cols] = new_row[intvar_cols] - new_row[intvar_cols] // 1
    new_row[intvar_cols] = np.where(new_row[intvar_cols] <= fractional_part, new_row[intvar_cols], fractional_part_helper * (1-new_row[intvar_cols]))
    non_intvar_cols = np.where(np.isin(np.arange(0,tableau.shape[1]), intvar_cols, invert=True))[0][1:]
    new_row[non_intvar_cols] = np.where(new_row[non_intvar_cols] >= 0, new_row[non_intvar_cols], fractional_part_helper * new_row[non_intvar_cols])
    new_row = new_row * -1
    
    basic_vars = np.append(basic_vars, tableau.shape[1])
    intvar_cols = np.append(intvar_cols, tableau.shape[1])
    tableau = np.vstack((tableau, new_row))
    tableau = np.hstack((tableau, np.zeros((tableau.shape[0], 1), dtype=tableau.dtype)))
    tableau[-1,-1] = 1

    return tableau, basic_vars
    

def dual_simplex(tableau, basic_vars, subtraction_row):
    while tableau[:,0].min() < 0:
        replaced_row_ind = tableau[:,0].argmin()

        if tableau[replaced_row_ind,1:].min() >= 0:
            raise two_phase.NoFeasibleSolutionException
        subtraction_row = np.pad(subtraction_row, (0, tableau.shape[1] - subtraction_row.shape[0]))
        multiplication_helper = subtraction_row[basic_vars]
        obj_row = tableau[:,1:] * multiplication_helper[:,None]
        obj_row = np.sum(obj_row, axis=0)  - subtraction_row[1:]
        theta_ratios = np.divide(obj_row, tableau[replaced_row_ind,1:], out=np.full(obj_row.shape, -np.inf), where=tableau[replaced_row_ind,1:] < 0)
        ind = theta_ratios.argmax() + 1
        
        basic_vars[replaced_row_ind] = ind
        tableau[replaced_row_ind,:] = tableau[replaced_row_ind,:] / tableau[replaced_row_ind,ind]

        multiplication_col = np.copy(tableau[:,ind])
        multiplication_col[replaced_row_ind] = 0
        tableau = tableau - multiplication_col[:,None] * tableau[replaced_row_ind,:][None,:]

        multiplication_helper[replaced_row_ind] = subtraction_row[ind]
        obj_row = tableau * multiplication_helper[:,None]
        obj_row = np.sum(obj_row, axis=0) - subtraction_row

        tableau = np.where(tableau - tableau // 1 > 1e-7, tableau, tableau // 1)
        tableau = np.where(tableau - tableau // 1 < 1 - 1e-7, tableau, tableau // 1 + 1)
    return obj_row, tableau, basic_vars