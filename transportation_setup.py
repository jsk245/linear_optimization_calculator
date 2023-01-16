import numpy as np

class ImproperFileFormatException(Exception):
    pass

class InvalidMethodException(Exception):
    pass

class NotEnoughSupplyException(Exception):
    pass

def get_first_char(my_string):
    return my_string[0]

def setup_problem(filename, method):
    fp = open(filename, "r")
    found_args = list(get_args(fp, 3))
    found_args = [arg.lstrip() for arg in found_args]
    found_args.sort(key=get_first_char)
    fp.close()
    supply = turn_to_int_list(found_args[2][7:])
    demand = turn_to_int_list(found_args[1][7:])
    costs = np.reshape(np.array(turn_to_int_list(found_args[0][6:])), (-1,len(demand)))
    if costs.shape[1] != len(demand):
        raise ImproperFileFormatException
    if min(supply) < 0 or min(demand) < 0:
        raise ImproperFileFormatException
    if not len(supply) or not len(demand):
        raise ImproperFileFormatException

    total_supply = sum(supply)
    total_demand = sum(demand)
    if total_supply < total_demand:
        raise NotEnoughSupplyException
    elif total_supply > total_demand:
        demand.append(total_supply - total_demand)
        costs = np.append(costs, [[0] for _ in range(len(supply))], axis=1)

    if method == 0:
        return min_cost_rule(supply, demand, costs, total_supply)
    elif method == 1:
        return vogel(supply, demand, costs)
    elif method == 2:
        return larson(supply, demand, costs)
    else:
        raise InvalidMethodException

def turn_to_int_list(mylist):
    mylist = mylist.split(",")
    return [int(s.strip()) for s in mylist]

def get_args(fp, num_semicolons):
    count = 0
    buffer = ""
    while count < num_semicolons:
        chunk = fp.read(4096)
        if not chunk:
            raise ImproperFileFormatException
        buffer += chunk
        while count < num_semicolons:
            try:
                chunk, buffer = buffer.split(";", 1)
                yield chunk
                count += 1
            except:
                break

def min_cost_rule(supply, demand, costs, total_supply):
    final_table = np.zeros(costs.shape)
    max_value = np.iinfo(costs.dtype).max
    while total_supply:
        ind = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
        amount_added = min(supply[ind[0]], demand[ind[1]])
        supply[ind[0]] -= amount_added
        demand[ind[1]] -= amount_added
        total_supply -= amount_added
        final_table[ind] = amount_added
        costs[ind] = max_value

    return final_table

def vogel(supply, demand, costs):
    demand_len = len(demand)
    final_table = np.zeros(costs.shape)
    indices_tracked = np.reshape(np.arange(len(supply) * demand_len),(-1, demand_len))
    curr_shape = list(costs.shape)
    while True:
        if curr_shape[0] == 1:
            fill_in_final(indices_tracked, final_table, demand, demand_len, 0)
            break
        elif curr_shape[1] == 1:
            fill_in_final(indices_tracked, final_table, supply, demand_len, 1)
            break
        else:
            col_diffs = costs.max(axis=0) - costs.min(axis=0)
            row_diffs = costs.max(axis=1) - costs.min(axis=1)
            if max(col_diffs) >= max(row_diffs):
                col_ind = np.argmax(col_diffs)
                row_ind = np.argmin(costs[:,col_ind])
            else:
                row_ind = np.argmax(row_diffs)
                col_ind = np.argmin(costs[row_ind,:])
            corresponding_final_ind = indices_tracked[row_ind,col_ind]
            corresponding_final_row_ind = corresponding_final_ind // demand_len
            corresponding_final_col_ind = corresponding_final_ind % demand_len
            
            amount_added = min(supply[corresponding_final_row_ind], 
                                demand[corresponding_final_col_ind])
            supply[corresponding_final_row_ind] -= amount_added
            demand[corresponding_final_col_ind] -= amount_added
            final_table[corresponding_final_row_ind, corresponding_final_col_ind] = amount_added
            if not supply[corresponding_final_row_ind]:
                costs = np.delete(costs, row_ind, 0)
                indices_tracked = np.delete(indices_tracked, row_ind, 0)
                curr_shape[0] -= 1
            else:
                costs = np.delete(costs, col_ind, 1)
                indices_tracked = np.delete(indices_tracked, col_ind, 1)
                curr_shape[1] -= 1
            
    return final_table

def fill_in_final(indices_tracked, final_table, constraints, demand_len, supply_left):
    indices_tracked = indices_tracked.flatten()
    for corresponding_final_ind in indices_tracked:
        corresponding_final_row_ind = corresponding_final_ind // demand_len
        corresponding_final_col_ind = corresponding_final_ind % demand_len
        if supply_left:
            final_table[corresponding_final_row_ind,corresponding_final_col_ind] = constraints[corresponding_final_row_ind]
        else:
            final_table[corresponding_final_row_ind,corresponding_final_col_ind] = constraints[corresponding_final_col_ind]

def larson(supply, demand, costs):
    row_averages = np.mean(costs, axis=1)
    col_averages = np.mean(costs, axis=0)
    costs = costs - row_averages[:,None]
    costs = costs - col_averages[None,:]
    return vogel(supply, demand, costs)