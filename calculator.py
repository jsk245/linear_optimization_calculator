import typer
import transportation_setup
import mip
import numpy as np

app = typer.Typer()

@app.command()
def setup_transport_problem(filename: str,
                            method: int = typer.Option(2, help="0 = Min Cost Rule, 1 = Vogel's Method, 2 = Larson's Method"), 
                            outfile: str = typer.Option(None, help="name of file to print final table to"),
                            include_brackets: bool = typer.Option(True, help="set to False to only print numbers separated by tabs")):
    """
    returns a table setup to solve a transportation problem
    """
    table = transportation_setup.setup_problem(filename, method)
    print_table(table, outfile, include_brackets)

@app.command()
def solve_mip(filename: str,
                num_cutting_planes: int = typer.Option(3, help="num of cutting planes to use before branching"),
                outfile: str = typer.Option(None,help="name of file to print final table to"),
                include_brackets: bool = typer.Option(True, help="set to False to only print numbers separated by tabs")):
    """
    returns a table setup to solve a transportation problem
    """
    tableau, obj_row, basic_vars, unbound_var_names, name_to_col_map = mip.solve(filename, num_cutting_planes)
    if outfile:
        out = open(outfile, 'w')
    else:
        out = None
    print("Optimal solution:\n{0}\n".format(obj_row[0]), file=out)
    print_var_vals(tableau,  basic_vars, unbound_var_names, name_to_col_map, out)
    print("Variable names and their corresponding column numbers (1-indexed):\n{0}\n".format(name_to_col_map), file=out)
    print("Tableau:", file=out)
    if outfile:
        out.close()
    print_table(tableau, outfile, include_brackets, mode='a')

@app.command()
def compare_simplex_speeds(min_num_vars: int,
                            max_num_vars: int,
                            num_repeats: int,
                            ignore_np_warnings: bool = typer.Option(True, help="set to False to print numpy error messages (usually fp errors)"),
                            print_reps: bool = typer.Option(False, help="set to True to print progress every num_repeats/10 tests")):
    """
    returns the average time to solve num_repeats randomly generated lp problems with 0 < min_num_vars <= max_num_vars variables
    """
    unvectorized_time, vectorized_time = mip.compare_speeds(min_num_vars, max_num_vars, num_repeats, ignore_np_warnings, print_reps)
    print("Average Unvectorized Time:", unvectorized_time)
    print("Average Vectorized Time:", vectorized_time)
    print("The vectorized time is approximately", unvectorized_time / vectorized_time, "times faster")


def print_table(table, outfile, include_brackets, mode='w'):
    if outfile:
        out = open(outfile, mode)
        if include_brackets:
            print(table, file=out)
        else:
            for i in range(table.shape[0]):
                print('\t'.join(map(str, table[i])), file=out)
        out.close()
    else:
        if include_brackets:
            print(table)
        else:
            for i in range(table.shape[0]):
                print('\t'.join(map(str, table[i])))

def print_var_vals(tableau,  basic_vars, unbound_var_names, name_to_col_map, out):
    print("Values of variables:", file=out)
    for name in name_to_col_map:
        ind = np.where(np.isin(basic_vars, name_to_col_map[name]))[0]
        val = 0
        if len(ind):
            val = tableau[ind,0][0]
        if name in unbound_var_names:
            ind = np.where(np.isin(basic_vars, name_to_col_map[name]+1))[0]
            if len(ind):
                val -= tableau[ind,0][0]
        print("{0}: {1}".format(name, val), file=out)
    print("", file=out)

if __name__ == "__main__":
    app()