# Produce data for 2 parameter tikz plot 
import re, glob
import numpy as np


def is_valid_template(path):
    '''* for all'''
    subs = path.split('*')
    return all(len(sub) > 0 for sub in subs)


def get_param_value(path, name):
    '''name_L{*}'''
    pattern = re.compile(rf'\w+{name}[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
    try: 
        matched, = pattern.findall(path)
    except ValueError:
        return None
    
    num = matched[matched.index(name)+len(name):]
    return float(num)


def get_data(path, col_name, normalize=False):
    '''Extract column'''
    with open(path) as lines:
        header = next(lines).strip().split()
        try:
            col = header.index(col_name)
        except ValueError:
            print(col_name, header)
            exit()
        data = [float(line.strip().split()[col]) for line in lines]
        if normalize:
            data = np.array(data)
            data /= data[0]
        
        return data

    
def contains(iterable, num, tol=1E-14):
    '''Is num in iterable?'''
    for i, item in enumerate(iterable):
        if abs(item-num) < tol:
            return i
    return None


def extract_data(template, variable_ranges, xcol, ycol, normalize, reverse_sort_x=False):
    '''Look for data in tables and align'''
    assert len(variable_ranges) == template.count('*')

    paths = dict()
    variables = tuple(variable_ranges.keys())
    for path in glob.glob(template):
        print(path)
        if 'tikz' in path: continue
        
        key = ()
        is_valid = True
        for variable in variables:
            value = get_param_value(path, variable)
            if value is None:
                break
            variable_range = variable_ranges[variable]
            # print('->', variable, value, variable_range)
            is_valid = not variable_range or contains(variable_range, value) is not None
            key = key + (value, )
            if not is_valid:
                break
        # Only valid can asssign
        if is_valid:
            paths[key] = path

        # print(path, key, variables, is_valid)
            
    # Want to align data for union of all indep
    X = set()
    for key in paths:
        path = paths[key]
        X.update(get_data(path, xcol))
    X = np.array(sorted(X, reverse=reverse_sort_x))

    aligned_y = dict()
    # Now we insert NaNs as indicator of missing data
    for key in paths:
        path = paths[key]
        x = get_data(path, xcol)
        # Make room for larger
        Y = np.nan*np.ones_like(X)
        y = get_data(path, ycol, normalize)
        for xi, yi in zip(x, y):
            # Look for where to insert
            idx = contains(X, xi)
            if idx is not None:
                Y[idx] = yi
        # Final data
        aligned_y[key] = Y

    return X, aligned_y, variables

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, os
    import numpy as np
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Where to get data
    parser.add_argument('directory', type=str, help='Directory with files')
    #parser.add_argument('-template', type=str, help='Template matchin txt files',
    #                    default='./steklov_2d/cont_u_mean/square/has_hole0/eigenvalues_EIGW_RTOL0.0001_WHICHcont_u_mean_NREFS5_REFSEED0.5_DIRICHLET_WHICHstrong_SHAPEsquare_OUTER_RADIUS*_INNER_RADIUS*_HAS_HOLE0.txt')

    parser.add_argument('-analysis', type=str, choices=('eigs', 'iters', 'sanity'), default='eigs')
    parser.add_argument('-bcs', type=str, default='DDDD')    
    parser.add_argument('-K_values', type=float, nargs='+', help='Select values for K',
                        # default=[1E0, 1E-2, 1E-4, 1E-8]
                        default=[1E0, 1E-1, 1E-2, 1E-3]
                        )
    parser.add_argument('-mu_values', type=float, nargs='+', help='Select values for mu',
                        default=[1E0])
    # How to get data
    parser.add_argument('-x_column', type=str, default='ndofs', help='Name of independent variable')
    # parser.add_argument('-y_column', type=str, default='eigw_min', help='Name of dependent variable')

    parser.add_argument('-y_column', type=str, default='', help='Name of dependent variable')        

    parser.add_argument('-normalize', type=int, default=0, choices=(0, 1))

    args, _ = parser.parse_known_args()

    K_values, mu_values = args.K_values, args.mu_values

    reverse_sort_x = {'h': True,
                      'ndofs': False}[args.x_column]

    Velm = 'RT_1'
    Qelm = 'DG_0'
    pc_type = 'rieszk'
    
    comparison = {}
    for bcs in args.bcs:
        print(bcs)
        # template = f'{args.analysis}/CG_2-CG_1/nitsche/radius100.0/results_LENGTH*_RADIUS100.0_SYMGRAD1_MU*_VELMCG_2_QELMCG_1_BCS{bcs}_NAVIER_GAMMA1_VBC_HOWnitsche_KSP_TYPEminres_PC_TYPElu.txt'
        # template = f'{args.analysis}/CG_2-CG_1/dirichlet/radiusinf/results_LENGTH*_RADIUSinf_SYMGRAD1_MU*_VELMCG_2_QELMCG_1_BCS{bcs}_NAVIER_GAMMA1_VBC_HOWdirichlet_KSP_TYPEminres_PC_TYPElu.txt'
        template = f'{args.analysis}/{Velm}-{Qelm}/radiusinf/results_LENGTH1_RADIUS*_MESHGENnative_K*_MMS_KINDbasic_VELM{Velm}_QELM{Qelm}_BCS{args.bcs}_KSP_TYPEminres_PC_TYPE{pc_type}.txt'
        assert is_valid_template(template)

        if args.y_column:
            y_column = args.y_column
        else:
            y_column = {'iters': 'niters',
                        'eigs': 'cond',
                        'ieigs': 'cond'}[args.analysis]
        print(template)
        X, aligned_y, variables = extract_data(template=os.path.join(args.directory, template),
                                               variable_ranges={'LENGTH': mu_values, 'K': K_values},
                                               xcol=args.x_column,
                                               ycol=y_column,
                                               normalize=args.normalize,
                                               reverse_sort_x=reverse_sort_x)
        # assert aligned_y
        # Now for printing in tikz
        data = np.column_stack([X] + [aligned_y[key] for key in aligned_y])
        header = ' '.join(['x'] + [f'{variables[0]}{key[0]:.4E}_{variables[1]}{key[1]:.4E}' for key in aligned_y])

        tikz_path, ext = os.path.splitext(template.replace('*', 'varied'))
        tikz_path = '_'.join([tikz_path, 'paramRobust',
                              f'x{args.x_column.upper()}', f'y{y_column.upper()}', f'NORMALIZED{args.normalize}'
                              'tikz'])
        tikz_path = f'{tikz_path}.txt'

        tikz_path = os.path.join(args.directory, tikz_path)
        with open(tikz_path, 'w') as out:
            out.write('%s\n' % header)
            np.savetxt(out, data)

        # Self inspection
        prev = None
        for key in sorted(aligned_y):
            print(key, '->', aligned_y[key])

        import tabulate
        headers = sorted(aligned_y)

        keys = sorted(tuple(aligned_y.keys()), reverse=False)
        table = np.vstack([key for key in keys])

        ultima = np.array([aligned_y[key][np.where(~np.isnan(aligned_y[key]))[0][-1]] for key in keys])
        disp_ultima = np.where(np.abs(ultima) > 50, np.round(ultima,0), np.round(ultima, 2))
        
        if all(len(aligned_y[key]) > 3 for key in keys):
            penultima = np.array([aligned_y[key][np.where(~np.isnan(aligned_y[key]))[0][-2]] for key in keys])
            rel = np.abs(ultima-penultima)/ultima

            table = np.c_[table, disp_ultima, np.abs(ultima-penultima), rel]

            headers = variables + ('data', 'diff', 'rel')
            print()
            print(tabulate.tabulate(table, headers=headers))
            print()
        else:
            table = np.c_[table, disp_ultima]

            headers = variables + ('data', )
            print()
            print(tabulate.tabulate(table, headers=headers))
            print()            

        print(os.path.abspath(tikz_path))

        comparison[bcs] = disp_ultima

if len(comparison) > 1:
    _, = set(map(len, comparison.values()))
    
    legend = np.vstack([key for key in keys])
    table = np.column_stack([comparison[bc] for bc in args.bcs])
    table = np.c_[legend, table]

    headers = variables + tuple(args.bcs)
    print()
    print(tabulate.tabulate(table, headers=headers, tablefmt='latex'))
    print()
        
