import pandas

if __name__ == '__main__':
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/3_transf_95.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 3 - 95:')
    print(top_10_rows)
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/3_transf_90.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 3 - 90:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/3_transf_75.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 3 - 75:')
    print(top_10_rows)
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/4_transf_95.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 4 - 95:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/4_transf_90.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 4 - 90:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/4_transf_75.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 4 - 75:')
    print(top_10_rows)
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/5_transf_95.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 5 - 95:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/5_transf_90.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 5 - 90:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/5_transf_75.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 5 - 75:')
    print(top_10_rows)
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_3_transf_95.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 3 - 95 mw:')
    print(top_10_rows)
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_3_transf_90.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 3 - 90 mw:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_3_transf_75.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 3 - 75 mw:')
    print(top_10_rows)
    
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_4_transf_95.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 4 - 95 mw:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_4_transf_90.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 4 - 90 mw:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_4_transf_75.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 4 - 75 mw:')
    print(top_10_rows)
    
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_5_transf_95.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 5 - 95 mw:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_5_transf_90.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 5 - 90 mw:')
    print(top_10_rows)
    df = pandas.read_csv('~/work/code/udacity-gym/metrics/new_data/mw_5_transf_75.csv')
    top_10_rows = df.nlargest(10, 'F1 Score')
    print(' 5 - 75 mw:')
    print(top_10_rows)
    
    
    
