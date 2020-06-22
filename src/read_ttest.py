import pandas as pd
from decimal import Decimal

def main():
    df = pd.read_pickle('./stats/ttest.pkl')
    ds = df.index.get_level_values(level=0).unique()
    ns = df.columns.get_level_values(level=1).unique()
    
    # Guardamos los resultados del ttest para la entropía
    ent = df['ent']
   
    print(ent.to_latex(sparsify = True, 
                       index_names = True, 
                       float_format="{:0.4e}".format), 
          file=open('./res/ttest_ent.tex', 'w'))
    
    # Guardamos los resultados indicando cuándo se rechaza y cuándo se acepta
    df2 = pd.DataFrame(index = ds, columns = ns)
    
    for d in ds:
        for n in ns:
            df2.loc[d, n] = "Rechazo" if ent.loc[(d,'p'), n] <= 0.05 \
                                      else "No rechazo"    
                                      
    print(df2.to_latex(sparsify = True, 
                       index_names = True),
          file=open('./res/ttest_ent_reg.tex', 'w'))    
                                      
    
    # Guardamos los resultados del ttest para la información mutua
    mi = df['mi']
   
    print(mi.to_latex(sparsify = True, 
                      index_names = True, 
                      float_format="{:.4e}".format),
          file=open('./res/ttest_mi.tex', 'w'))
    
    # Guardamos los resultados indicando cuándo se rechaza y cuándo se acepta
    df2 = pd.DataFrame(index = ds, columns = ns)
    
    for d in ds:
        for n in ns:
            df2.loc[d, n] = "Rechazo" if mi.loc[(d,'p'), n] <= 0.05 \
                                      else "No rechazo"    
                                      
    print(df2.to_latex(sparsify = True, 
                       index_names = True),
          file=open('./res/ttest_mi_reg.tex', 'w'))    
    

if __name__ == "__main__":
    main()
