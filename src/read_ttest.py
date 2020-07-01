'''
    read_ttest.py
    Copyright (C) 2020  Sofía Almeida

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
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
