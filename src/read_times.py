'''
    read_times.py
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

def main():
    # Leemos los tiempos obtenidos
    df = pd.read_pickle('./stats/times.pkl')
    
    # mutual_info.py
    # Entropía
    mi_ent = df.loc[('ent',[2, 10, 25]),('mi', [1000,30000,100000])].copy()
    mi_ent_std = pd.DataFrame(index = mi_ent.index, columns = mi_ent.columns)

    # Calculamos la media y desviación típica
    for i in mi_ent.index: 
        for c in mi_ent.columns:
            mi_ent_std.loc[i, c] = mi_ent.loc[i, c].std()
            mi_ent.loc[i, c] = mi_ent.loc[i, c].mean()
            
    # Imprimimos los resultados (a un .tex y por pantalla)
    print("mutual_info.py\nEntropía\n TIEMPOS MEDIOS")
    print(mi_ent.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/mi_ent.tex', 'w'))
    print(mi_ent, '\n')
    
    print("DESVIACIÓN TÍPICA\n", mi_ent_std)
    print(mi_ent_std.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/mi_ent_std.tex', 'w'))
    print("\n\n")
    
    # Información mutua
    mi_mi = df.loc[('mi',[2, 10, 25]),('mi', [1000,30000,100000])].copy()
    mi_mi_std = pd.DataFrame(index = mi_mi.index, columns = mi_mi.columns)

    # Calculamos la media y desviación típica
    for i in mi_mi.index: 
        for c in mi_mi.columns:
            mi_mi_std.loc[i, c] = mi_mi.loc[i, c].std()
            mi_mi.loc[i, c] = mi_mi.loc[i, c].mean()
       
    # Imprimimos los resultados     
    print("mutual_info.py\nInformación mutua")
    print("TIEMPOS MEDIOS")
    print(mi_mi.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/mi_mi.tex', 'w'))
    
    print(mi_mi, "\n")
    print("DESVIACIÓN TÍPICA\n", mi_mi_std)
    print(mi_mi_std.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/mi_mi_std.tex', 'w'))
    print("\n\n")
    
    
    #----------------------------------------------------------------------------------------------
    # entropy_estimators.py
    # Entropía
    ee_ent = df.loc[('ent',[2, 10, 25]),('ee', [1000,30000,100000])].copy()
    ee_ent_std = pd.DataFrame(index = ee_ent.index, columns = ee_ent.columns)

    # Calculamos la media y desviación típica
    for i in ee_ent.index: 
        for c in ee_ent.columns:
            ee_ent_std.loc[i, c] = ee_ent.loc[i, c].std()
            ee_ent.loc[i, c] = ee_ent.loc[i, c].mean()
           
    # Imprimimos los resultados 
    print("entropy_estimators.py\nEntropía\n TIEMPOS MEDIOS")
    print(ee_ent, '\n')
    print(ee_ent.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/ee_ent.tex', 'w'))
    print("DESVIACIÓN TÍPICA\n", ee_ent_std)
    print(ee_ent_std.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/ee_ent_std.tex', 'w'))
    print("\n\n")
    # Información mutua
    ee_mi = df.loc[('mi',[2, 10, 25]),('ee', [1000,30000,100000])].copy()
    ee_mi_std = pd.DataFrame(index = ee_mi.index, columns = ee_mi.columns)

    # Calculamos la media y desviación típica
    for i in ee_mi.index: 
        for c in ee_mi.columns:
            ee_mi_std.loc[i, c] = ee_mi.loc[i, c].std()
            ee_mi.loc[i, c] = ee_mi.loc[i, c].mean()
      
    # Imprimimos los resultados      
    print("entropy_estimators.py\nInformación mutua")
    print("TIEMPOS MEDIOS")
    print(ee_mi, "\n")
    print(ee_mi.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/ee_mi.tex', 'w'))
    print("DESVIACIÓN TÍPICA\n", ee_mi_std)
    print(ee_mi_std.to_latex(sparsify = False, 
                         index_names = False, 
                         float_format="{:0.4f}".format),
          file=open('./res/ee_mi_std.tex', 'w'))
          
    # ee_mi.loc[('mi', 25),('ee', 100000)] nos permite recuperar un vector concreto
    
if __name__ == "__main__":
    main()
