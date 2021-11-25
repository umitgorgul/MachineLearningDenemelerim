import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var

tiklamalar = [0] * d #o ana kadarki tıklamalar
toplam = 0 # toplam odul
secilenler = []
birler = [0] * d
sifirlar = [0] * d


for n in range(1,N):
    ad = 0 #seçilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate (birler[i] + 1, sifirlar[i] + 1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
            
    secilenler.append(ad)
    
    tiklamalar[ad] = tiklamalar[ad]+ 1
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    if odul == 1:
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad] = sifirlar[ad]+1
        

    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

plt.hist(secilenler)
plt.show()
