# =============================================================================
# ODEV 
# 1. Dataset yüklenimi
# 2. Bağımlı değişkeni ve diğer değişkenlerin ayrımı 
# 3. Kategorik değerlerin sayısal değerlere döndürüp yeni dataset hazırlanması
# 4. Test seti ile dataset ayrımı 
# 5. Model ile tahmin etme 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("B:\GitHUB\BTK_Akademi_AI\Datasetler/odev_tenis.csv")

